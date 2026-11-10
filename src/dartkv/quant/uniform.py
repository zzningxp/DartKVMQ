"""Asymmetric uniform quantization primitives (shared by all methods).

Conventions (KVarN / RAQ-MLA Eq.15 / KIVI compatible):
    s = (max - min) / (2^b - 1)
    q = clamp(round((x - min) / s), 0, 2^b - 1)
    x_hat = s * q + min

All functions are dimension-generic: pass `dim` along which each group gets
its own (scale, min). Group size g means x is reshaped so each contiguous
chunk of g along `dim` shares (s, z).
"""
from __future__ import annotations

import torch


def _group_view(x: torch.Tensor, dim: int, group_size: int | None):
    """Return a view of x where the group axis is -1 and dim moved to the end.

    If group_size is None, the whole `dim` is one group (dummy group axis
    of size 1, squeezed again by the caller on return).
    """
    x = x.transpose(dim, -1)
    if group_size is None:
        return x, x.reshape(*x.shape[:-1], 1, x.shape[-1])
    n = x.shape[-1]
    assert n % group_size == 0, (
        f"dim size {n} not divisible by group_size {group_size}; pad first"
    )
    return x, x.reshape(*x.shape[:-1], n // group_size, group_size)


def quantize_asym(
    x: torch.Tensor,
    bits: int,
    dim: int = -1,
    group_size: int | None = None,
    clip_percentile: float | None = None,
    range_quantile: float | None = None,
    eps: float = 1e-10,
):
    """Quantize along `dim` in groups of `group_size` (None = whole dim).

    clip_percentile: if set (e.g. 0.94), clamp each group to +-[p-th percentile
    of |x|] before computing (min, max)  (RAQ-MLA latent clipping, symmetric).

    range_quantile: if set (e.g. 0.005), compute the asymmetric range from
    signed percentiles [q, 1-q] instead of (min, max); values outside get
    clamped at round time (official KVarN KVARN_RTN_QUANTILE semantics —
    sacrifices outliers for finer bulk resolution; critical for 2-bit on
    outlier-heavy K like Qwen3's).

    Returns (q, scale, zero) where dequant = scale * q + zero broadcast over
    the group. With group_size=g: shapes (..., n/g, g) / (..., n/g, 1) /
    (..., n/g, 1). With group_size=None: q has x's shape; scale/zero are
    (..., 1) placed at `dim`, broadcast over the whole quantized axis.
    """
    assert bits >= 1 and bits <= 16
    levels = 2**bits - 1
    x_t, g = _group_view(x, dim, group_size)

    if clip_percentile is not None:
        tau = torch.quantile(
            g.abs().float(), clip_percentile, dim=-1, keepdim=True
        ).clamp_min(eps)
        g = g.clamp(-tau, tau)

    gf = g.float()
    if range_quantile:
        gmin = torch.quantile(gf, range_quantile, dim=-1, keepdim=True)
        gmax = torch.quantile(gf, 1.0 - range_quantile, dim=-1, keepdim=True)
    else:
        gmin = gf.amin(dim=-1, keepdim=True)
        gmax = gf.amax(dim=-1, keepdim=True)
    scale = ((gmax - gmin) / levels).clamp_min(eps)
    zero = gmin  # asymmetric: zero-point == min (dequant offset)
    q = torch.clamp(((gf - zero) / scale).round(), 0, levels)
    if group_size is None:
        # drop the dummy group axis: (..., 1, n) -> x's shape
        q = q.reshape(x_t.shape).transpose(dim, -1)
        scale = scale.reshape(*x_t.shape[:-1], 1).transpose(dim, -1)
        zero = zero.reshape(*x_t.shape[:-1], 1).transpose(dim, -1)
    return q, scale, zero


def dequantize_asym(
    q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor
) -> torch.Tensor:
    return (q * scale + zero).to(scale.dtype if scale.is_floating_point else torch.float32)


def quantize_dequantize(
    x: torch.Tensor,
    bits: int,
    dim: int = -1,
    group_size: int | None = None,
    clip_percentile: float | None = None,
) -> torch.Tensor:
    """Simulated (fake) quantization; returns x_hat with original shape/dtype."""
    out_dtype = x.dtype
    moved = x.float().transpose(dim, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    gs = group_size if group_size is not None else flat.shape[-1]
    n = flat.shape[-1]
    pad = (-n) % gs
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    grouped = flat.reshape(-1, gs)
    q, s, z = quantize_asym(grouped, bits, dim=-1, group_size=None,
                            clip_percentile=clip_percentile)
    dq = dequantize_asym(q, s, z).reshape(flat.shape)
    dq = dq[..., :n]
    dq = dq.reshape(moved.shape).transpose(dim, -1)
    return dq.to(out_dtype)


def pack_int(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned int tensor (values < 2^bits) into uint8 along last dim.

    Supports bits in {1,2,4,8}. Returned last dim = ceil(n * bits / 8).
    """
    assert bits in (1, 2, 4, 8)
    n = q.shape[-1]
    if bits == 8:
        return q.to(torch.uint8)
    per_byte = 8 // bits
    pad = (-n) % per_byte
    if pad:
        q = torch.nn.functional.pad(q, (0, pad))
    q = q.reshape(*q.shape[:-1], -1, per_byte).to(torch.uint8)
    out = torch.zeros(q.shape[:-1], dtype=torch.uint8, device=q.device)
    for i in range(per_byte):
        out |= q[..., i] << (bits * i)
    return out


def unpack_int(packed: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    assert bits in (1, 2, 4, 8)
    if bits == 8:
        return packed[..., :n].int() if n is not None else packed.int()
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    outs = []
    total = packed.shape[-1] * per_byte
    for i in range(per_byte):
        outs.append(((packed.int() >> (bits * i)) & mask))
    out = torch.stack(outs, dim=-1).reshape(*packed.shape[:-1], total)
    return out[..., :n] if n is not None else out


def effective_bits(bits: int, group_size: int, meta_bits: int = 32) -> float:
    """Effective bits/element incl. per-group metadata (scale+zero).

    meta_bits: storage width of (scale, zero) pair combined in bits
    (e.g. 2 x FP16 = 32; KVarN: 2 x FP8 scale + FP16 zp = 32 too).
    """
    return bits + meta_bits / group_size
