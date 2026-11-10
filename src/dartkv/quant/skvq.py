"""SKVQ channel-group clipped dynamic quantization (official port).

Op-for-op port of the fake-quant path in external/SKVQ/KV_process.py
SKVQuantProcessor.quant_pytorch (arXiv 2405.06219). Math is replicated in
the INPUT dtype (no fp32 upcast) so outputs match the official engine
bit-for-bit on bf16 models:

  per group of channels (group = contiguous chunk of the flat H*D axis of
  ONE token, or a calibration-derived variable-size cluster after reorder):
      gmin, gmax = aminmax(group)
      gmin *= clip, gmax *= clip            # unified clipping (0.91-0.97)
      zp = gmin
      scale = clamp((gmax - gmin)/max_int, min=1e-5)
      q = clamp(round((x - zp)/scale), 0, max_int)
      x_hat = q.to(dtype) * scale + zp      # fake quant (dequant inline)
  optional fp8: scale/zp stored e4m3 (cast AFTER q, BEFORE dequant)
  fractional bits: 1.5 -> max_int = 2 (three levels)

Group bounds are a [n_groups+1] int tensor over the (possibly reordered)
flat channel axis; without reorder use arange(0, hidden+1, gsize).
"""
from __future__ import annotations

import torch


def max_levels(bits: float) -> int:
    """SKVQ level count: 2^b - 1 for integer b; 1.5 bits -> 3 levels."""
    if bits == 16:
        return None
    if bits == int(bits):
        return (1 << int(bits)) - 1
    assert bits == 1.5, f"unsupported fractional bits {bits}"
    return 2


def group_bounds(hidden: int, group_size: int, device=None) -> torch.Tensor:
    """Contiguous equal-size bounds [0, g, 2g, ..., hidden] (no reorder)."""
    assert hidden % group_size == 0, (hidden, group_size)
    return torch.arange(0, hidden + 1, group_size, device=device)


def quant_dequant_channels(
    x: torch.Tensor,
    bits: float,
    bounds: torch.Tensor,
    clip: float = 1.0,
    reorder_idx: torch.Tensor | None = None,
    fp8: bool = False,
) -> torch.Tensor:
    """Fake-quantize x along the last (flat channel) axis, per token.

    x:        [..., hidden] (rows are tokens; channel = h*D + d, HF order)
    bounds:   [n_groups+1] group start/end on the REORDERED axis
    clip:     unified clipping factor (official per-layer `clipping[i]`)
    reorder_idx: [hidden] calibration channel permutation (or None)
    Returns dequantized x, original shape/dtype.
    """
    if bits == 16 or x.shape[-1] == 0:
        return x
    dtype = x.dtype
    max_int = max_levels(bits)
    if reorder_idx is not None:
        reorder_idx = reorder_idx.to(x.device).long()
        x = x[..., reorder_idx]
    out = torch.empty_like(x)
    for g0, g1 in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
        t = x[..., g0:g1]
        gmin, gmax = t.aminmax(dim=-1, keepdim=True)
        gmin = gmin * clip
        gmax = gmax * clip
        zp = gmin
        scale = torch.clamp((gmax - gmin) / max_int, min=1e-5)
        q = t.sub(zp).div(scale).clamp(0, max_int).round()
        if fp8:
            scale = scale.to(torch.float8_e4m3fn).to(dtype)
            zp = zp.to(torch.float8_e4m3fn).to(dtype)
        out[..., g0:g1] = q.to(dtype).mul(scale).add(zp)
    if reorder_idx is not None:
        out = out.gather(-1, reorder_idx.argsort(-1).expand(out.shape))
    return out
