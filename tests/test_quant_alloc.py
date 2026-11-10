"""Unit tests for quant primitives + MCKP. CPU-only, no model download."""
import math
import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dartkv.quant.uniform import (quantize_asym, dequantize_asym,
                                  quantize_dequantize, pack_int, unpack_int,
                                  effective_bits)
from dartkv.quant.hadamard import hadamard_matrix, hadamard
from dartkv.quant.varn import variance_normalize, imbalance
from dartkv.alloc.mckp import mckp_lagrangian, bennett_reverse_waterfilling


def test_asym_roundtrip():
    torch.manual_seed(0)
    x = torch.randn(4, 128) * 3
    for bits in (2, 4, 8):
        q, s, z = quantize_asym(x, bits, dim=-1, group_size=64)
        assert q.max() <= 2**bits - 1 and q.min() >= 0
        xdq = dequantize_asym(q, s, z)
        xg = x.reshape(x.shape[0], x.shape[1] // 64, 64)  # grouped view
        step = s.expand_as(q)
        err = (xg - xdq).abs()
        assert (err <= 0.501 * step).all(), f"{bits}: max err {err.max()} vs step {step.max()}"


def test_fake_quant_shape_and_dtype():
    x = torch.randn(2, 5, 128).half()
    xq = quantize_dequantize(x, 2, dim=-1, group_size=128)
    assert xq.shape == x.shape and xq.dtype == x.dtype
    # non-divisible dim padding path
    x2 = torch.randn(3, 100)
    x2q = quantize_dequantize(x2, 4, dim=-1, group_size=32)
    assert x2q.shape == x2.shape


def test_clipping_reduces_range():
    x = torch.randn(256)
    x[0] = 100.0  # outlier
    xdq = quantize_dequantize(x, 2, dim=-1, group_size=128, clip_percentile=0.94)
    step = (xdq.max() - xdq.min()) / 3
    # with clipping, the bulk should quantize finer than without
    xdq_nc = quantize_dequantize(x, 2, dim=-1, group_size=128)
    bulk_err = (x[1:] - xdq[1:]).abs().mean()
    bulk_err_nc = (x[1:] - xdq_nc[1:]).abs().mean()
    assert bulk_err < bulk_err_nc


def test_pack_unpack():
    torch.manual_seed(1)
    for bits in (1, 2, 4, 8):
        n = 40 if bits != 8 else 37
        q = torch.randint(0, 2**bits, (3, n))
        p = pack_int(q, bits)
        q2 = unpack_int(p, bits, n)
        assert torch.equal(q, q2), f"bits={bits}"


def test_hadamard_orthogonal():
    H = hadamard_matrix(128)
    assert torch.allclose(H @ H, torch.eye(128), atol=1e-5)
    assert torch.allclose(H, H.T, atol=1e-6)
    x = torch.randn(2, 4, 128)
    xh = hadamard(x, dim=-1)
    # preserves inner products
    a = (x[0, 0] * x[0, 1]).sum()
    b = (xh[0, 0] * xh[0, 1]).sum()
    assert abs(a - b) < 1e-3


def test_varn_reduces_imbalance_and_reconstructs():
    torch.manual_seed(2)
    # tiles with wildly different row/col scales
    row_scale = torch.rand(16, 128, 1) * 10 + 0.1
    col_scale = torch.rand(16, 1, 128) * 10 + 0.1
    T = torch.randn(16, 128, 128) * row_scale * col_scale
    imb_before = imbalance(T)
    Tn, Sc, Sr = variance_normalize(T, iterations=8)
    imb_after = imbalance(Tn)
    assert (imb_after <= imb_before + 1e-4).all(), (imb_before, imb_after)
    # exact reconstruction: (Tn * Sc) * Sr == T
    rec = (Tn * Sc) * Sr
    assert torch.allclose(rec, T, atol=1e-3)


def test_mckp_budget_and_ordering():
    torch.manual_seed(3)
    w = torch.rand(100) ** 3  # heavy-tailed weights
    eps = {0: 1.0, 2: 0.35, 4: 0.12, 8: 0.03, 16: 0.0}
    budget = 100 * 4.0
    alloc = mckp_lagrangian(w, eps, budget)
    total = alloc.sum().item()
    assert abs(total - budget) / budget < 0.05, total
    assert alloc.max() <= 16 and alloc.min() >= 0
    # monotonicity: bigger weight -> (weakly) more bits
    order = torch.argsort(w)
    top, bottom = order[-10:], order[:10]
    assert alloc[top].min() >= alloc[bottom].max() - 2

    # tighter budget -> fewer bits overall, more zeros
    alloc_tight = mckp_lagrangian(w, eps, 100 * 1.0)
    assert alloc_tight.sum() < total
    assert (alloc_tight == 0).sum() > (alloc == 0).sum()


def test_bennett_waterfilling():
    w = torch.rand(50) * 10
    sigma = torch.rand(50) + 0.1
    b = bennett_reverse_waterfilling(w, sigma, budget=50 * 4)
    assert abs(b.sum().item() - 200) / 200 < 0.01
    assert (b >= 0).all()
    # top-weight units get more bits
    idx = torch.argsort(w * sigma)
    assert b[idx[-5:]].min() >= b[idx[:5]].max() - 1e-6


def test_effective_bits():
    assert math.isclose(effective_bits(2, 128, 32), 2.25)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
