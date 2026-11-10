"""SKVQ alignment tests: our port vs the OFFICIAL SKVQ code
(external/SKVQ, untracked local snapshot), imported directly. CPU-only,
no CUDA kernel needed (official fake-quant pytorch path). If the snapshot
is absent (e.g. fresh clone) the alignment tests are skipped.

Covers: codec equivalence (no-reorder / reorder+variable clusters /
fp8 / 1.5-bit), full sliding-window layout equivalence against the
official SlidingKVCacheManager across prefill + decode, window
exactness, 16-bit identity, bits accounting.
"""
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dartkv.methods.skvq import SkvqMethod  # noqa: E402
from dartkv.methods.base import RunContext  # noqa: E402
from dartkv.quant.skvq import (group_bounds,  # noqa: E402
                               quant_dequant_channels)

SKVQ_DIR = ROOT / "external" / "SKVQ"
if SKVQ_DIR.exists():
    sys.path.insert(0, str(SKVQ_DIR))
    from KV_process import SKVQuantProcessor  # noqa: E402
    from KVcache_manager import SlidingKVCacheManager  # noqa: E402
else:  # pragma: no cover - external/ is an untracked local snapshot
    SKVQuantProcessor = SlidingKVCacheManager = None

H, D, HIDDEN = 3, 16, 48   # kv heads / head dim / flat channel axis


def _ctx(layers=1, heads=H, dim=D):
    return RunContext(num_layers=layers, num_kv_heads=heads, head_dim=dim,
                      dtype="torch.float32", device="cpu")


def _kv(b, t, seed=0, spread=8.0):
    gen = torch.Generator().manual_seed(seed)
    tok = torch.rand(b, H, t, 1, generator=gen) * spread + 0.2
    ch = torch.rand(b, H, 1, D, generator=gen) * spread + 0.2
    k = torch.randn(b, H, t, D, generator=gen) * tok * ch
    v = torch.randn(b, H, t, D, generator=gen) * tok * ch
    return k, v


def _official_proc(bits_k=2, bits_v=2, gsize=8, clip=0.92, reorder_meta=None,
                   fp8=False, dtype=torch.float32):
    proc = SKVQuantProcessor(
        K_target_bitwidth=bits_k, V_target_bitwidth=bits_v, gsize=gsize,
        hidden=HIDDEN, clipping=[clip] * 4, reorder_meta=reorder_meta,
        KIVI_mode=False, fp8=fp8, fake_quant=True)
    proc.layer_idx = 0
    return proc


def _rand_reorder_meta(hidden=HIDDEN, seed=0, min_cluster=3):
    """Random permutation + random-size cluster bounds (official format)."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    for s in (seed, seed + 100):
        idx = torch.randperm(hidden, generator=gen)
        sizes = []
        left = hidden
        while left > 0:
            sz = min(int(torch.randint(min_cluster, 9, (1,), generator=gen)),
                     left)
            sizes.append(sz)
            left -= sz
        bounds = torch.tensor([0] + list(torch.tensor(sizes).cumsum(0)))
        out.append((idx, bounds))
    (kidx, kb), (vidx, vb) = out
    return ({"k": kidx, "v": vidx}, {"k": kb, "v": vb})


def _ours_flat(t, bits, gsize, clip, reorder_idx=None, bounds=None, fp8=False):
    """Our codec on one [B,H,T,D] tensor -> same shape (official layout)."""
    b, h, seq, d = t.shape
    flat = t.transpose(1, 2).reshape(b * seq, h * d)
    if bounds is None:
        bounds = group_bounds(h * d, gsize)
    out = quant_dequant_channels(flat, bits, bounds, clip, reorder_idx, fp8)
    return out.reshape(b, seq, h, d).transpose(1, 2)


# -- codec equivalence -----------------------------------------------------

def test_codec_matches_official_no_reorder():
    k, v = _kv(2, 7, seed=1)
    for t, bits, name in ((k, 2, "k"), (v, 2, "v")):
        proc = _official_proc(bits_k=2, bits_v=2, gsize=8, clip=0.92)
        ref, _, _ = proc.quantization(name, t.clone(), impl="py")
        ours = _ours_flat(t, bits, gsize=8, clip=0.92)
        assert ours.shape == ref.shape == t.shape
        assert torch.allclose(ours, ref, atol=1e-6, rtol=0), \
            (name, (ours - ref).abs().max().item())


def test_codec_matches_official_bf16():
    k, v = _kv(1, 11, seed=2)
    t = k.to(torch.bfloat16)
    proc = _official_proc(bits_k=2, bits_v=2, gsize=8, clip=0.92)
    ref, _, _ = proc.quantization("k", t.clone(), impl="py")
    ours = _ours_flat(t, 2, gsize=8, clip=0.92)
    assert ours.dtype == torch.bfloat16
    assert torch.equal(ours, ref)   # same dtype flow -> bitwise


def test_codec_matches_official_reorder_fp8_1p5bit():
    k, v = _kv(2, 5, seed=3)
    meta = _rand_reorder_meta()
    proc = _official_proc(bits_k=2, bits_v=1.5, gsize=8, clip=0.95,
                          reorder_meta=meta, fp8=True)
    for t, bits, name, m in ((k, 2, "k", meta[0]["k"]),
                             (v, 1.5, "v", meta[0]["v"])):
        ref, _, _ = proc.quantization(name, t.clone(), impl="py")
        bounds = meta[1][name if name == "k" else "v"]
        ours = _ours_flat(t, bits, gsize=8, clip=0.95,
                          reorder_idx=meta[0][name], bounds=bounds, fp8=True)
        assert torch.allclose(ours, ref, atol=1e-6, rtol=0), \
            (name, (ours - ref).abs().max().item())


# -- layout equivalence vs official manager --------------------------------

def _run_official_manager(k_all, v_all, window, sink, proc_cfg):
    """Prefill in one call + one-token decodes; returns list of
    (K_cache, V_cache) after prefill and each decode step."""
    mgr = SlidingKVCacheManager(pre_rope=False, window_size=window,
                                attention_sink=sink, processor_config=proc_cfg,
                                full_prefill=True, fake_quant=True)
    t = k_all.shape[2]
    outs = []
    kp = vp = None
    for step in range(t):
        kf, vf = k_all[:, :, step:step + 1], v_all[:, :, step:step + 1]
        kp, vp = mgr(kp, vp, kf, vf, layer_idx=0)
        # kp/vp = (sink, quant, window, scale, zp) per K/V
        kc = torch.cat([x for x in kp[:3] if x is not None], dim=2)
        vc = torch.cat([x for x in vp[:3] if x is not None], dim=2)
        outs.append((kc, vc))
    return outs


def _official_proc_cfg(bits_k, bits_v, gsize, clip, reorder_meta, fp8=False):
    return dict(K_target_bitwidth=bits_k, V_target_bitwidth=bits_v,
                gsize=gsize, hidden=HIDDEN, clipping=[clip] * 4,
                reorder_meta=reorder_meta, KIVI_mode=False, fp8=fp8,
                fake_quant=True)


def test_layout_matches_official_manager():
    """Feed the same sequence to the official SlidingKVCacheManager (one
    prefill call + single-token decodes) and our method; caches must agree
    at every step."""
    T, W, S, G = 40, 16, 4, 8
    k, v = _kv(1, T, seed=4)
    for label, meta in (("plain", None),
                        ("reorder", _rand_reorder_meta(seed=7))):
        rod_kw = ({} if meta is None else
                  {"reorder_file": _write_rod_file(meta)})
        ours = SkvqMethod(bits_k=2, bits_v=2, group_size=G, window=W,
                          sink_tokens=S, clip=0.92, **rod_kw)
        ours.init_run(_ctx())
        proc_cfg = _official_proc_cfg(2, 2, G, 0.92, meta)
        our_outs = []
        for step in range(T):
            ours.ingest(0, k[:, :, step:step + 1], v[:, :, step:step + 1],
                        step == 0)
            our_outs.append(ours.get_cache(0))
        off = _run_official_manager(k, v, W, S, proc_cfg)
        for step in (T // 2, T - 1):
            Kc, Vc = our_outs[step]
            ok, ov = off[step]
            assert Kc.shape == ok.shape, (label, step, Kc.shape, ok.shape)
            assert torch.allclose(Kc, ok, atol=1e-6, rtol=0), \
                (label, step, (Kc - ok).abs().max().item())
            assert torch.allclose(Vc, ov, atol=1e-6, rtol=0)


def _write_rod_file(meta):
    import tempfile
    path = pathlib.Path(tempfile.mkdtemp()) / "rod.pt"
    torch.save({"reorder_indices": [(meta[0]["k"], meta[0]["v"])],
                "cluster_st_inds": [(meta[1]["k"], meta[1]["v"])]}, path)
    return str(path)


# -- method-level unit tests ------------------------------------------------

def test_window_exactness_and_regions():
    T, W, S, G = 40, 16, 4, 8
    k, v = _kv(1, T, seed=5)
    m = SkvqMethod(bits_k=2, bits_v=2, group_size=G, window=W, sink_tokens=S)
    m.init_run(_ctx())
    m.ingest(0, k, v, True)
    st = m.state[0]
    assert st["sink_k"].shape[2] == S
    assert st["win_k"].shape[2] == W
    assert st["n_quant"] == T - S - W
    dec = [torch.randn(1, H, 1, D) for _ in range(5)]
    for dk in dec:  # decode steps: window stays exactly W
        m.ingest(0, dk, torch.randn(1, H, 1, D), False)
    assert st["win_k"].shape[2] == W
    assert st["n_quant"] == T - S - W + 5
    assert m.total_tokens(0) == T + 5
    # sink exact, window exact (both FP16 regions): last 5 tokens are the
    # decode inputs; the first 11 window slots are late-prefill tokens
    Kc, _ = m.get_cache(0)
    assert torch.equal(Kc[:, :, :S], k[:, :, :S])
    assert torch.equal(Kc[:, :, -5:], torch.cat(dec, dim=2))


def test_chunked_prefill_equivalence():
    T, W, S, G = 40, 16, 4, 8
    k, v = _kv(1, T, seed=6)
    a = SkvqMethod(bits_k=2, bits_v=2, group_size=G, window=W, sink_tokens=S)
    a.init_run(_ctx())
    a.ingest(0, k, v, True)
    b = SkvqMethod(bits_k=2, bits_v=2, group_size=G, window=W, sink_tokens=S)
    b.init_run(_ctx())
    for lo, hi in ((0, 15), (15, 25), (25, 40)):  # chunked prefill
        b.ingest(0, k[:, :, lo:hi], v[:, :, lo:hi], lo == 0)
    Ka, _ = a.get_cache(0)
    Kb, _ = b.get_cache(0)
    assert torch.equal(Ka, Kb)   # tokens quantize independently -> exact


def test_16bit_identity():
    T, W, S, G = 30, 8, 4, 8
    k, v = _kv(1, T, seed=8)
    m = SkvqMethod(bits_k=16, bits_v=16, group_size=G, window=W,
                   sink_tokens=S)
    m.init_run(_ctx())
    m.ingest(0, k, v, True)
    Kc, Vc = m.get_cache(0)
    assert torch.equal(Kc, k) and torch.equal(Vc, v)


def test_bits_report():
    m = SkvqMethod(bits_k=2, bits_v=2, group_size=128, window=128,
                   sink_tokens=5)
    r = m.bits_report()
    assert abs(r["bpe_k"] - 2.25) < 1e-9
    assert abs(r["bpe_v"] - 2.25) < 1e-9
    assert r["fp16_eq_tokens_per_head"] == 133.0
    m8 = SkvqMethod(bits_k=2, bits_v=1.5, group_size=128, fp8_scales=True)
    r8 = m8.bits_report()
    assert abs(r8["bpe_k"] - 2.125) < 1e-9
    assert abs(r8["bpe_v"] - 1.625) < 1e-9


def test_reorder_helps_on_clustered_channels():
    """SKVQ's own ablation direction (paper: reorder + clip improves 2-bit).
    The mechanism reorder fixes: a few CONSISTENT channel outliers sharing a
    group with normal channels blow up the 2-bit range and crush the bulk
    (real KV structure, cf. Qwen3 K channel outliers). An oracle reorder
    separating them must beat contiguous channel groups by a wide margin."""
    gen = torch.Generator().manual_seed(9)
    b, h, t, d = 1, H, 64, D
    n = b * t
    flat = torch.randn(n, h * d, generator=gen)
    flat[:, :6] *= 40.0                        # 6 consistent outlier channels
    flat *= torch.rand(n, 1, generator=gen) * 3 + 0.5   # token scale spread
    scatter = torch.randperm(h * d, generator=gen)
    flat = flat[:, scatter]                     # outliers spread across heads
    reorder = scatter.argsort()                 # oracle: cluster them back
    bounds = torch.tensor([0, 6, 27, 48])       # 6 outlier + two bulk groups
    plain = quant_dequant_channels(flat, 2, group_bounds(h * d, 16), 0.92)
    rod = quant_dequant_channels(flat, 2, bounds, 0.92, reorder_idx=reorder)
    rel_plain = (flat - plain).norm() / flat.norm()
    rel_rod = (flat - rod).norm() / flat.norm()
    assert rel_rod < 0.6 * rel_plain, (rel_rod.item(), rel_plain.item())


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                fails += 1
                import traceback; traceback.print_exc()
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
