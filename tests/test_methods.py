"""Method-level tests: tile codecs (KIVI / KVarN), grouped layout sawtooth,
bits accounting. CPU-only, no model download."""
import math
import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dartkv.methods.base import RunContext
from dartkv.methods.full import FullMethod
from dartkv.methods.kivi import KiviMethod
from dartkv.methods.kvarn import KvarnMethod


def _ctx(layers=2, heads=4, dim=64):
    return RunContext(num_layers=layers, num_kv_heads=heads, head_dim=dim,
                      dtype="torch.float32", device="cpu")


def _imbalanced_tile(b, h, g, d, seed=0, spread=8.0):
    """KV-like structure: per-token magnitude spread AND channel outliers."""
    gen = torch.Generator().manual_seed(seed)
    tok = torch.rand(b, h, g, 1, generator=gen) * spread + 0.2
    ch = torch.rand(b, h, 1, d, generator=gen) * spread + 0.2
    k = torch.randn(b, h, g, d, generator=gen) * tok * ch
    v = torch.randn(b, h, g, d, generator=gen) * tok * ch
    return k, v


def test_full_identity():
    m = FullMethod()
    m.init_run(_ctx(layers=1))
    k1, v1 = torch.randn(1, 4, 10, 64), torch.randn(1, 4, 10, 64)
    k2, v2 = torch.randn(1, 4, 5, 64), torch.randn(1, 4, 5, 64)
    m.ingest(0, k1, v1, True)
    m.ingest(0, k2, v2, False)
    K, V = m.get_cache(0)
    assert K.shape == (1, 4, 15, 64)
    assert torch.equal(K, torch.cat([k1, k2], dim=2))


def test_kivi_roundtrip_bound():
    torch.manual_seed(0)
    k, v = _imbalanced_tile(1, 4, 128, 64, seed=1)
    m = KiviMethod(bits_k=2, bits_v=2, group_size=128)
    m.init_run(_ctx(dim=64))
    st = m.encode_tile(k, v, 0)
    assert st.k_codes.shape == (1, 4, 64, 1, 128)   # [B,H,D,M,G] channel-major
    kd, vd = m.decode_tile(st, 0)
    assert kd.shape == k.shape and vd.shape == v.shape
    # per-channel K step (over tokens), broadcast back to [B,H,G,D]
    step_k = st.k_params["scale"].squeeze(-1).expand(
        -1, -1, -1, 128).transpose(2, 3)
    assert ((k - kd).abs() <= 0.501 * step_k).all(), \
        ((k - kd).abs().max(), step_k.min())
    # per-token V step (over channels)
    step_v = st.v_params["scale"]
    assert ((v - vd).abs() <= 0.501 * step_v + 1e-6).all()
    # codes in range
    assert st.k_codes.max() <= 3 and st.k_codes.min() >= 0
    assert st.v_codes.max() <= 3 and st.v_codes.min() >= 0


def test_kvarn_16bit_nearlossless():
    """Catches orientation/broadcast bugs: with 16-bit codes the transform
    chain (Hadamard + VarN + folded scales) must reconstruct near-exactly.
    A wrong transpose is silent when G == D but destroys values here."""
    k, v = _imbalanced_tile(1, 4, 128, 64, seed=2)
    m = KvarnMethod(bits_k=16, bits_v=16, group_size=128, varn_iters=8)
    m.init_run(_ctx(dim=64))
    st = m.encode_tile(k, v, 0)
    kd, vd = m.decode_tile(st, 0)
    rel_k = (k - kd).norm() / k.norm()
    rel_v = (v - vd).norm() / v.norm()
    assert rel_k < 1e-3 and rel_v < 1e-3, (rel_k, rel_v)


def test_kvarn_2bit_beats_plain_rtn_on_imbalanced():
    """The whole point of Hadamard + VarN: on token-magnitude/channel-outlier
    data, 2-bit KVarN must beat 2-bit plain RTN (KIVI codec) by a margin."""
    k, v = _imbalanced_tile(1, 4, 128, 64, seed=3, spread=8.0)
    mk = KvarnMethod(bits_k=2, bits_v=2, group_size=128, varn_iters=8)
    mk.init_run(_ctx(dim=64))
    kd, vd = mk.decode_tile(mk.encode_tile(k, v, 0), 0)
    mi = KiviMethod(bits_k=2, bits_v=2, group_size=128)
    mi.init_run(_ctx(dim=64))
    kid, vid = mi.decode_tile(mi.encode_tile(k, v, 0), 0)
    for name, a, b, ref in (("K", kd, kid, k), ("V", vd, vid, v)):
        rel_kv = (ref - a).norm() / ref.norm()
        rel_ki = (ref - b).norm() / ref.norm()
        assert rel_kv < 0.8 * rel_ki, \
            (name, rel_kv.item(), rel_ki.item())


def test_grouped_sawtooth_layout():
    G, S = 64, 32
    m = KvarnMethod(bits_k=2, bits_v=2, group_size=G, sink_tokens=S,
                    varn_iters=8)
    m.init_run(_ctx(layers=1, heads=4, dim=64))
    torch.manual_seed(4)
    K = torch.randn(1, 4, 100, 64)
    V = torch.randn(1, 4, 100, 64)
    m.ingest(0, K, V, True)
    st = m.state[0]
    assert st["sink_k"].shape[2] == S
    assert len(st["groups"]) == 1          # 68 tail tokens -> 1 sealed group
    assert st["tail_k"].shape[2] == 4
    assert m.total_tokens(0) == 100

    # sink is exact (fp16 region, untouched)
    Kc, _ = m.get_cache(0)
    assert torch.equal(Kc[:, :, :S], K[:, :, :S])

    # 60 decode steps: tail 4+60=64 -> seals into 2nd group exactly
    dec = [torch.randn(1, 4, 1, 64) for _ in range(60)]
    for t in dec:
        m.ingest(0, t, torch.randn(1, 4, 1, 64), False)
    assert len(st["groups"]) == 2
    assert st["tail_k"].shape[2] == 0
    assert m.total_tokens(0) == 160


def test_grouped_sink_tops_up_across_calls():
    """Prefill shorter than sink window: sink must keep collecting across
    calls instead of permanently under-filling."""
    m = KvarnMethod(bits_k=2, bits_v=2, group_size=64, sink_tokens=32)
    m.init_run(_ctx(layers=1, heads=2, dim=64))
    torch.manual_seed(5)
    for _ in range(15):  # 3 tokens per call x 15 calls = 45 tokens
        m.ingest(0, torch.randn(1, 2, 3, 64), torch.randn(1, 2, 3, 64), True)
    assert m.state[0]["sink_k"].shape[2] == 32
    assert m.state[0]["tail_k"].shape[2] == 13
    assert len(m.state[0]["groups"]) == 0


def test_grouped_end_to_end_16bit():
    """Full ingest -> get_cache path with 16-bit codec reconstructs the whole
    sequence (sink exact + groups/tail through codec)."""
    G, S = 64, 32
    m = KvarnMethod(bits_k=16, bits_v=16, group_size=G, sink_tokens=S,
                    varn_iters=8)
    m.init_run(_ctx(layers=1, heads=2, dim=64))
    torch.manual_seed(6)
    K = torch.randn(1, 2, 100, 64)
    V = torch.randn(1, 2, 100, 64)
    m.ingest(0, K, V, True)
    dK = torch.randn(1, 2, 50, 64)
    dV = torch.randn(1, 2, 50, 64)
    m.ingest(0, dK, dV, False)
    Kc, Vc = m.get_cache(0)
    Kref = torch.cat([K, dK], dim=2)
    Vref = torch.cat([V, dV], dim=2)
    assert Kc.shape == Kref.shape
    assert torch.equal(Kc[:, :, :S], Kref[:, :, :S])   # sink exact
    assert torch.equal(Kc[:, :, 96:], Kref[:, :, 96:])  # open tail exact (fp16)
    assert (Kc - Kref).norm() / Kref.norm() < 1e-3
    assert (Vc - Vref).norm() / Vref.norm() < 1e-3


def test_batched_chunk_equals_per_group():
    """encode_tiles on a 2-group chunk must match two encode_tile calls."""
    k, v = _imbalanced_tile(1, 2, 128, 64, seed=7)
    for cls, kw in ((KiviMethod, {}), (KvarnMethod, {"varn_iters": 8})):
        m = cls(bits_k=2, bits_v=2, group_size=64, **kw)
        m.init_run(_ctx(layers=1, heads=2, dim=64))
        sts = m.encode_tiles(k, v, 0)              # one chunk, M=2
        assert len(sts) == 1
        kb, vb = m.decode_tiles(sts, 0)            # [B,H,128,D]
        s1 = m.encode_tile(k[:, :, :64], v[:, :, :64], 0)
        s2 = m.encode_tile(k[:, :, 64:], v[:, :, 64:], 0)
        k1, v1 = m.decode_tiles([s1], 0)
        k2, v2 = m.decode_tiles([s2], 0)
        # tolerance covers fp32 matmul reassociation between sliced and
        # whole-tile Hadamard (values reach ~30 -> ~1e-3 noise)
        assert torch.allclose(kb, torch.cat([k1, k2], dim=2),
                              atol=2e-3, rtol=1e-4), cls
        assert torch.allclose(vb, torch.cat([v1, v2], dim=2),
                              atol=2e-3, rtol=1e-4), cls


def test_bits_reports():
    ki = KiviMethod(group_size=128)
    assert math.isclose(ki.bits_report()["bpe_k"], 2.25)
    kv = KvarnMethod(group_size=128)
    r = kv.bits_report()
    assert math.isclose(r["bpe_k"], 2.25)   # 2 + (2*fp8 + fp16)/128
    assert math.isclose(r["bpe_v"], 2.25)


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
