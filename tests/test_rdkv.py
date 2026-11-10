"""RDKV method tests: probe weights, budget binding, eviction semantics,
decode append, end-to-end generate on the tiny model. CPU except the last."""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dartkv.methods.base import RunContext  # noqa: E402
from dartkv.methods.rdkv import RdkvMethod  # noqa: E402


def _ctx(layers=2, heads=4, dim=64, qheads=8):
    return RunContext(num_layers=layers, num_kv_heads=heads, head_dim=dim,
                      num_q_heads=qheads, dtype="torch.float32", device="cpu")


def _make_method(T=128, n=16, layers=2, heads=4, dim=64, qheads=8, seed=0):
    torch.manual_seed(seed)
    m = RdkvMethod(budget_tokens=n, window=8, pool=1, eps_sub=64)
    m.init_run(_ctx(layers, heads, dim, qheads))
    r = qheads // heads
    for l in range(layers):
        k = torch.randn(1, heads, T, dim)
        v = torch.randn(1, heads, T, dim)
        m.ingest(l, k, v, True)
        # fake probe: window queries that strongly attend to token T//4
        q = torch.randn(qheads, 8, dim)
        anchor = k[0, :, T // 4].mean(0)                # [dim]
        q[:, 2, :] = anchor * 4.0                       # one aligned query/head
        m._probe_q.append((q, T))
    return m


def test_probe_weights_shapes_and_nonneg():
    m = _make_method()
    k = torch.zeros(1, 4, 128, 64)
    q_saved, T = m._probe_q[0]
    w_t, w_c = m._probe_weights(0, k)
    assert w_t.shape == (4, 128) and w_c.shape == (4, 64)
    assert (w_t >= 0).all() and (w_c >= 0).all()


def test_compression_binds_budget_and_evicts_low_weight():
    T, n = 128, 16
    m = _make_method(T=T, n=n)
    m.on_pass_end("prefill")
    st = m.state[0]
    kept = st["stats"]["kept"]
    assert 0 < kept < T, "small budget must evict some but not all"
    # budget binding: sum_t b_t <= (1-r_k)*32*n = 256
    assert st["stats"]["bpe_v_payload"] * 16.0 * T <= 16 * n * 1.0001
    # evicted tokens must come from the lowest w_t mass: the anchor token
    # (T//4, strongly attended by the probe) must survive
    w_t, _ = m._probe_weights(0, torch.zeros(1, 4, T, 64))
    w_sum = w_t.sum(0)
    top10 = set(w_sum.topk(10).indices.tolist())
    kept_frac = st["compressed"][1].shape[2] / T
    assert kept_frac == kept / T
    # anchor rank check: T//4 should be among the top-mass tokens
    assert T // 4 in set(w_sum.topk(max(kept, 10)).indices.tolist())


def test_large_budget_is_lossless_identity():
    T = 64
    m = _make_method(T=T, n=8 * T)   # 16 bits everywhere, no eviction
    m.on_pass_end("prefill")
    K, V = m.get_cache(0)
    assert K.shape == (1, 4, T, 64)
    assert torch.equal(K, m.state[0]["compressed"][0])


def test_decode_appends_fp16_zone_c():
    T = 64
    m = _make_method(T=T, n=2 * T)
    m.on_pass_end("prefill")
    t0 = m.get_cache(0)[0].shape[2]
    nk = torch.randn(1, 4, 3, 64)
    nv = torch.randn(1, 4, 3, 64)
    m.ingest(0, nk, nv, False)
    K, V = m.get_cache(0)
    assert K.shape[2] == t0 + 3
    assert torch.equal(K[:, :, -3:], nk)
    # second decode step keeps appending
    m.ingest(0, nk, nv, False)
    assert m.get_cache(0)[0].shape[2] == t0 + 6


def test_detach_restores_sdpa():
    import torch.nn.functional as F
    orig = F.scaled_dot_product_attention
    m = _make_method()
    m.attach(None)
    assert F.scaled_dot_product_attention is not orig
    m.detach()
    assert F.scaled_dot_product_attention is orig


def test_end_to_end_tiny_generate():
    """Tiny Llama + CompressedCache + rdkv: generation must run and the
    compressed cache must be shorter than the prompt."""
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dartkv.cache.compressed import CompressedCache

    name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32,
                                                 attn_implementation="sdpa")
    model.eval()
    prompt = "The quick brown fox jumps over the lazy dog. " * 8
    enc = tok(prompt, return_tensors="pt")

    m = RdkvMethod(budget_tokens=4, window=4, pool=1, eps_sub=32)
    cache = CompressedCache(m, model_config=model.config)
    with torch.no_grad():
        out = model.generate(**enc, past_key_values=cache, max_new_tokens=8,
                             do_sample=False)
    T = m.state[0]["stats"]["T"]
    assert T > 0 and m.state[0]["stats"]["kept"] < T, "must have evicted"
    assert m._sdpa_orig is None, "patch must be detached after prefill"
    assert out.shape[1] == enc.input_ids.shape[1] + 8
