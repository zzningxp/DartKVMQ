"""Smoke test: run a tiny HF model through CompressedCache end-to-end.

Manual prefill + decode loop (does not depend on generate()'s cache plumbing),
then generate() as a second check. Env: HF mirror + cuda1 (cuda0 is busy).

Usage:
  /home/daodao/anaconda3/envs/dart312/bin/python scripts/smoke_cache.py
"""
import os
import sys
import pathlib

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from dartkv.cache.compressed import CompressedCache  # noqa: E402
from dartkv.methods.full import FullMethod  # noqa: E402
from dartkv.methods.kivi import KiviMethod  # noqa: E402
from dartkv.methods.kvarn import KvarnMethod  # noqa: E402
from dartkv.methods.skvq import SkvqMethod  # noqa: E402

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
N_NEW = 16


def manual_run(model, method, prompt_ids, n_new):
    cache = CompressedCache(method, model_config=model.config)
    with torch.no_grad():
        out = model(input_ids=prompt_ids, past_key_values=cache,
                    use_cache=True)
        cur = out.logits[:, -1].argmax(-1, keepdim=True)
        gen = [cur]
        for _ in range(n_new - 1):
            out = model(input_ids=cur, past_key_values=cache, use_cache=True)
            assert torch.isfinite(out.logits).all()
            cur = out.logits[:, -1].argmax(-1, keepdim=True)
            gen.append(cur)
    return torch.cat(gen, dim=1), cache


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, device_map="cuda")
    model.eval()
    cfg = model.config
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    print(f"model: L={cfg.num_hidden_layers} kv_heads={kv_heads} head_dim={hd}")
    assert hd & (hd - 1) == 0, f"head_dim {hd} not power of 2 (Hadamard)"

    prompt = "Once upon a time there was a small robot"
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    n_prompt = ids.shape[1]
    print(f"prompt tokens: {n_prompt}")

    # small layout so the codec path actually runs on a short sequence
    methods = [
        ("full", FullMethod()),
        ("kivi", KiviMethod(bits_k=2, bits_v=2, group_size=8)),
        ("kvarn", KvarnMethod(bits_k=2, bits_v=2, group_size=8,
                              sink_tokens=4, varn_iters=8)),
        ("skvq", SkvqMethod(bits_k=2, bits_v=2, group_size=8, window=4,
                            sink_tokens=2, clip=0.92)),
    ]
    outs = {}
    for name, m in methods:
        torch.manual_seed(0)
        gen, cache = manual_run(model, m, ids, N_NEW)
        n_cached = cache.get_seq_length(0)
        # the final sampled token is never fed back, so cache sees total-1
        total = n_prompt + N_NEW - 1
        assert n_cached == total, (name, n_cached, total)
        # storage shape checks on layer 0
        st = getattr(m, "state", None)
        if st is not None and name == "skvq":
            s0 = st[0]
            print(f"  [{name}] quant={s0['n_quant']} "
                  f"sink={s0['sink_k'].shape[2] if s0['sink_k'] is not None else 0} "
                  f"win={s0['win_k'].shape[2] if s0['win_k'] is not None else 0}")
        elif st is not None:
            print(f"  [{name}] groups={len(st[0]['groups'])} "
                  f"sink={st[0]['sink_k'].shape[2] if st[0]['sink_k'] is not None else 0} "
                  f"tail={st[0]['tail_k'].shape[2]}")
        outs[name] = gen
        print(f"[{name}] gen ok: {tok.decode(gen[0])!r}")

    # quantized methods must be finite (asserted) and should perturb the
    # output relative to full (2-bit on random weights -> likely differs)
    for q in ("kivi", "kvarn", "skvq"):
        same = torch.equal(outs[q], outs["full"])
        print(f"  {q} == full: {same} (not required to differ)")

    # generate() path with the cache (plumbing check)
    m = KvarnMethod(bits_k=2, bits_v=2, group_size=8, sink_tokens=4)
    cache = CompressedCache(m, model_config=cfg)
    with torch.no_grad():
        out = model.generate(ids, past_key_values=cache, max_new_tokens=N_NEW,
                             do_sample=False)
    assert out.shape[1] == n_prompt + N_NEW, out.shape
    assert cache.get_seq_length(0) >= n_prompt + N_NEW - 1
    print(f"[kvarn via generate()] ok: {tok.decode(out[0][n_prompt:])!r}")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
