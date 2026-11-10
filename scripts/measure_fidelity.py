"""Measure K/V reconstruction fidelity of each codec on REAL Qwen3-4B
activations from a Line-Retrieval prefill prompt (L=100, sample 0).

For each layer: take the fp16-precision K/V the model wrote during prefill,
run method.encode_tiles -> decode_tiles on the sealed body region, and report
per-token cosine similarity and relative L2 error vs the original.
Codec-quality probe, independent of downstream task brittleness.
"""
import os
import pathlib
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402

from dartkv.bench import build_prompt  # noqa: E402
from dartkv.methods.base import RunContext  # noqa: E402
from dartkv.methods.kivi import KiviMethod  # noqa: E402
from dartkv.methods.kvarn import KvarnMethod  # noqa: E402
from dartkv.tasks.line_retrieval import LineRetrievalTask  # noqa: E402

MODEL_ID = "Qwen/Qwen3-4B"
LINES = 100
G = 128


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa").eval()

    task = LineRetrievalTask(lines=LINES, seed=20260604)
    s = task.samples(1)[0]
    text = build_prompt(tok, s["prompt"])
    enc = tok(text, return_tensors="pt").to("cuda")
    print(f"prompt tokens: {enc.input_ids.shape[1]}")

    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        model(**enc, past_key_values=cache)

    methods = {
        "kivi@2": KiviMethod(bits_k=2, bits_v=2, group_size=G),
        "kvarn@2": KvarnMethod(bits_k=2, bits_v=2, group_size=G,
                               sink_tokens=128),
        "kvarn_h@2": KvarnMethod(bits_k=2, bits_v=2, group_size=G,
                                 sink_tokens=128, varn_iters=0),
    }
    n_layers = len(cache.layers)
    cfg = model.config.get_text_config()
    ctx = RunContext(num_layers=n_layers, num_kv_heads=cfg.num_key_value_heads,
                     head_dim=cfg.head_dim, num_q_heads=cfg.num_attention_heads,
                     dtype=torch.bfloat16, device="cuda")
    for name, m in methods.items():
        m.init_run(ctx)
        agg = {"cos_k": [], "rel_k": [], "cos_v": [], "rel_v": []}
        worst = {"cos_k": 1.0, "layer": -1}
        for li in range(n_layers):
            k = cache.layers[li].keys
            v = cache.layers[li].values
            T = k.shape[2]
            M = (T - m.S) // G
            kb = k[:, :, m.S:m.S + M * G, :].contiguous()
            vb = v[:, :, m.S:m.S + M * G, :].contiguous()
            st = m.encode_tiles(kb, vb, li)[0]
            kd, vd = m.decode_tiles([st], li)
            kf, vf = kb.float(), vb.float()
            kdf, vdf = kd.float(), vd.float()
            cos_k = F.cosine_similarity(kdf.flatten(2), kf.flatten(2), dim=2)
            cos_v = F.cosine_similarity(vdf.flatten(2), vf.flatten(2), dim=2)
            rel_k = (kdf - kf).norm() / kf.norm()
            rel_v = (vdf - vf).norm() / vf.norm()
            agg["cos_k"].append(cos_k.mean().item())
            agg["cos_v"].append(cos_v.mean().item())
            agg["rel_k"].append(rel_k.item())
            agg["rel_v"].append(rel_v.item())
            mc = cos_k.mean().item()
            if mc < worst["cos_k"]:
                worst = {"cos_k": mc, "layer": li}
        n = n_layers
        print(f"\n== {name}  (body: T-{m.S} tokens, {M} groups)")
        print(f"   K: mean cos {sum(agg['cos_k'])/n:.4f}  "
              f"rel-L2 {sum(agg['rel_k'])/n:.4f}  "
              f"worst layer {worst['layer']} cos {worst['cos_k']:.4f}")
        print(f"   V: mean cos {sum(agg['cos_v'])/n:.4f}  "
              f"rel-L2 {sum(agg['rel_v'])/n:.4f}")

    # raw outlier structure of K/V (why 2-bit hurts): per-channel and
    # per-token max/median ratios on the fp16 reference
    k = cache.layers[0].keys.float()
    v = cache.layers[0].values.float()
    chan_max = k[0].amax(dim=(1, 2))          # [D]
    tok_max = v[0].amax(dim=(1, 2))           # [kvH] -> over heads
    print("\n-- layer-0 K channel max spread: "
          f"median {chan_max.median():.1f} p95 {chan_max.quantile(0.95):.1f} "
          f"max {chan_max.max():.1f}")
    vt = v[0].amax(dim=2)                     # [H, T]
    print("-- layer-0 V token max spread: "
          f"median {vt.median():.1f} p95 {vt.quantile(0.95):.1f} "
          f"max {vt.max():.1f}")


if __name__ == "__main__":
    main()
