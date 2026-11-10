"""Clean codec comparison in BALANCED space (no inverse-rotation algebra):
same rotated tile [D, G] -> our varn+RTN+fold vs official batched store,
compare the dequantized balanced tile elementwise."""
import importlib.util
import os
import pathlib
import sys
import types

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402

from dartkv.bench import build_prompt  # noqa: E402
from dartkv.quant.hadamard import hadamard_matrix  # noqa: E402
from dartkv.quant.uniform import quantize_asym, dequantize_asym  # noqa: E402
from dartkv.quant.varn import variance_normalize as our_varn  # noqa: E402
from dartkv.tasks.line_retrieval import LineRetrievalTask  # noqa: E402

OFFICIAL = ROOT / "external" / "KVarN" / "vllm" / "model_executor" / "layers" / "quantization" / "kvarn"


def load_official():
    for name in ["vllm", "vllm.model_executor", "vllm.model_executor.layers",
                 "vllm.model_executor.layers.quantization",
                 "vllm.model_executor.layers.quantization.kvarn",
                 "vllm.model_executor.layers.quantization.kvarn.sinkhorn"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    spec = importlib.util.spec_from_file_location(
        "official_kvarn_sinkhorn", OFFICIAL / "sinkhorn.py")
    sink = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sink)
    sys.modules["vllm.model_executor.layers.quantization.kvarn.sinkhorn"] = sink
    spec2 = importlib.util.spec_from_file_location(
        "official_kvarn_store", OFFICIAL.parent.parent.parent.parent
        / "v1" / "attention" / "ops" / "kvarn_store.py")
    store = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(store)
    return sink, store


def main():
    sink, store = load_official()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa").eval()
    task = LineRetrievalTask(lines=100, seed=20260604)
    s = task.samples(1)[0]
    enc = tok(build_prompt(tok, s["prompt"]), return_tensors="pt").to("cuda")
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        model(**enc, past_key_values=cache)

    D = G = 128
    Hm = hadamard_matrix(D, device="cuda")   # orthonormal, official convention

    for layer in [0, 17, 35]:
        for head in [0, 3]:
            k = cache.layers[layer].keys.float()
            tile0 = k[0, head, 128:128 + G, :].to("cuda")   # [G, D]
            kt = (tile0 @ Hm).T.contiguous().unsqueeze(0)   # [1, D, G]

            # ours (min/max RTN, folds channel scale)
            bal, sc, sr = our_varn(kt, iterations=8)
            q, s_rt, z_rt = quantize_asym(bal[0], 2, dim=-1, group_size=None)
            kd_ours = (dequantize_asym(
                q, s_rt * sr[0], z_rt * sr[0]) * sc[0])      # [D, G]

            # official batched store (KVARN_RTN_QUANTILE unset -> min/max)
            bal_o, sc_o, sr_o = sink.variance_normalize_batched(kt, iterations=8)
            out = store.kvarn_store_tile_k_batch_from_sinkhorn(
                bal_o, sc_o.reshape(1, -1), sr_o.reshape(1, -1), bits=2)
            pb = out["q_packed_uint8"][0].to(torch.int32)    # [D, G/4]
            codes = torch.stack([(pb >> (2 * i)) & 3 for i in range(4)], -1)
            codes = codes.reshape(D, G).float()
            kd_off = (codes * out["s_col_K"][0].unsqueeze(-1)
                      + out["zp_K"][0].unsqueeze(-1)) * out["s_row_K"][0].unsqueeze(0)

            same = torch.equal(kd_ours, kd_off)
            diff = (kd_ours - kd_off).norm() / kd_off.norm()
            cos_ours = F.cosine_similarity(kd_ours.flatten(), kt[0].flatten(), 0)
            cos_off = F.cosine_similarity(kd_off.flatten(), kt[0].flatten(), 0)
            print(f"L{layer:2d} h{head}: identical={same} rel-diff "
                  f"{diff:.3e} cos(ours) {cos_ours:.4f} cos(off) {cos_off:.4f}")


if __name__ == "__main__":
    main()
