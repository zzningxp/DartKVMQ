"""SKVQ reorder-index calibration (arXiv 2405.06219 Sec. channel reorder).

Collects per-channel (min, max) stats of k_proj/v_proj outputs over
wikitext-2 windows (seed 42, official recipe), then KMeans-clusters the
(min, max) pairs per layer per K/V (official `minmax` metric,
n_init=10) into kv_hidden/gsize clusters. Output files use the OFFICIAL
format consumed by both their engine and our SkvqMethod:

  {"reorder_indices": [(k_idx[layer], v_idx[layer]), ...],
   "cluster_st_inds": [(k_bounds[layer], v_bounds[layer]), ...]}

Stats are cached under data/skvq_calib/ (gitignored); reorder files go
to configs/skvq_calib/ (model-specific hyperparams; note `*.pt` is
gitignored — `git add -f` them if you want them in the repo).

Run (cuda1, HF mirror):
  /home/daodao/anaconda3/envs/dart312/bin/python scripts/calibrate_skvq.py \
      --model Qwen/Qwen3-4B
"""
import argparse
import os
import pathlib
import random
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

try:  # official implementation, zero drift when importable
    sys.path.insert(0, str(ROOT / "external" / "SKVQ"))
    from calibration import get_reorder_indices
except Exception as e:  # pragma: no cover - fallback below
    print(f"[calibrate_skvq] official import failed ({e}); "
          "using local same-semantics copy")
    get_reorder_indices = None


def local_get_reorder_indices(data, n_cluster, save_path, metric="minmax",
                              num_layers=32):
    """Same math as external/SKVQ/calibration.py::get_reorder_indices (minmax)."""
    from sklearn.cluster import KMeans
    cmin, cmax = data["min"], data["max"]
    reorder_indices, cluster_st_inds = [], []
    for layer in range(num_layers):
        per = []
        for t in ("k", "v"):
            feats = torch.stack((cmin[t][layer], cmax[t][layer])).T
            km = KMeans(n_cluster, n_init=10).fit(feats)
            label = torch.from_numpy(km.labels_)
            idx = label.argsort()
            bounds = torch.zeros(n_cluster + 1, dtype=torch.int64)
            bounds[1:] = label.bincount().cumsum(0).to(torch.int64)
            per.append((idx, bounds))
        reorder_indices.append((per[0][0], per[1][0]))
        cluster_st_inds.append((per[0][1], per[1][1]))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"reorder_indices": reorder_indices,
                "cluster_st_inds": cluster_st_inds}, save_path)


def get_wikitext_windows(tokenizer, nsample, seqlen, seed=42):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                      split="train")
    ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    random.seed(seed)
    out = []
    for _ in range(nsample):
        i = random.randint(0, ids.shape[1] - seqlen - 1)
        out.append(ids[:, i:i + seqlen])
    return out


@torch.no_grad()
def collect_stats_proj(model, tokenizer, nsample, seqlen):
    """Per-layer per-channel min/max of k_proj/v_proj outputs (official
    stat hook, min/max only — the `minmax` reorder metric needs no more).

    NOTE: these are PRE-RoPE (and pre-QK-norm) activations — they match the
    cached tensor only in the official pre_rope=True engine."""
    layers = model.model.layers
    res = {"min": {"k": [None] * len(layers), "v": [None] * len(layers)},
           "max": {"k": [None] * len(layers), "v": [None] * len(layers)},
           "absmax": {"k": [None] * len(layers), "v": [None] * len(layers)}}

    def hook(ttype, layer_idx):
        def fn(_m, _x, y):
            y = y.reshape(-1, y.shape[-1]).float().cpu()
            mn, mx = res["min"][ttype][layer_idx], res["max"][ttype][layer_idx]
            am = y.abs().amax(0)
            res["min"][ttype][layer_idx] = (y.amin(0) if mn is None
                                            else torch.minimum(y.amin(0), mn))
            res["max"][ttype][layer_idx] = (y.amax(0) if mx is None
                                            else torch.maximum(y.amax(0), mx))
            ab = res["absmax"][ttype][layer_idx]
            res["absmax"][ttype][layer_idx] = (am if ab is None
                                               else torch.maximum(am, ab))
        return fn

    hooks = []
    for i, layer in enumerate(layers):
        hooks.append(layer.self_attn.k_proj.register_forward_hook(
            hook("k", i)))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(
            hook("v", i)))

    windows = get_wikitext_windows(tokenizer, nsample, seqlen)
    for j, ids in enumerate(windows):
        model(ids.to(model.device))
        if (j + 1) % 16 == 0:
            print(f"  {j + 1}/{len(windows)} windows", flush=True)
    for h in hooks:
        h.remove()
    return res


@torch.no_grad()
def collect_stats_cache(model, tokenizer, nsample, seqlen):
    """Per-layer per-channel min/max of the POST-norm/POST-RoPE tensors that
    actually land in the cache (what a post-RoPE harness like ours stores).
    Uses a plain DynamicCache per window and folds [H_kv, T, D] -> [T, H*D]
    (the flat axis SkvqMethod quantizes)."""
    from transformers.cache_utils import DynamicCache
    n_layers = model.config.num_hidden_layers
    res = {"min": {"k": [None] * n_layers, "v": [None] * n_layers},
           "max": {"k": [None] * n_layers, "v": [None] * n_layers},
           "absmax": {"k": [None] * n_layers, "v": [None] * n_layers}}

    def acc(ttype, i, t):     # t: [T, H*D] float32 cpu
        mn, mx = res["min"][ttype][i], res["max"][ttype][i]
        am = t.abs().amax(0)
        res["min"][ttype][i] = (t.amin(0) if mn is None
                                else torch.minimum(t.amin(0), mn))
        res["max"][ttype][i] = (t.amax(0) if mx is None
                                else torch.maximum(t.amax(0), mx))
        ab = res["absmax"][ttype][i]
        res["absmax"][ttype][i] = (am if ab is None
                                   else torch.maximum(am, ab))

    windows = get_wikitext_windows(tokenizer, nsample, seqlen)
    for j, ids in enumerate(windows):
        cache = DynamicCache(config=model.config)
        model(ids.to(model.device), past_key_values=cache)
        for i in range(n_layers):
            kc = cache.layers[i].keys    # [B, H_kv, T, D]
            vc = cache.layers[i].values
            k = (kc[0].permute(1, 0, 2).reshape(-1, kc.shape[1] * kc.shape[3])
                 .float().cpu())          # [T, H*D] flat channel order
            v = (vc[0].permute(1, 0, 2).reshape(k.shape[0], -1)
                 .float().cpu())
            acc("k", i, k)
            acc("v", i, v)
        if (j + 1) % 16 == 0:
            print(f"  {j + 1}/{len(windows)} windows", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--nsample", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--group-sizes", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--hook", choices=["proj", "cache"], default="proj",
                    help="proj = official k/v_proj outputs (PRE-RoPE; matches "
                         "their pre_rope=True engine). cache = the actual "
                         "cached post-norm/post-RoPE tensors (matches a "
                         "post-RoPE harness like ours)")
    args = ap.parse_args()

    short = args.model.split("/")[-1].lower()
    stats_dir = ROOT / "data" / "skvq_calib"
    out_dir = ROOT / "configs" / "skvq_calib"
    stats_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:  # models without explicit head_dim (e.g. Llama-2)
        head_dim = cfg.hidden_size // cfg.num_attention_heads
    kv_hidden = cfg.num_key_value_heads * head_dim
    n_layers = cfg.num_hidden_layers

    tag = "" if args.hook == "proj" else "_postrod"
    stats_path = stats_dir / f"{short}_wikitext2_n{args.nsample}" \
                             f"_len{args.seqlen}{tag}_stats.pt"
    if stats_path.exists():
        print(f"stats cached: {stats_path}")
        res = torch.load(stats_path)
        if "absmax" not in res:  # older stats file: absmax == max(|min|,|max|)
            res["absmax"] = {
                t: [torch.maximum(res["min"][t][i].abs(),
                                  res["max"][t][i].abs())
                    for i in range(n_layers)] for t in ("k", "v")}
    else:
        print(f"collecting k/v min-max stats ({args.hook} hook) on "
              f"{args.nsample} x {args.seqlen} wikitext2 tokens ...")
        fn = (collect_stats_proj if args.hook == "proj"
              else collect_stats_cache)
        res = fn(model, tok, args.nsample, args.seqlen)
        torch.save(res, stats_path)

    fn = get_reorder_indices or local_get_reorder_indices
    for g in args.group_sizes:
        assert kv_hidden % g == 0, (kv_hidden, g)
        n_cluster = kv_hidden // g
        out_path = out_dir / f"{short}_wikitext2_rod_g{g}{tag}.pt"
        print(f"clustering g={g} (n_cluster={n_cluster}) -> {out_path}")
        fn(res, n_cluster, str(out_path), metric="minmax",
           num_layers=n_layers)
    print("done")


if __name__ == "__main__":
    main()
