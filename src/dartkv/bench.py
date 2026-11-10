"""Unified benchmark runner: model x method x task -> JSON in results/.

Design:
  - model loaded once (bf16, cuda1 via CUDA_VISIBLE_DEVICES from caller)
  - per sample: FRESH method instance + CompressedCache (methods own per-run
    state), chat-templated prompt, model.generate, task.score
  - sampling presets per model family (paper-recommended decoding), CLI
    overrides win
"""
from __future__ import annotations

import json
import os
import pathlib
import time

import torch

from .cache.compressed import CompressedCache
from .methods.full import FullMethod
from .methods.kivi import KiviMethod
from .methods.kvarn import KvarnMethod
from .methods.rdkv import RdkvMethod
from .methods.skvq import SkvqMethod
from .tasks.line_retrieval import LineRetrievalTask

# paper-recommended decoding per model family (KVarN App. D).
# key 'llama-3' (not 'llama-3.1') so local snapshot dirs like
# Meta-Llama-3___1-8B match too.
GEN_PRESETS: dict[str, dict] = {
    "qwen3": dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20),
    "llama-3": dict(do_sample=True, temperature=0.6, top_p=0.9),
    "phi-4": dict(do_sample=False),
}

_TEMPLATE_DIR = pathlib.Path(__file__).parent / "chat_templates"


def ensure_chat_template(tokenizer):
    """Fill in the official chat template for snapshots that ship without
    one (our local ModelScope copy of Meta-Llama-3.1-8B-Instruct has a
    tokenizer_config.json stripped of chat_template; upstream has it).
    Keeps prompts byte-identical to HF/vLLM stacks."""
    if getattr(tokenizer, "chat_template", None) is not None:
        return
    name = (getattr(tokenizer, "name_or_path", "") or "").lower()
    if "llama-3" in name:
        tokenizer.chat_template = (_TEMPLATE_DIR
                                   / "llama-3.1.jinja").read_text()


def make_method(spec: str, bits: int = 2, group: int = 128, sink: int = 128,
                varn_iters: int = 8, bits_k: int | None = None,
                bits_v: int | None = None, rtn_quantile: float = 0.0,
                budget: int = 1024, window: int = 128, skvq_clip: float = 0.92,
                skvq_fp8: bool = False, reorder_file: str | None = None):
    """spec: full | kivi | kvarn | kvarn_h | skvq | rdkv
    (kvarn_h = Hadamard-only = KVarN w/o VarN, i.e. varn_iters=0 → scales
    collapse to 1; paper Tab.4 'Hadamard' row).
    bits_k/bits_v: K/V-only ablation overrides (default to bits).
    sink applies to kivi too (paper App. D: all quantized methods run the
    unified three-region layout S=128/G=128/R=128).
    rtn_quantile: KVarN percentile-range RTN (official KVARN_RTN_QUANTILE).
    budget: RDKV n — FP16-equivalent tokens per layer (paper budget axis).
    skvq extras: window W (sliding FP16 window; `group` doubles as the
    channel gsize), skvq_clip (unified clipping 0.91-0.97, official
    default 0.92), skvq_fp8 (e4m3 scale/zp storage), reorder_file
    (calibration reorder indices, see scripts/calibrate_skvq.py)."""
    bk = bits if bits_k is None else bits_k
    bv = bits if bits_v is None else bits_v
    if isinstance(bk, float) and bk == int(bk):
        bk = int(bk)
    if isinstance(bv, float) and bv == int(bv):
        bv = int(bv)
    if spec == "full":
        return FullMethod()
    if spec == "kivi":
        return KiviMethod(bits_k=bk, bits_v=bv, group_size=group,
                          sink_tokens=sink)
    if spec == "kvarn":
        return KvarnMethod(bits_k=bk, bits_v=bv, group_size=group,
                           sink_tokens=sink, varn_iters=varn_iters,
                           rtn_quantile=rtn_quantile)
    if spec == "kvarn_h":
        return KvarnMethod(bits_k=bk, bits_v=bv, group_size=group,
                           sink_tokens=sink, varn_iters=0,
                           rtn_quantile=rtn_quantile)
    if spec == "skvq":
        return SkvqMethod(bits_k=bk, bits_v=bv, group_size=group,
                          window=window, sink_tokens=sink, clip=skvq_clip,
                          fp8_scales=skvq_fp8, reorder_file=reorder_file)
    if spec == "rdkv":
        return RdkvMethod(budget_tokens=budget)
    raise ValueError(spec)


def gen_kwargs_for(model_id: str, overrides: dict | None = None):
    for key, kw in GEN_PRESETS.items():
        if key in model_id.lower():
            out = dict(kw)
            break
    else:
        out = dict(do_sample=False)
    if overrides:
        if overrides.get("greedy"):
            out = {"do_sample": False}
        else:
            out.update({k: v for k, v in overrides.items() if v is not None})
            if any(v is not None for v in overrides.values()):
                out["do_sample"] = True
    return out


def build_prompt(tokenizer, user_text: str,
                 enable_thinking: bool = False) -> str:
    """Chat-template the prompt. enable_thinking=False for Qwen3
    non-thinking evals (template inserts an empty <think> block); True
    for thinking-mode tasks (MATH500 on Qwen3). Other templates ignore
    the extra jinja variable."""
    ensure_chat_template(tokenizer)
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking)
    except TypeError:  # template signature without the kwarg
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False, add_generation_prompt=True)


# chat-templated text already carries bos ({{- bos_token }} in the
# template); tokenizing with add_special_tokens=True would double it on
# Llama (Qwen adds no bos, hence unnoticed until Llama runs)
build_prompt.add_special_tokens = False


def run_task(model, tokenizer, task, method_spec: str, n: int,
             max_new: int, gen_over: dict | None = None, seed: int = 0,
             method_kw: dict | None = None, log_every: int = 10,
             prompter=None):
    device = model.device
    gen_kw = gen_kwargs_for(model.config._name_or_path, gen_over)
    method_kw = method_kw or {}
    prompter = prompter or build_prompt
    samples = task.samples(n)
    records = []
    t0 = time.time()
    for i, s in enumerate(samples):
        torch.manual_seed(seed + i)
        method = make_method(method_spec, **method_kw)
        cache = CompressedCache(method, model_config=model.config)
        text = prompter(tokenizer, s["prompt"])
        enc = tokenizer(text, return_tensors="pt",
                        add_special_tokens=getattr(
                            prompter, "add_special_tokens", True)).to(device)
        n_in = enc.input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(**enc, past_key_values=cache,
                                 max_new_tokens=max_new, **gen_kw)
        gen = tokenizer.decode(out[0][n_in:], skip_special_tokens=True)
        sc = task.score(s, gen)
        records.append({"i": i, "meta": s["meta"], "answer": s["answer"],
                        "pred": sc["pred"], "correct": sc["correct"],
                        "n_prompt_tokens": n_in,
                        "gen_text": gen[:80], **sc})
        if (i + 1) % log_every == 0:
            acc = sum(r["correct"] for r in records) / len(records)
            print(f"  [{i + 1}/{n}] acc={acc:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    acc = sum(r["correct"] for r in records) / max(1, len(records))
    summary = {"task": task.name, "method": method_spec,
               "method_kw": method_kw, "n": n, "max_new": max_new,
               "gen_kwargs": {k: v for k, v in gen_kw.items()},
               "acc": acc, "wall_s": time.time() - t0}
    if records and "loose" in records[0]:  # IFEval: loose prompt-level acc
        summary["acc_loose"] = (sum(r["loose"] for r in records)
                                / max(1, len(records)))
    # last sample's method instance holds the actual run stats (e.g. RDKV
    # retention/bpe are only known after compression)
    bits = method.bits_report()
    return {"summary": summary, "bits_report": bits, "records": records}


def save_report(report, model_id: str, out_dir: str, tag: str | None = None):
    os.makedirs(out_dir, exist_ok=True)
    if tag:
        name = tag
    else:
        m = model_id.split("/")[-1]
        name = f"{m}__{report['summary']['method']}__{report['summary']['task']}"
        mk = report["summary"].get("method_kw") or {}
        if mk.get("bits"):
            name += f"__k{mk['bits']}v{mk['bits']}"
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"saved {path}")
    return path
