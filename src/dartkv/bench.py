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
from .methods.pmkvq import PmKvqMethod
from .methods.pmkvq_seg import PmKvqSegMethod
from .methods.pmkvq_read import ReadAwareSegMethod
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
                skvq_fp8: bool = False, reorder_file: str | None = None,
                requant_mode: str = "cascade", init_bits: int = 16,
                min_bits: int = 2, shrink: str = "layer",
                shrink_frac: float = 0.25, k_axis: str = "token",
                seg_g: int = 128, read_half_life: float = 0.0,
                read_damage: bool = False, read_invert: bool = False,
                uniform_depth: bool = False,
                beta: float = 1.0, evict: bool = False):
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
    if spec == "pmkvq":
        # budget shares the RDKV n axis (fp16-equivalent tokens per
        # (layer,head)); requant_mode: cascade (official) | direct (E1);
        # init_bits: pin the middle at N bits from the start (E1 arm:
        # 'never enjoyed precision' control); min_bits: floor — paper
        # uses 2 for Qwen ({2,4}) and 4 for Llama ({4,8})
        return PmKvqMethod(budget_tokens=budget, requant_mode=requant_mode,
                           init_bits=init_bits, min_bits=min_bits)
    if spec in ("pmkvq_read", "pmkvq_input"):
        # input-aware schedule (user-named 2026-11-16; code id
        # pmkvq_read kept for result compatibility) + design-round-1
        # knobs: half_life (decayed use counts), damage (value x
        # quant-error ordering), invert (least-used first, CoT-direction
        # control), beta (early trigger), evict (0-bit rung unifying
        # RDKV eviction with the quant ladder)
        return ReadAwareSegMethod(budget_tokens=budget, init_bits=init_bits,
                                  min_bits=min_bits, shrink_frac=shrink_frac,
                                  k_axis=k_axis, seg_g=seg_g,
                                  n_window=window, half_life=read_half_life,
                                  damage=read_damage, invert=read_invert,
                                  uniform_depth=uniform_depth,
                                  beta=beta, evict=evict)
    if spec == "pmkvq_seg":
        # research variant: gradual (oldest/newest-first, shrink_frac)
        # degradation schedule + optional KIVI-style per-channel K axis
        return PmKvqSegMethod(budget_tokens=budget, init_bits=init_bits,
                              min_bits=min_bits, shrink=shrink,
                              shrink_frac=shrink_frac, k_axis=k_axis,
                              n_window=window, beta=beta, evict=evict)
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


_NVML_HANDLE = None
_THERMAL_WARNED = False


def _gpu_temp() -> float:
    """Physical GPU temp (C) via pynvml; maps CUDA_VISIBLE_DEVICES.
    Returns -1 if unavailable; callers must treat -1 as 'sensor lost'
    and say so once (a silent -1 disables thermal pacing invisibly —
    bug found 2026-11-13 when an in-function import made every call
    after the first raise UnboundLocalError -> constant -1)."""
    global _NVML_HANDLE, _THERMAL_WARNED
    try:
        import pynvml   # idempotent after first import; must run each call
        if _NVML_HANDLE is None:
            pynvml.nvmlInit()
            idx = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")
                      .split(",")[0])
            _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(idx)
        return float(pynvml.nvmlDeviceGetTemperature(
            _NVML_HANDLE, pynvml.NVML_TEMPERATURE_GPU))
    except Exception as e:
        if not _THERMAL_WARNED:
            _THERMAL_WARNED = True
            print(f"  [thermal] WARN: GPU temp sensor unavailable "
                  f"({type(e).__name__}) — pacing disabled", flush=True)
        return -1.0


def run_task(model, tokenizer, task, method_spec: str, n: int,
             max_new: int, gen_over: dict | None = None, seed: int = 0,
             method_kw: dict | None = None, log_every: int = 10,
             prompter=None, skip_layers: int = 0):
    device = model.device
    gen_kw = gen_kwargs_for(model.config._name_or_path, gen_over)
    method_kw = method_kw or {}
    prompter = prompter or build_prompt
    samples = task.samples(n)
    records = []
    n_oom = 0
    t0 = time.time()
    # thermal pacing (user 2026-11-13): 400W is the 5090 power floor and
    # heavy tasks heat the card 31->90C in ~90s, so pace by temperature —
    # pause at DARTKV_TMAX, resume at DARTKV_TRESUME. A blind duty cycle
    # (DARTKV_DUTY_WORK/SLEEP) is also honoured if set. All off by default.
    t_max = float(os.environ.get("DARTKV_TMAX", "0"))
    t_resume = float(os.environ.get("DARTKV_TRESUME", "0")) or t_max - 8
    duty_work = float(os.environ.get("DARTKV_DUTY_WORK", "0"))
    duty_rest = float(os.environ.get("DARTKV_DUTY_SLEEP", "0"))
    last_duty = time.time()
    for i, s in enumerate(samples):
        if t_max > 0:
            t = _gpu_temp()
            if t >= t_max:
                t_pause = time.time()
                while t > t_resume:
                    time.sleep(5)
                    t = _gpu_temp()
                print(f"  [thermal] {t_max:.0f}C cap: paused "
                      f"{time.time() - t_pause:.0f}s, resumed at {t:.0f}C",
                      flush=True)
        elif duty_work and time.time() - last_duty >= duty_work:
            time.sleep(duty_rest)   # blind duty cycle (fallback mode)
            last_duty = time.time()
        torch.manual_seed(seed + i)
        method = make_method(method_spec, **method_kw)
        cache = CompressedCache(method, model_config=model.config,
                                skip_layers=skip_layers)
        text = prompter(tokenizer, s["prompt"])
        enc = tokenizer(text, return_tensors="pt",
                        add_special_tokens=getattr(
                            prompter, "add_special_tokens", True)).to(device)
        n_in = enc.input_ids.shape[1]
        try:
            with torch.no_grad():
                out = model.generate(**enc, past_key_values=cache,
                                     max_new_tokens=max_new, **gen_kw)
        except torch.OutOfMemoryError:
            n_oom += 1
            print(f"  [{i + 1}/{n}] CUDA OOM — sample skipped "
                  f"({n_oom} so far)", flush=True)
            del cache
            torch.cuda.empty_cache()
            continue
        except RuntimeError as e:  # cublas/SDPA surface OOM as RuntimeError
            if "out of memory" not in str(e).lower():
                raise
            n_oom += 1
            print(f"  [{i + 1}/{n}] CUDA OOM (RuntimeError) — sample skipped "
                  f"({n_oom} so far)", flush=True)
            del cache
            torch.cuda.empty_cache()
            continue
        gen = tokenizer.decode(out[0][n_in:], skip_special_tokens=True)
        sc = task.score(s, gen)
        records.append({"i": i, "meta": s["meta"], "answer": s["answer"],
                        "pred": sc["pred"], "correct": sc["correct"],
                        "n_prompt_tokens": n_in,
                        "n_gen_tokens": int(out.shape[1] - n_in),
                        "method_stats": method.stats(),
                        # full text: enables offline re-judging with the
                        # paper's scorer without re-running (2026-11-16;
                        # the old 80-char cap made saved runs unjudgeable)
                        "gen_text": gen, **sc})
        if (i + 1) % log_every == 0:
            acc = sum(r["correct"] for r in records) / len(records)
            print(f"  [{i + 1}/{n}] acc={acc:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    acc = sum(r["correct"] for r in records) / max(1, len(records))
    summary = {"task": task.name, "method": method_spec,
               "method_kw": method_kw, "n": n, "max_new": max_new,
               "skip_layers": skip_layers,
               "gen_kwargs": {k: v for k, v in gen_kw.items()},
               "acc": acc, "n_oom": n_oom, "n_scored": len(records),
               "wall_s": time.time() - t0}
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
