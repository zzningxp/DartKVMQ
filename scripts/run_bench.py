"""Benchmark CLI.

Examples (always run with CUDA_VISIBLE_DEVICES=1, HF mirror):
  python scripts/run_bench.py --model Qwen/Qwen3-4B --method kvarn \
      --task line_retrieval --lines 100 200 300 400 500 600 --n 100
  python scripts/run_bench.py --model Qwen/Qwen3-4B --method full \
      --task line_retrieval --lines 100 200 --n 20 --tag pilot
"""
import argparse
import os
import pathlib
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
# eager attention at long contexts needs the fp32 softmax tile + weights to
# coexist; expandable segments avoids fragmentation OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from dartkv.bench import run_task, save_report, build_prompt  # noqa: E402
from dartkv.tasks.ifeval import IFEvalTask  # noqa: E402
from dartkv.tasks.line_retrieval import LineRetrievalTask  # noqa: E402


class RawPrompter:
    """Bypass the chat template (paper prompts may be raw completions)."""

    add_special_tokens = True  # raw text: let the tokenizer add bos

    def __call__(self, tokenizer, user_text):
        return user_text


class ChatPrompter:

    add_special_tokens = False  # template already emits bos_token

    def __init__(self, enable_thinking: bool = False):
        self.enable_thinking = enable_thinking

    def __call__(self, tokenizer, user_text):
        return build_prompt(tokenizer, user_text,
                            enable_thinking=self.enable_thinking)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", default="full",
                    choices=["full", "kivi", "kvarn", "kvarn_h", "skvq",
                             "rdkv"])
    ap.add_argument("--task", default="line_retrieval",
                    choices=["line_retrieval", "ifeval", "longbench",
                             "math500", "gsm8k"])
    ap.add_argument("--datasets", type=str, nargs="+", default=None,
                    help="longbench dataset names (default: EN16)")
    ap.add_argument("--max-len", type=int, default=32768,
                    help="longbench middle-truncation token budget")
    ap.add_argument("--lines", type=int, nargs="+",
                    default=[100, 200, 300, 400, 500, 600])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--bits", type=float, default=2,
                    help="quant bits (SKVQ allows 1.5)")
    ap.add_argument("--bits-k", type=float, default=None,
                    help="bits for K only (ablation; defaults to --bits)")
    ap.add_argument("--bits-v", type=float, default=None,
                    help="bits for V only (ablation; defaults to --bits)")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--sink", type=int, default=128)
    ap.add_argument("--varn-iters", type=int, default=8)
    ap.add_argument("--rtn-quantile", type=float, default=0.0,
                    help="KVarN percentile-range RTN q (official "
                         "KVARN_RTN_QUANTILE; 0 = min/max like the paper)")
    ap.add_argument("--budget", type=int, default=1024,
                    help="RDKV budget n: FP16-equivalent tokens per layer "
                         "(paper axis {64,128,256,512,1024,2048})")
    ap.add_argument("--window", type=int, default=128,
                    help="SKVQ sliding FP16 window W (paper 128; 0 = off)")
    ap.add_argument("--skvq-clip", type=float, default=0.92,
                    help="SKVQ unified clipping (paper: 0.91-0.97)")
    ap.add_argument("--skvq-fp8", action="store_true",
                    help="SKVQ: store scale/zero in fp8 e4m3 (official "
                         "longbench configs)")
    ap.add_argument("--reorder-file", type=str, default=None,
                    help="SKVQ calibration reorder .pt (official format; "
                         "see scripts/calibrate_skvq.py). None = contiguous "
                         "channel groups (SKVQ w/o reorder ablation)")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task-seed", type=int, default=20260604)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-chat", action="store_true",
                    help="raw completion prompt (skip chat template)")
    ap.add_argument("--greedy", action="store_true",
                    help="force do_sample=False (true greedy)")
    ap.add_argument("--thinking", action="store_true",
                    help="Qwen3 thinking mode (MATH500/1c protocol); "
                         "default is non-thinking")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="model load dtype (default bfloat16)")
    ap.add_argument("--attn", default="sdpa",
                    choices=["sdpa", "eager", "flex"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.greedy:
        gen_over = dict(temperature=None, top_p=None, top_k=None,
                        greedy=True)
    else:
        gen_over = dict(temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k)
    prompter = RawPrompter() if args.no_chat else \
        ChatPrompter(enable_thinking=args.thinking)

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map="cuda",
        attn_implementation=args.attn)
    model.eval()
    print(f"model {args.model} on {model.device}")

    method_kw = dict(bits=args.bits, group=args.group, sink=args.sink,
                     varn_iters=args.varn_iters,
                     rtn_quantile=args.rtn_quantile, budget=args.budget,
                     window=args.window, skvq_clip=args.skvq_clip,
                     skvq_fp8=args.skvq_fp8, reorder_file=args.reorder_file)
    if args.bits_k is not None or args.bits_v is not None:
        # K/V asymmetric ablation: make_method takes bits (mean) for
        # dispatch; pass through explicit kwargs for methods that accept them
        bk = args.bits_k if args.bits_k is not None else args.bits
        bv = args.bits_v if args.bits_v is not None else args.bits
        method_kw = dict(bits=bk, bits_v=bv, bits_k=bk, group=args.group,
                         sink=args.sink, varn_iters=args.varn_iters,
                         rtn_quantile=args.rtn_quantile, budget=args.budget,
                         window=args.window, skvq_clip=args.skvq_clip,
                         skvq_fp8=args.skvq_fp8,
                         reorder_file=args.reorder_file)
    out_dir = args.out or os.path.join("results", args.task)

    if args.task == "ifeval":
        runs = [("all", IFEvalTask())]
    elif args.task == "math500":
        from dartkv.tasks.math500 import MATH500Task
        runs = [("all", MATH500Task())]
        if args.max_new == 32:  # CLI default is line-retrieval's
            args.max_new = 8192  # paper: 8192 for Qwen3 thinking
    elif args.task == "gsm8k":
        from dartkv.tasks.gsm8k import GSM8KTask
        runs = [("all", GSM8KTask())]
        if args.max_new == 32:
            args.max_new = 256  # benchmark.md Core protocol
    elif args.task == "longbench":
        from dartkv.tasks.longbench import EN16, LongBenchTask
        lb = []
        for ds in (args.datasets or EN16):
            t = LongBenchTask(dataset=ds, max_len=args.max_len)
            t.set_tokenizer(tok)
            lb.append((ds, t))
        runs = lb
    else:
        runs = [(L, LineRetrievalTask(lines=L, seed=args.task_seed))
                for L in args.lines]

    for L, task in runs:
        print(f"== task={args.task} L={L} method={args.method} "
              f"n={args.n}", flush=True)
        max_new = (task.max_gen if args.task == "longbench"
                   else args.max_new)
        lb_prompter = (RawPrompter() if args.task == "longbench"
                       and not task.needs_chat else None)
        run_prompter = lb_prompter or prompter
        report = run_task(model, tok, task, args.method, n=args.n,
                          max_new=max_new, gen_over=gen_over,
                          seed=args.seed, method_kw=method_kw,
                          prompter=run_prompter)
        if args.task == "line_retrieval":
            report["summary"]["lines"] = L
        if args.task == "longbench":
            report["summary"]["dataset"] = L
        report["summary"]["model"] = args.model
        if args.tag:
            tag = f"{args.tag}__L{L}" if args.task == "line_retrieval" \
                else (f"{args.tag}__{L}" if args.task == "longbench"
                      else args.tag)
        else:
            tag = f"{args.model.split('/')[-1]}__{args.method}__{args.task}"
            if args.method == "rdkv":
                tag += f"__n{args.budget}"
            elif args.method == "skvq":
                fmt = lambda x: int(x) if x == int(x) else x  # noqa: E731
                tag += (f"__k{fmt(args.bits_k or args.bits)}"
                        f"v{fmt(args.bits_v or args.bits)}"
                        f"w{args.window}g{args.group}"
                        f"{'rod' if args.reorder_file else ''}"
                        f"{'fp8' if args.skvq_fp8 else ''}"
                        f"c{args.skvq_clip}")
            elif args.method != "full":
                fmt = lambda x: int(x) if x == int(x) else x  # noqa: E731
                tag += f"__k{fmt(args.bits)}v{fmt(args.bits)}"
        save_report(report, args.model, out_dir, tag=tag)
        accs = [f"acc={report['summary']['acc']:.3f}"]
        if "acc_loose" in report["summary"]:
            accs.append(f"loose={report['summary']['acc_loose']:.3f}")
        print(f"   {' '.join(accs)}", flush=True)


if __name__ == "__main__":
    main()
