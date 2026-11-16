"""AIME-24 task (paper main-table alignment, 2026-11-15).

Dataset: HuggingFaceH4/aime_2024 (30 problems, integer answers 0-999).
Prompt: DeepSeek-R1 recommended reasoning prompt. Score: exact integer
match on the last boxed/number in the generation (paper judge.py口径的
贪心简化版).
"""
from __future__ import annotations

import re

_TEMPLATE = (
    "{problem}\n\nPlease reason step by step, and put your final answer "
    "within \\boxed{{}}."
)
_BOX_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_ANS_RE = re.compile(r"final answer|answer is")


class AimeTask:
    name = "aime"

    def __init__(self, year: str = "2024", seed: int = 20260604):
        from datasets import load_dataset
        self.ds = load_dataset("HuggingFaceH4/aime_2024", split="train")

    def samples(self, n: int):
        out = []
        for i, row in enumerate(self.ds):
            if len(out) >= min(n, len(self.ds)):
                break
            out.append({"prompt": _TEMPLATE.format(problem=row["problem"]),
                        "answer": str(int(row["answer"])),
                        "meta": {"id": i}})
        return out

    @classmethod
    def score(cls, sample, generated_text: str):
        # paper judge (eval_aime.math_postprocess) never returns empty:
        # boxed -> 'final answer|answer is' phrase -> first sentence.
        # Our old boxed-only scorer left 50% of AIME preds empty
        # (greedy+16k truncation), scoring them as wrong (2026-11-16).
        boxes = _BOX_RE.findall(generated_text)
        pred_raw = boxes[-1] if boxes else ""
        if not pred_raw:
            for seg in generated_text.split("."):
                if _ANS_RE.search(seg.lower()):
                    pred_raw = seg
                    break
            else:
                pred_raw = generated_text.split(".")[0]
        nums = _NUM_RE.findall(pred_raw)
        pred = nums[-1].rstrip(".0") if nums else pred_raw
        try:                                   # fallback text may be
            pred = str(int(float(pred)))       # non-numeric (crashed the
        except (ValueError, TypeError):        # 2026-11-16 sampled runs)
            pass
        try:
            ok = int(pred) == int(sample["answer"])
        except ValueError:
            ok = False
        return {"pred": pred[:40], "correct": bool(ok),
                "extraction": "boxed" if boxes else "fallback"}
