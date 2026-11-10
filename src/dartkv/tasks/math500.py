"""MATH-500 task — the 500-problem test subset of MATH (Lightman et al.
2023, "Let's Verify Step by Step"), as used by KVarN Tab.1 (Qwen3-4B,
thinking mode, Avg@3 → we approximate with greedy per docs/benchmark.md).

Data: data/math500/test.jsonl — official HuggingFaceH4/MATH-500 schema
{problem, solution, answer, subject, level, unique_id}, fetched via
ModelScope mirror (AI-ModelScope/MATH-500); fields verified 2026-11-10.

Scoring: last \\boxed{...} (balanced-brace) in the generation, light
LaTeX normalization, exact string match against `answer`.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import shutil

_HERE = pathlib.Path(__file__).resolve().parent

# light normalization shared by pred and gold (pilot-grade; extend if the
# 1c alignment shows systematic format misses)
_STRIP_PAIRS = [
    ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\ ", ""), ("$", ""),
    ("\\left", ""), ("\\right", ""),
    ("dfrac", "frac"), ("tfrac", "frac"),
    ("\\{", "{"), ("\\}", "}"),
]


def _norm(s: str) -> str:
    for a, b in _STRIP_PAIRS:
        s = s.replace(a, b)
    return "".join(s.split())


def last_boxed(text: str) -> str | None:
    """Content of the LAST \\boxed{...} with balanced braces."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None  # unbalanced (truncated generation)


class MATH500Task:
    name = "math500"

    def __init__(self, data_file: str | None = None):
        p = pathlib.Path(data_file) if data_file else (
            _HERE.parents[2] / "data" / "math500" / "test.jsonl")
        if not p.exists():  # one-time pull (HuggingFaceH4/MATH-500, ungated,
            # hf-mirror reachable; same pattern as gsm8k/longbench tasks)
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from huggingface_hub import hf_hub_download
            src = hf_hub_download("HuggingFaceH4/MATH-500", "test.jsonl",
                                  repo_type="dataset")
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, p)
        self.rows = [json.loads(l) for l in p.read_text().splitlines()
                     if l.strip()]

    def samples(self, n: int):
        n = min(n, len(self.rows))
        return [{"prompt": r["problem"], "answer": str(r["answer"]),
                 "meta": {"subject": r["subject"], "level": r["level"],
                          "unique_id": r["unique_id"]}}
                for r in self.rows[:n]]

    @staticmethod
    def score(sample, generated_text: str):
        pred = last_boxed(generated_text)
        ok = pred is not None and _norm(pred) == _norm(sample["answer"])
        return {"correct": ok, "pred": pred or ""}
