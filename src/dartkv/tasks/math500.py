"""MATH-500 task — the 500-problem test subset of MATH (Lightman et al.
2023, "Let's Verify Step by Step"), as used by KVarN Tab.1 (Qwen3-4B,
thinking mode, Avg@3 → we approximate with greedy per docs/benchmark.md).

Data: data/math500/test.jsonl — official HuggingFaceH4/MATH-500 schema
{problem, solution, answer, subject, level, unique_id}, fetched via
ModelScope mirror (AI-ModelScope/MATH-500); fields verified 2026-11-10.

Scoring: last \\boxed{...} (balanced-brace) in the generation vs `answer`
after math normalization. Normalizer validated 2026-11-10: the original
exact-match version under-counted by 6-9 pts/run (full 77.0 vs paper 82.6);
every flip of the fixed version was eyeballed on the m100 matrix
(docs/experiments.md). Rules: \\text{}/\\mathrm{} wrappers, braceless
\\sqrt2, order-insensitive comma sets, "x=" prefixes, base subscripts
(52_8), ^\\circ/degrees, thousands commas, dfrac/tfrac, \\left\\right.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import shutil

_HERE = pathlib.Path(__file__).resolve().parent

_STRIP_PAIRS = [
    ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\ ", ""), ("$", ""),
    ("\\left", ""), ("\\right", ""),
    ("dfrac", "frac"), ("tfrac", "frac"),
    ("\\{", "{"), ("\\}", "}"),
]


def _norm(s: str) -> str:
    for a, b in _STRIP_PAIRS:
        s = s.replace(a, b)
    s = s.replace("\\text{", "{").replace("\\mathrm{", "{")
    s = s.replace("^\\circ", "").replace("\\circ", "").replace("°", "")
    s = s.replace("\\%", "").replace("\\degree", "")
    s = re.sub(r"\\sqrt\s*(\d)", r"\\sqrt{\1}", s)   # \sqrt2 -> \sqrt{2}
    s = re.sub(r"\\frac\s*(\d)\s*(\d)", r"\\frac{\1}{\2}", s)  # \frac12
    s = s.replace("{,}", ",")                        # 1{,}000 -> 1,000
    s = re.sub(r"^\s*[a-zA-Z]\s*=\s*", "", s)        # x=5 -> 5
    s = re.sub(r"(\d),(\d{3})\b", r"\1\2", s)        # 1,000 -> 1000
    s = s.rstrip(".").strip()
    s = "".join(s.split())
    while s.startswith("{") and s.endswith("}") and s.count("{") == 1:
        s = s[1:-1]
    while (s.startswith("(") and s.endswith(")") and s.count("(") == 1
           and "," not in s):
        s = s[1:-1]
    return s


def _answers_equal(pred: str, gold: str) -> bool:
    p, g = _norm(pred), _norm(gold)
    if p == g:
        return True
    if "," in p and "," in g:  # order-insensitive comma set: 1,-2 == -2, 1
        if sorted(x for x in p.split(",") if x) == \
           sorted(x for x in g.split(",") if x):
            return True
    if re.sub(r"_\{?\d+\}?$", "", p) == re.sub(r"_\{?\d+\}?$", "", g):
        return True                                 # base subscript: 52_8
    return False


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
        ok = pred is not None and _answers_equal(pred, sample["answer"])
        return {"correct": ok, "pred": pred or ""}
