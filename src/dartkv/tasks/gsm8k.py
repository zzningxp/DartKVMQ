"""GSM8K task — Core benchmark #1 (cheap math sanity; no direct paper
target, internal regression signal per docs/benchmark.md).

Protocol: subsample (first n, deterministic) of the GSM8K test split
(1319 problems), greedy, max_new=256, exact-match on the final numeric
answer (GSM8K official: number after '####'; fall back to the LAST
number in the generation).

Data: data/gsm8k/test.jsonl ({question, answer-with-CoT}). Populated
once from the local HF-cache mirror of openai/gsm8k (already on disk
via hf-mirror, so no network); a ModelScope mirror works too if ever
needed (search 'gsm8k').
"""
from __future__ import annotations

import json
import os
import pathlib
import re

_HERE = pathlib.Path(__file__).resolve().parent
_NUM_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def _to_float(s: str) -> float:
    return float(s.replace(",", "").replace("$", ""))


def _final_number(text: str) -> str | None:
    """Number after the last '####' if present, else the last number."""
    if "####" in text:
        text = text.split("####")[-1]
    nums = _NUM_RE.findall(text)
    return nums[-1] if nums else None


class GSM8KTask:
    name = "gsm8k"

    def __init__(self, data_file: str | None = None):
        p = pathlib.Path(data_file) if data_file else (
            _HERE.parents[2] / "data" / "gsm8k" / "test.jsonl")
        if not p.exists():  # one-time pull from the local HF cache
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from datasets import load_dataset
            ds = load_dataset("openai/gsm8k", "main", split="test")
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w") as f:
                for r in ds:
                    f.write(json.dumps(
                        {"question": r["question"], "answer": r["answer"]})
                        + "\n")
        self.rows = [json.loads(l) for l in p.read_text().splitlines()
                     if l.strip()]

    def samples(self, n: int):
        n = min(n, len(self.rows))
        return [{"prompt": r["question"],
                 "answer": r["answer"].split("####")[-1].strip(),
                 "meta": {"i": i}}
                for i, r in enumerate(self.rows[:n])]

    @staticmethod
    def score(sample, generated_text: str):
        pred = _final_number(generated_text)
        ok = False
        if pred is not None:
            try:
                ok = abs(_to_float(pred)
                         - _to_float(sample["answer"])) < 1e-4
            except ValueError:
                ok = pred.strip() == sample["answer"].strip()
        return {"correct": ok, "pred": pred or ""}
