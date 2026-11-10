"""LongBench task (THUDM/LongBench) — RDKV alignment target (对齐实验 2a).

Protocol follows the official LongBench repo verbatim:
  - prompt format per dataset (config/dataset2prompt.json)
  - middle truncation to max_len tokens (keep both ends)
  - greedy decoding, per-dataset max_new (config/dataset2maxlen.json)
  - per-dataset metric (eval.py dataset2metric), max over ground truths,
    first-line strip for chat-y datasets (trec/triviaqa/samsum/lsht)
  - chat template is NOT applied for trec/triviaqa/samsum/lsht/lcc/
    repobench-p (pred.py: "chat models are better off without build
    prompts on these tasks")
Metrics vendored verbatim in longbench_metrics.py (deps: jieba,
fuzzywuzzy, rouge).

Default subset: EN16 (15 English + qmsum) — RDKV paper "LongBench 16
tasks"; zh datasets selectable explicitly.
"""
from __future__ import annotations

import json
import os
import pathlib

import torch

from .longbench_metrics import (
    qa_f1_score, qa_f1_zh_score, rouge_zh_score, rouge_score,
    classification_score, retrieval_score, retrieval_zh_score,
    count_score, code_sim_score)

_HERE = pathlib.Path(__file__).resolve().parent
_D2P = json.loads((_HERE / "longbench_prompts.json").read_text())
_D2MAXGEN = json.loads((_HERE / "longbench_maxgen.json").read_text())

DATASET2METRIC = {
    "narrativeqa": qa_f1_score, "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score, "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score, "2wikimqa": qa_f1_score,
    "musique": qa_f1_score, "dureader": rouge_zh_score,
    "gov_report": rouge_score, "qmsum": rouge_score,
    "multi_news": rouge_score, "vcsum": rouge_zh_score,
    "trec": classification_score, "triviaqa": qa_f1_score,
    "samsum": rouge_score, "lsht": classification_score,
    "passage_retrieval_en": retrieval_score, "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score, "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}
# pred.py line 63: no chat prompt for these (raw completion)
NO_CHAT = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
# eval.py scorer(): strip to first line for these
FIRST_LINE = {"trec", "triviaqa", "samsum", "lsht"}

EN16 = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
        "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
        "samsum", "passage_count", "passage_retrieval_en", "lcc",
        "repobench-p"]
ZH5 = ["multifieldqa_zh", "dureader", "vcsum", "lsht", "passage_retrieval_zh"]


class LongBenchTask:
    name = "longbench"

    def __init__(self, dataset: str, tokenizer=None, max_len: int = 32768,
                 data_dir: str | None = None):
        self.dataset = dataset
        self.max_len = max_len
        self.needs_chat = dataset not in NO_CHAT
        self.max_gen = _D2MAXGEN[dataset]
        self._rows = self._load(data_dir)

    def _load(self, data_dir):
        """Local jsonl under data/longbench/data/ (auto-downloaded once
        from THUDM/LongBench data.zip; the datasets-lib loader script is
        unsupported in datasets>=3)."""
        root = pathlib.Path(data_dir) if data_dir else (
            _HERE.parents[2] / "data" / "longbench" / "data")
        p = root / f"{self.dataset}.jsonl"
        if not p.exists():
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            import zipfile
            from huggingface_hub import hf_hub_download
            z = hf_hub_download("THUDM/LongBench", "data.zip",
                                repo_type="dataset")
            root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(root.parent)
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        return rows

    def samples(self, n: int):
        fmt = _D2P[self.dataset]
        out = []
        for i, row in enumerate(self._rows[:n]):
            prompt = fmt.format(**row)
            if self.max_len is not None and hasattr(self, "_tok"):
                ids = self._tok(prompt)
                if len(ids) > self.max_len:
                    half = self.max_len // 2
                    prompt = (self._detok(ids[:half])
                              + self._detok(ids[len(ids) - half:]))
            out.append({"prompt": prompt, "answer": row["answers"],
                        "meta": {"dataset": self.dataset, "idx": i,
                                 "length": row.get("length", 0),
                                 "all_classes": row.get("all_classes"),
                                 "answers": row["answers"]}})
        return out

    # tokenizer plumbing (set by run_bench) ------------------------------
    def set_tokenizer(self, tok):
        self._tok = lambda t: tok(t, truncation=False).input_ids
        self._detok = lambda ids: tok.decode(ids, skip_special_tokens=True)

    def score(self, sample, generated_text: str):
        pred = generated_text
        if self.dataset in FIRST_LINE:
            pred = pred.lstrip("\n").split("\n")[0]
        best = 0.0
        for gt in sample["meta"]["answers"]:
            s = DATASET2METRIC[self.dataset](
                pred, gt, all_classes=sample["meta"]["all_classes"])
            best = max(best, float(s))
        return {"correct": best, "pred": pred[:80], "score": best}
