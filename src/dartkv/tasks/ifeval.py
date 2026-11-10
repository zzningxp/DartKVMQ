"""IFEval task — 541 instruction-following prompts (google/IFEval), scored
with the OFFICIAL checker (vendored verbatim in dartkv.tasks.ifeval_checker).

Paper alignment (KVarN, arXiv 2606.03458 App. D): 541 prompts,
max_new_tokens=1280, three-region layout with S=32 (G=128, R=128);
reports strict + loose prompt-level accuracy. Primary alignment model is
Llama-3.1-8B-Instruct (paper: FP16 71.1/76.8, KIVI 70.9/74.5, Hadamard
70.8/75.0, KVarN 71.0/76.5 strict/loose).

Scoring logic = official evaluation_lib.test_instruction_following_strict /
_loose, called verbatim; per-instruction results land in each record so
per-type breakdowns can be recomputed offline.
"""
from __future__ import annotations

import os

from .ifeval_checker import evaluation_lib

_DEFAULT_REPO = "google/IFEval"
_DEFAULT_FILE = "ifeval_input_data.jsonl"


def _data_path() -> str:
    """Locate ifeval_input_data.jsonl: HF cache first (offline-safe), then
    hub download via mirror."""
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo_id=_DEFAULT_REPO, filename=_DEFAULT_FILE,
                               repo_type="dataset", local_files_only=True)
    except Exception:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        return hf_hub_download(repo_id=_DEFAULT_REPO, filename=_DEFAULT_FILE,
                               repo_type="dataset")


class IFEvalTask:
    name = "ifeval"

    def __init__(self, data_path: str | None = None, seed: int = 0):
        self.data_path = data_path or _data_path()
        self.seed = seed  # unused: dataset is fixed (paper uses all 541)

    def samples(self, n: int):
        inputs = evaluation_lib.read_prompt_list(self.data_path)
        if n is not None:
            inputs = inputs[:n]
        return [{"prompt": ex.prompt, "answer": None,
                 "meta": {"key": ex.key,
                          "instruction_id_list": ex.instruction_id_list,
                          "kwargs": ex.kwargs}}
                for ex in inputs]

    def score(self, sample, generated_text: str):
        ex = evaluation_lib.InputExample(
            key=sample["meta"]["key"],
            instruction_id_list=sample["meta"]["instruction_id_list"],
            prompt=sample["prompt"], kwargs=sample["meta"]["kwargs"])
        p2r = {ex.prompt: generated_text}
        strict = evaluation_lib.test_instruction_following_strict(ex, p2r)
        loose = evaluation_lib.test_instruction_following_loose(ex, p2r)
        return {"correct": strict.follow_all_instructions,
                "loose": loose.follow_all_instructions,
                "pred": f"{sum(strict.follow_instruction_list)}"
                        f"/{len(strict.follow_instruction_list)}",
                "follow_list": strict.follow_instruction_list}
