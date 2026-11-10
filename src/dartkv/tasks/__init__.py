"""Benchmark tasks. Each task exposes:
    samples(n) -> list of {prompt, answer, meta}
    score(sample, generated_text) -> {correct: bool, ...}
"""
from .line_retrieval import LineRetrievalTask  # noqa: F401
