"""Probe whether a model snapshot is complete in the HF cache.
Usage: python scripts/probe_model.py Qwen/Qwen3-4B
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download  # noqa: E402

try:
    p = snapshot_download(sys.argv[1], local_files_only=True)
    print("READY", p)
except Exception as e:
    print("NOT_READY", type(e).__name__, str(e)[:200])
