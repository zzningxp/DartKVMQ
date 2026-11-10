"""Download models/datasets via hf-mirror. Run:
  /home/daodao/anaconda3/envs/dart312/bin/python scripts/download_models.py [name ...]
Names: qwen3-4b, llama31-8b, wikitext. Default: qwen3-4b wikitext.
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download  # noqa: E402
from datasets import load_dataset  # noqa: E402

TARGETS = {
    "qwen3-4b": ("model", "Qwen/Qwen3-4B"),
    "llama31-8b": ("model", "meta-llama/Llama-3.1-8B-Instruct"),
    "wikitext": ("dataset", "Salesforce/wikitext", "wikitext-2-raw-v1"),
}


def main(argv):
    names = argv[1:] or ["qwen3-4b", "wikitext"]
    for name in names:
        spec = TARGETS[name]
        if spec[0] == "model":
            print(f"downloading model {spec[1]} ...", flush=True)
            p = snapshot_download(spec[1])
            print("DONE", name, p, flush=True)
        else:
            print(f"downloading dataset {spec[1]} {spec[2]} ...", flush=True)
            d = load_dataset(spec[1], spec[2])
            print("DONE", name, d, flush=True)


if __name__ == "__main__":
    main(sys.argv)
