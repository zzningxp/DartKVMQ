"""Download IFEval input data (google/IFEval) + wikitext via HF mirror."""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import hf_hub_download  # noqa: E402

p = hf_hub_download(repo_id="google/IFEval", filename="ifeval_input_data.jsonl",
                    repo_type="dataset")
print("ifeval:", p)
with open(p) as f:
    lines = f.readlines()
print("samples:", len(lines))
print("first:", lines[:1])
