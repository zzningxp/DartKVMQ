"""Download LongBench data.zip via mirror, extract only line_retrieval jsonl,
print the template + line-count stats.
"""
import json
import os
import re
import zipfile

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import hf_hub_download  # noqa: E402

p = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
print("zip at", p)
zf = zipfile.ZipFile(p)
names = [n for n in zf.namelist() if "line_retrieval" in n]
print("matching entries:", names)
target = [n for n in names if n.endswith("line_retrieval_en.jsonl")][0]
rows = [json.loads(l) for l in zf.read(target).decode().splitlines()]
print("n =", len(rows), "fields:", list(rows[0].keys()))

ex = rows[0]
print("=== context head ===")
print(ex["context"][:1000])
print("=== context tail ===")
print(ex["context"][-800:])
print("=== input ===")
print(repr(ex["input"]))
print("=== answers ===")
print(repr(ex["answers"]))
print("=== length / etc fields ===")
print({k: v for k, v in ex.items() if k not in ("context", "input", "answers")})

ncounts = []
for r in rows:
    m = re.search(r"(\d+) lines", r["context"])
    ncounts.append(int(m.group(1)) if m else -1)
from collections import Counter
print("line-count distribution:", Counter(ncounts))
