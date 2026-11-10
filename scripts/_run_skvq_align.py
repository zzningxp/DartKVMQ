"""One-off driver: run the SKVQ L100 alignment trio sequentially on cuda1."""
import os
import subprocess
import sys

PY = "/home/daodao/anaconda3/envs/dart312/bin/python"
BENCH = "/home/daodao/Develop/DartKVMQ/scripts/run_bench.py"
ROD = "configs/skvq_calib/qwen3-4b_wikitext2_rod_g128.pt"
BASE = ["--model", "Qwen/Qwen3-4B", "--method", "skvq", "--task",
        "line_retrieval", "--lines", "100", "--n", "30", "--greedy",
        "--bits", "2", "--group", "128", "--sink", "5", "--skvq-clip",
        "0.92"]

RUNS = [
    ("skvq_paper", ["--window", "128", "--reorder-file", ROD]),
    ("skvq_noreod", ["--window", "128"]),
    ("skvq_w0", ["--window", "0", "--reorder-file", ROD]),
]

env = dict(os.environ, HF_ENDPOINT="https://hf-mirror.com",
           CUDA_VISIBLE_DEVICES="1")
for tag, extra in RUNS:
    print(f"===== {tag} =====", flush=True)
    r = subprocess.run([PY, BENCH] + BASE + extra + ["--tag", tag],
                       cwd="/home/daodao/Develop/DartKVMQ", env=env)
    if r.returncode != 0:
        sys.exit(f"{tag} failed with {r.returncode}")
print("ALL DONE")
