"""Post-RoPE reorder probes: does cache-matched calibration rescue K@2?"""
import os
import subprocess
import sys

PY = "/home/daodao/anaconda3/envs/dart312/bin/python"
BENCH = "/home/daodao/Develop/DartKVMQ/scripts/run_bench.py"
ROD = "configs/skvq_calib/qwen3-4b_wikitext2_rod_g128_postrod.pt"
BASE = ["--model", "Qwen/Qwen3-4B", "--method", "skvq", "--task",
        "line_retrieval", "--lines", "100", "--n", "30", "--greedy",
        "--group", "128", "--sink", "5", "--skvq-clip", "0.92",
        "--window", "128", "--reorder-file", ROD]

RUNS = [
    ("skvq_k2v16_postrod", ["--bits-k", "2", "--bits-v", "16"]),
    ("skvq_22_postrod", ["--bits", "2"]),
    ("skvq_k2v16_postrod_g32", ["--bits-k", "2", "--bits-v", "16",
                                "--group", "32",
                                "--reorder-file",
                                "configs/skvq_calib/"
                                "qwen3-4b_wikitext2_rod_g32_postrod.pt"]),
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
