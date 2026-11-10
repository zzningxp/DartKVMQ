"""scratch driver: MATH500 n=100 x {full,kivi,kvarn_h,kvarn} on Qwen3-4B.

Run: /home/daodao/anaconda3/envs/dart312/bin/python scripts/_run_math500_matrix.py
(对齐实验 1c core-iteration size; paper Tab.1 full-500 runs come later)
"""
import pathlib
import subprocess
import sys

PY = sys.executable
HERE = pathlib.Path(__file__).resolve().parent
RUN = HERE / "run_bench.py"

for m in ["full", "kivi", "kvarn_h", "kvarn"]:
    cmd = [PY, str(RUN), "--model", "Qwen/Qwen3-4B", "--method", m,
           "--task", "math500", "--thinking", "--greedy", "--n", "100",
           "--tag", f"m100_{m}"]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
print("ALL DONE")
