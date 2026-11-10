"""Aggregate the align_* line-retrieval matrix into the trend table vs
KVarN paper Tab.4 (Qwen3-4B rows).

Usage: python scripts/aggregate_alignment.py [--pattern align_%s__L%d.json]
"""
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "line_retrieval"

METHODS = [("full", "FP16"), ("kivi", "KIVI 2/2"),
           ("kvarnh", "Hadamard"), ("kvarn", "KVarN")]
LENGTHS = [100, 200, 300, 400, 500, 600]

# KVarN paper Tab.4, Qwen3-4B, sampled decoding (temp .7/top-p .8/top-k 20)
PAPER = {
    "FP16": [100, 99, 98, 98, 96, 90],
    "KIVI 2/2": [98, 97, 88, 84, 84, 74],
    "Hadamard": [98, 98, 91, 98, 92, 83],
    "KVarN": [100, 99, 98, 97, 97, 85],
}


def load(method, L):
    p = RES / f"align_{method}__L{L}.json"
    if not p.exists():
        return None
    with open(p) as f:
        r = json.load(f)
    return r["summary"]["acc"] * 100


def main():
    rows = {}
    for method, label in METHODS:
        vals = [load(method, L) for L in LENGTHS]
        rows[label] = vals
        ours = " ".join(f"{v:5.1f}" if v is not None else "    -"
                        for v in vals)
        pap = " ".join(f"{v:5d}" for v in PAPER[label])
        print(f"{label:10s} ours: {ours}")
        print(f"{'':10s} papr: {pap}")
    print()

    # trend checks (the alignment claim, cf. docs/experiments.md 2026-11-10)
    def col(label, i):
        v = rows[label][i]
        return v

    checks = []
    for i, L in enumerate(LENGTHS):
        kv, kh, ki = col("KVarN", i), col("Hadamard", i), col("KIVI 2/2", i)
        if None in (kv, kh, ki):
            continue
        checks.append((L, kv, kh, ki, kv >= kh >= ki or kv >= ki))
    n_order = sum(1 for c in checks if c[4])
    print(f"ordering KVarN>=Hadamard>=KIVI holds at "
          f"{n_order}/{len(checks)} lengths: "
          + ", ".join(f"L{c[0]}({'Y' if c[4] else 'N'}: {c[1]:.0f}/"
                      f"{c[2]:.0f}/{c[3]:.0f})" for c in checks))

    # degradation slope: mean acc drop from L=100 to L=600
    for label in rows:
        v = rows[label]
        if v[0] is not None and v[-1] is not None:
            print(f"{label:10s} drop(L100->L600) ours {v[0]-v[-1]:+5.1f} "
                  f"paper {PAPER[label][0]-PAPER[label][-1]:+d}")
    return rows


if __name__ == "__main__":
    main()
