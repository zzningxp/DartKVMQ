"""Aggregate results/*.json into a summary table (markdown + csv).

Usage: python scripts/aggregate.py [results_dir ...]
"""
import glob
import json
import os
import sys


def load_all(roots):
    rows = []
    for root in roots:
        for p in sorted(glob.glob(os.path.join(root, "**", "*.json"),
                                  recursive=True)):
            try:
                with open(p) as f:
                    r = json.load(f)
            except Exception as e:
                print(f"skip {p}: {e}")
                continue
            if "summary" not in r:
                continue
            s = r["summary"]
            bits = r.get("bits_report", {})
            rows.append({
                "file": os.path.relpath(p, root),
                "task": s.get("task"),
                "lines": s.get("lines"),
                "model": s.get("model", ""),
                "method": s.get("method"),
                "method_kw": s.get("method_kw"),
                "n": s.get("n"),
                "acc": s.get("acc"),
                "bpe_k": bits.get("bpe_k"),
                "bpe_v": bits.get("bpe_v"),
                "wall_s": round(s.get("wall_s", 0), 1),
            })
    return rows


def main(argv):
    roots = argv[1:] or ["results"]
    rows = load_all(roots)
    if not rows:
        print("no results found")
        return
    cols = ["file", "task", "lines", "model", "method", "method_kw", "n",
            "acc", "bpe_k", "wall_s"]
    # markdown
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r.get(c)) for c in cols) + " |")
    # csv
    out = os.path.join(roots[0], "summary.csv")
    with open(out, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c)) for c in cols) + "\n")
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main(sys.argv)
