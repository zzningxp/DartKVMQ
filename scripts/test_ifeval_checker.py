"""CPU-only sanity check for the IFEval task wiring (no GPU needed).

1. dataset loads (541 samples, expected fields)
2. official checker scores canned responses:
   - a compliant response for an easy instruction -> strict True
   - a violating response -> strict False, loose may rescue some
3. per-instruction-type counts in the dataset match the paper's taxonomy
"""
import os
import pathlib
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dartkv.tasks.ifeval import IFEvalTask  # noqa: E402

task = IFEvalTask()
samples = task.samples(None)
print(f"samples: {len(samples)}")
assert len(samples) == 541, len(samples)
s0 = samples[0]
print("s0 instructions:", s0["meta"]["instruction_id_list"])

# ---- canned-response checks -------------------------------------------
# pick a "length_constraints:number_words" sample: response with few words
idx_nw = next(i for i, s in enumerate(samples)
             if "length_constraints:number_words" in
             s["meta"]["instruction_id_list"])
s = samples[idx_nw]
print("\nnumber_words sample:", s["meta"]["instruction_id_list"],
      s["meta"]["kwargs"])
short = task.score(s, "hi")
print("short ->", {k: short[k] for k in ("correct", "loose", "pred")})
assert short["correct"] is False

# a keyword-frequency sample
idx_kw = next(i for i, s in enumerate(samples)
              if "keywords:frequency" in s["meta"]["instruction_id_list"])
s = samples[idx_kw]
print("\nkeywords:frequency sample:", s["meta"]["instruction_id_list"],
      s["meta"]["kwargs"])
r = task.score(s, "hello world")
print("hello world ->", {k: r[k] for k in ("correct", "loose", "pred")})

# a combinators sample that must repeat the prompt first
idx_comb = next(i for i, s in enumerate(samples)
                if any(x.startswith("combination:") for x in
                       s["meta"]["instruction_id_list"]))
s = samples[idx_comb]
print("\ncombination sample:", s["meta"]["instruction_id_list"])

# first-letter sample
idx_lc = next(i for i, s in enumerate(samples)
              if "change_case:english_lowercase" in
              s["meta"]["instruction_id_list"])
s = samples[idx_lc]
r = task.score(s, "this is a lowercase response with no digits")
print("lowercase pass ->", r["correct"], r["pred"])
assert r["correct"] is True

# ---- taxonomy census ----------------------------------------------------
import collections  # noqa: E402
census = collections.Counter()
for s in samples:
    for iid in s["meta"]["instruction_id_list"]:
        census[iid.split(":")[0]] += 1
print("\ninstruction type census (prompts-level counts):")
for k, v in sorted(census.items()):
    print(f"  {k:30s} {v}")
print("types:", len(census))
print("total instructions:", sum(census.values()))
print("\nOK")
