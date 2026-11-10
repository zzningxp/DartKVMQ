"""Analyze Line-Retrieval failures: is the prediction another line's code
(misretrieval) or a hallucinated code? Where are the failing k values?"""
import json
import re
import sys

path = sys.argv[1]
r = json.load(open(path))
recs = r["records"]
fails = [x for x in recs if not x["correct"]]
print(f"n={len(recs)} acc={1 - len(fails) / len(recs):.3f} fails={len(fails)}")
for f in fails:
    print(f"  i={f['i']:3d} k={f['meta']['k']:3d} answer={f['answer']} "
          f"pred={f['pred']} gen={f['gen_text'][:40]!r}")
# k distribution of failures vs all
ks_all = [x["meta"]["k"] for x in recs]
ks_fail = [f["meta"]["k"] for f in fails]
print("k of fails:", sorted(ks_fail))
print("mean k all/fail:", sum(ks_all) / len(ks_all),
      sum(ks_fail) / max(1, len(ks_fail)))
