"""scratch: GSM8K task data+scoring checks (run: python scripts/_test_gsm8k_scratch.py)"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from dartkv.tasks.gsm8k import GSM8KTask  # noqa: E402

t = GSM8KTask()
ss = t.samples(3)
print("n rows:", len(t.rows))
print("gold sample:", repr(ss[0]["answer"]))
print("question head:", ss[0]["prompt"][:80])

cases = {
    "official-hash":  "reasoning...\n#### 1,234",
    "no-hash-last":   "So the answer is 42.",
    "hash-then-junk": "#### 72 (seventy-two)",
    "float":          "#### 3.50",
    "none":           "no numbers here",
    "comma-gold-72":  "#### 72",
}
for name, g in cases.items():
    gold = "3.5" if name == "float" else ("72" if "72" in name else "1234")
    print(name, "->", GSM8KTask.score({"answer": gold}, g))
