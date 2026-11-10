"""scratch: MATH500 task scoring checks (run: python scripts/_test_math500_scratch.py)"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from dartkv.tasks.math500 import MATH500Task  # noqa: E402

t = MATH500Task()
ss = t.samples(5)
print("n rows:", len(t.rows), "| meta:", ss[0]["meta"])

s = {"answer": r"\left( 3, \frac{\pi}{2} \right)"}
cases = {
    "boxed-match": r"The answer is \boxed{(3, \frac{\pi}{2})}.",
    "dfrac-left":  r"Answer: $\boxed{\left(3, \dfrac{\pi}{2}\right)}$",
    "no-boxed":    "no box here",
    "last-boxed":  r"think \boxed{wrong} then \boxed{(3, \frac{\pi}{2})}",
    "unbalanced":  r"truncated \boxed{(3, \frac{\pi}{2}",
    "diff-answer": r"\boxed{\text{yes}}",
}
for name, g in cases.items():
    print(name, "->", MATH500Task.score(s, g))

s2 = {"answer": "0.5"}
print("plain:", MATH500Task.score(s2, r"\boxed{0.5}"))
