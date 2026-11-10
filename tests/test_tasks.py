"""Task-level tests: Line-Retrieval construction + scorer semantics."""
import re
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dartkv.tasks.line_retrieval import LineRetrievalTask


def test_line_retrieval_construction():
    t = LineRetrievalTask(lines=5, seed=7)
    ss = t.samples(3)
    assert len(ss) == 3
    s = ss[0]
    assert s["prompt"].startswith(
        "Below is a list of numbered lines, each with a unique "
        "10-character alphanumeric code.\nline 1: ")
    assert re.fullmatch(r"[A-Z0-9]{10}", s["answer"])
    assert 1 <= s["meta"]["k"] <= 5
    # deterministic given seed
    assert ss == LineRetrievalTask(lines=5, seed=7).samples(3)
    # codes unique within a sample
    codes = set(re.findall(r"[A-Z0-9]{10}", s["prompt"]))
    assert len(codes) == 5


def test_line_retrieval_scorer():
    t = LineRetrievalTask(lines=3, seed=1)
    s = t.samples(1)[0]
    # last-code-wins
    assert t.score(s, s["answer"])["correct"]
    assert t.score(s, f"The code is {s['answer']}")["correct"]
    assert t.score(s, f"AAAA1111BBBB {s['answer']}")["correct"]
    other = "ZZZZZZZZZ9"
    assert not t.score(s, other)["correct"]
    # if two codes appear, the LAST one is the prediction (paper rule)
    rec = t.score(s, f"{other} then the answer is {s['answer']}")
    assert rec["pred"] == s["answer"] and rec["correct"]
    rec = t.score(s, f"{s['answer']} but maybe {other}")
    assert rec["pred"] == other and not rec["correct"]
    # no code at all
    rec = t.score(s, "I don't know")
    assert rec["pred"] == "" and not rec["correct"]


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
