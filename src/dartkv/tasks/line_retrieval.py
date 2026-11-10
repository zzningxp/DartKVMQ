"""Line-Retrieval task — KVarN paper (arXiv 2606.03458) App. D, task type due
to Li et al. 2023 ([13] there).

Construction (paper, verbatim spec):
  L ∈ {100,200,300,400,500,600} numbered lines, each holding a random
  10-character alphanumeric code drawn uniformly from {A..Z} ∪ {0..9}.
  Ask for the code at a randomly chosen line k; 100 samples per L;
  max_new_tokens=32; exact match on the LAST 10-char alphanumeric token
  found in the output.

Template:
  Below is a list of numbered lines, each with a unique 10-character
  alphanumeric code.
  line 1: <c1>
  ...
  What is the code on line k? Reply with only the 10-character code,
  nothing else.
"""
from __future__ import annotations

import random
import re

_ALPHABET = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [
    str(d) for d in range(10)
]
_CODE_RE = re.compile(r"\b[A-Z0-9]{10}\b")

_HEADER = ("Below is a list of numbered lines, each with a unique "
           "10-character alphanumeric code.")


class LineRetrievalTask:
    name = "line_retrieval"

    def __init__(self, lines: int = 100, seed: int = 20260604):
        self.lines = lines
        self.seed = seed

    def samples(self, n: int):
        rng = random.Random(self.seed)
        out = []
        for _ in range(n):
            codes = ["".join(rng.choices(_ALPHABET, k=10))
                     for _ in range(self.lines)]
            k = rng.randint(1, self.lines)
            body = "\n".join(f"line {i + 1}: {c}"
                             for i, c in enumerate(codes))
            prompt = (f"{_HEADER}\n{body}\n"
                      f"What is the code on line {k}? Reply with only the "
                      f"10-character code, nothing else.")
            out.append({"prompt": prompt, "answer": codes[k - 1],
                        "meta": {"lines": self.lines, "k": k}})
        return out

    @staticmethod
    def score(sample, generated_text: str):
        found = _CODE_RE.findall(generated_text)
        pred = found[-1] if found else ""
        return {"correct": pred == sample["answer"], "pred": pred}

    def extra_report(self, records):
        """Accuracy grouped per line-count if this run sweeps multiple L."""
        return {}
