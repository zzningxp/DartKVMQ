"""Failure-verification exams (失效预测验证考卷, 2026-11-13 user direction).

Derivation table lives in docs/experiments.md「失效预测与验证考卷」. Each
exam targets a *predicted* failure mode of KV compression, with predictions
frozen before running:

  multikey  — many same-format keys, ask one value (line_retrieval with
              dense key interference). Predicted: K-quant collapses harder
              than line_retrieval (score-margin story); eviction fine.
  chain     — magic value passed along a chain of variables; chain length
              = number of exact matches required (RULER variable_tracking).
              Predicted: K-quant damage compounds with hop count.
  wordfreq  — one word repeated among many; ask the most frequent word
              (RULER frequent/common words). Info spread thin over the whole
              context. Predicted: eviction collapses, quantization fine.
  advline   — line_retrieval where D distractor codes differ from the
              target code by exactly ONE character. Predicted: K-quant
              error rate explodes vs plain line_retrieval at same L.

All exams are programmatic, seeded, exact-match scored; scale ~1-1.5k
tokens at default sizes so they are directly comparable to
line_retrieval L100 and each run stays a short GPU burst.
"""
from __future__ import annotations

import random
import re

_HEX = "0123456789abcdef"
_CONS = "bdfgklmnprstvz"
_VOW = "aeiou"
_WORD_RE = re.compile(r"\b[a-z]{5,10}\b")
_HEX8_RE = re.compile(r"\b[0-9a-f]{8}\b")


def _hex8(rng):
    return "".join(rng.choices(_HEX, k=8))


def _word(rng):
    """Tokenizer-friendly pseudo-word, 2-3 syllables."""
    n = rng.randint(2, 3)
    return "".join(c + rng.choice(_VOW) for c in rng.choices(_CONS, k=n))


class MultiKeyRetrievalTask:
    """Same-format key interference: keys are unique 8-hex strings."""
    name = "fail_exam"

    def __init__(self, keys: int = 100, seed: int = 20260604):
        self.keys = keys
        self.seed = seed

    def samples(self, n: int):
        rng = random.Random(self.seed)
        out = []
        for _ in range(n):
            ks = set()
            while len(ks) < self.keys:
                ks.add(_hex8(rng))
            ks = list(ks)
            ws, seen = [], set()
            while len(ws) < self.keys:
                w = _word(rng)
                if w not in seen:
                    seen.add(w)
                    ws.append(w)
            body = "\n".join(f"key {k}: {w}" for k, w in zip(ks, ws))
            k = rng.randrange(self.keys)
            prompt = (
                "Below is a list of access records. Each line names one key "
                "and the word assigned to that key.\n"
                f"{body}\n"
                f"What word is assigned to key {ks[k]}? Reply with only the "
                "word, nothing else.")
            out.append({"prompt": prompt, "answer": ws[k],
                        "meta": {"exam": "multikey", "keys": self.keys}})
        return out

    @staticmethod
    def score(sample, generated_text: str):
        found = _WORD_RE.findall(generated_text.lower())
        pred = found[-1] if found else ""
        return {"correct": pred == sample["answer"], "pred": pred}


class ChainTrackingTask:
    """Hop-count stress: value handed along a chain of variables; all
    statements shuffled so the chain must be traced link by link."""

    name = "fail_exam"

    def __init__(self, hops: int = 2, vars: int = 20, seed: int = 20260604):
        self.hops = hops
        self.vars = vars
        self.seed = seed

    def samples(self, n: int):
        rng = random.Random(self.seed)
        out = []
        for _ in range(n):
            ids = rng.sample(range(100), self.hops + 1 + self.vars)
            chain = ids[: self.hops + 1]      # value flows chain[0] -> ... -> chain[-1]
            others = ids[self.hops + 1:]
            magic = _hex8(rng)
            stmts = [f"The magic value {magic} is assigned to variable "
                     f"VAR_{chain[0]:02d}."]
            for a, b in zip(chain, chain[1:]):
                stmts.append(f"Variable VAR_{a:02d} is assigned to variable "
                             f"VAR_{b:02d}.")
            for v in others:
                stmts.append(f"Variable VAR_{v:02d} is assigned to its own "
                             f"value {_hex8(rng)}.")
            rng.shuffle(stmts)
            prompt = (
                "Below, a magic value is passed along a chain of variables. "
                "All statements are listed in random order.\n"
                + "\n".join(stmts) + "\n"
                f"What is the value of variable VAR_{chain[-1]:02d} at the "
                "end of the chain? Reply with only the 8-character value, "
                "nothing else.")
            out.append({"prompt": prompt, "answer": magic,
                        "meta": {"exam": "chain", "hops": self.hops}})
        return out

    @staticmethod
    def score(sample, generated_text: str):
        found = _HEX8_RE.findall(generated_text.lower())
        pred = found[-1] if found else ""
        return {"correct": pred == sample["answer"], "pred": pred}


class MidNeedleTask:
    """Position-controlled needle: numbered lines of random 10-char codes,
    the asked line placed at a controlled depth (head ~5%, middle ~50%,
    tail ~95%). Tests eviction's probe/recency bias: SnapKV-style scores
    computed from the LAST queries should keep heads & tails and
    under-allocate the middle (lost-in-the-middle for eviction, not for
    the full model). K-quant damage is predicted position-INdependent
    (it tracks candidate similarity, not position)."""

    name = "fail_exam"
    _CODE_RE = re.compile(r"\b[A-Z0-9]{10}\b")

    def __init__(self, lines: int = 200, pos: float = 0.5,
                 seed: int = 20260604):
        self.lines = lines
        self.pos = pos
        self.seed = seed

    def samples(self, n: int):
        rng = random.Random(self.seed)
        out = []
        for _ in range(n):
            codes, seen = [], set()
            while len(codes) < self.lines:
                c = "".join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                                        k=10))
                if c not in seen:
                    seen.add(c)
                    codes.append(c)
            body = "\n".join(f"line {i + 1}: {c}"
                             for i, c in enumerate(codes))
            k = int(self.pos * (self.lines - 1))       # 0-based depth
            prompt = (
                "Below is a list of numbered lines, each with a unique "
                "10-character code.\n"
                f"{body}\n"
                f"What is the code on line {k + 1}? Reply with only the "
                "10-character code, nothing else.")
            out.append({"prompt": prompt, "answer": codes[k],
                        "meta": {"exam": "midneedle", "lines": self.lines,
                                 "pos": self.pos, "depth": k + 1}})
        return out

    @classmethod
    def score(cls, sample, generated_text: str):
        found = cls._CODE_RE.findall(generated_text)
        pred = found[-1] if found else ""
        return {"correct": pred == sample["answer"], "pred": pred}


class FactCoverTask:
    """Long-form generation with uniform contribution: N log entries each
    carry one unique access code; the model must write a summary listing
    every code. Coverage = fraction of codes reproduced. This is the exam
    shape that eviction should fail (evicted entries' codes are
    unrecoverable) while quantization should not (V@2 carries content;
    no near-tie selection). Per-third recall recorded for position
    analysis."""

    name = "fail_exam"

    def __init__(self, entries: int = 60, seed: int = 20260604):
        self.entries = entries
        self.seed = seed

    def samples(self, n: int):
        rng = random.Random(self.seed)
        out = []
        for _ in range(n):
            codes, seen = [], set()
            while len(codes) < self.entries:
                c = _hex8(rng)
                if c not in seen:
                    seen.add(c)
                    codes.append(c)
            body = "\n".join(
                f"Entry {i + 1}: node {_word(rng)} reported status "
                f"{rng.choice(['OK', 'WARN', 'OK', 'OK'])}, access code "
                f"{c}."
                for i, c in enumerate(codes))
            prompt = (
                "Below is a system log. Every entry contains exactly one "
                "access code.\n"
                f"{body}\n"
                "Write a summary of this log as a list of every access "
                "code, one code per line, in any order. Do not omit any "
                "entry's code.")
            out.append({"prompt": prompt, "answer": " ".join(codes),
                        "meta": {"exam": "factcover",
                                 "entries": self.entries,
                                 "codes": codes}})
        return out

    @staticmethod
    def score(sample, generated_text: str):
        codes = sample["meta"]["codes"]
        found = set(_HEX8_RE.findall(generated_text.lower()))
        hits = [c in found for c in codes]
        n3 = max(1, len(codes) // 3)
        return {"correct": sum(hits) / len(codes),
                "pred": f"{sum(hits)}/{len(codes)} codes",
                "cov_early": sum(hits[:n3]) / n3,
                "cov_mid": sum(hits[n3:2 * n3]) / n3,
                "cov_late": sum(hits[2 * n3:]) / max(1, len(hits[2 * n3:]))}


class WordFreqTask:
    """Thin-spread information: one target word repeated far more often than
    fillers; the answer only exists as an aggregate over the whole context."""

    name = "fail_exam"

    def __init__(self, words: int = 400, pool: int = 80,
                 target_freq: int = 25, filler_freq: int = 4,
                 seed: int = 20260604):
        self.words = words
        self.pool = pool
        self.target_freq = target_freq
        self.filler_freq = filler_freq
        self.seed = seed

    def samples(self, n: int):
        rng = random.Random(self.seed)
        out = []
        for _ in range(n):
            vocab, seen = [], set()
            while len(vocab) < self.pool:
                w = _word(rng)
                if w not in seen:
                    seen.add(w)
                    vocab.append(w)
            target = vocab[0]
            fillers = vocab[1:]
            stream = [target] * self.target_freq
            for w in fillers:
                stream += [w] * self.filler_freq
            while len(stream) < self.words:
                stream.append(rng.choice(fillers))
            rng.shuffle(stream)
            rows = [" ".join(stream[i:i + 10])
                    for i in range(0, len(stream), 10)]
            prompt = (
                "Below is a long stream of words. Exactly one word appears "
                "much more often than every other word.\n"
                + "\n".join(rows) + "\n"
                "Which word appears most frequently? Reply with only that "
                "word, nothing else.")
            out.append({"prompt": prompt, "answer": target,
                        "meta": {"exam": "wordfreq", "words": len(stream)}})
        return out

    @staticmethod
    def score(sample, generated_text: str):
        found = _WORD_RE.findall(generated_text.lower())
        pred = found[-1] if found else ""
        return {"correct": pred == sample["answer"], "pred": pred}


class AdvLineRetrievalTask:
    """line_retrieval with D distractor codes at one-character edit distance
    from the target code (random positions, never line k itself)."""

    name = "fail_exam"

    _ALPHABET = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [
        str(d) for d in range(10)]
    _CODE_RE = re.compile(r"\b[A-Z0-9]{10}\b")

    def __init__(self, lines: int = 100, adv: int = 5,
                 seed: int = 20260604):
        self.lines = lines
        self.adv = adv
        self.seed = seed

    def samples(self, n: int):
        rng = random.Random(self.seed)
        out = []
        for _ in range(n):
            codes, seen = [], set()
            while len(codes) < self.lines:
                c = "".join(rng.choices(self._ALPHABET, k=10))
                if c not in seen:
                    seen.add(c)
                    codes.append(c)
            k = rng.randint(1, self.lines)
            target = codes[k - 1]
            # adversarial distractors: 1-char edits of the target, placed on
            # distinct lines (never line k, never an already-edited line)
            free = [j for j in range(self.lines) if j != k - 1]
            rng.shuffle(free)
            placed = 0
            while placed < self.adv and free:
                j = free.pop()
                pos = rng.randrange(10)
                rep = rng.choice(self._ALPHABET)
                if rep == target[pos]:
                    continue
                c = target[:pos] + rep + target[pos + 1:]
                if c not in seen:
                    seen.add(c)
                    codes[j] = c
                    placed += 1
            assert placed == self.adv, "could not place all distractors"
            body = "\n".join(f"line {i + 1}: {c}"
                             for i, c in enumerate(codes))
            prompt = (
                "Below is a list of numbered lines, each with a unique "
                "10-character alphanumeric code.\n"
                f"{body}\n"
                f"What is the code on line {k}? Reply with only the "
                "10-character code, nothing else.")
            out.append({"prompt": prompt, "answer": target,
                        "meta": {"exam": "advline", "lines": self.lines,
                                 "adv": self.adv, "k": k}})
        return out

    @classmethod
    def score(cls, sample, generated_text: str):
        found = cls._CODE_RE.findall(generated_text)
        pred = found[-1] if found else ""
        return {"correct": pred == sample["answer"], "pred": pred}
