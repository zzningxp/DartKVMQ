# Vendored LongBench official configs & metrics

Source: https://github.com/THUDM/LongBench (`LongBench/` dir), fetched
2026-11-10. Untouched copies live in `external/longbench_official/`.

- `longbench_metrics.py` = `LongBench/metrics.py` **verbatim** (diff to prove)
- `longbench_prompts.json` = `LongBench/config/dataset2prompt.json` verbatim
- `longbench_maxgen.json` = `LongBench/config/dataset2maxlen.json` verbatim
  (per-dataset max_new_tokens)

Scoring semantics in `dartkv/tasks/longbench.py` follow `LongBench/eval.py`
(`scorer`) and prompt/decoding follow `LongBench/pred.py` (middle
truncation; no chat template on trec/triviaqa/samsum/lsht/lcc/repobench-p;
greedy). Deps: jieba, fuzzywuzzy, rouge.
