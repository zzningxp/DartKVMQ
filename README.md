# DartKVMQ

KV-Cache Management & Quantization research codebase.

## Goal

Reproduce and unify three recent KV-cache compression methods under **one codebase + one benchmark suite**, targeting **standard MHA models** (e.g. Llama-family), then iterate our own method on top:

| # | Method | arXiv | Type | Source |
|---|--------|-------|------|--------|
| 1 | **RDKV** | [2605.08317](https://arxiv.org/abs/2605.08317) | Joint eviction + quantization via rate-distortion bit allocation (post-prefill, reverse water-filling) | paper only |
| 2 | **RAQ-MLA** | [2610.33625](https://arxiv.org/abs/2610.33625) | Role-aware quantization (K → score error, V → value error) + fitted invertible transform; adapt from MLA to MHA | paper only |
| 3 | **KVarN** | [2606.03458](https://arxiv.org/abs/2606.03458) | Calibration-free Hadamard rotation + dual-axis variance normalization, 2-bit, fixes decode-time error accumulation | [huawei-csl/KVarN](https://github.com/huawei-csl/KVarN) (vLLM impl) |
| 4 | ours | — | TBD | — |

## Layout

```
docs/        roadmap, paper notes, design decisions, experiment logs
external/    upstream snapshots (KVarN)
src/dartkv/
  quant/     asymmetric uniform quant, Hadamard, VarN (aligned w/ official ref)
  alloc/     MCKP Lagrangian + Bennett reverse water-filling (RDKV)
  methods/   full / kivi / kvarn (three-region grouped layout)
  cache/     CompressedCache — HF transformers Cache routing through a method
  tasks/     benchmark tasks (line_retrieval first)
  bench.py   unified runner: model x method x task -> results/*.json
scripts/     entry points (run_bench.py, smoke_cache.py, download_models.py)
configs/     experiment configs
tests/       unit tests (CPU, no model downloads)
results/     experiment outputs (json/csv/plots)
```

## Quick start

```bash
PY=/home/daodao/anaconda3/envs/dart312/bin/python
# unit tests (CPU)
$PY tests/test_methods.py && $PY tests/test_quant_alloc.py && $PY tests/test_tasks.py
# Line-Retrieval pilot (KVarN alignment experiment 1a; cuda1 + hf-mirror)
$PY scripts/run_bench.py --model Qwen/Qwen3-4B --method kvarn --lines 100 200 --n 20 --tag pilot
```

See [ROADMAP.md](ROADMAP.md) for the plan and current status.
