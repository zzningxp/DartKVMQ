# Unified Benchmark Specification

One benchmark suite for **all** methods (RDKV / KVarN / RAQ-MHA / ours / baselines). No per-method benchmarks. Two regimes are covered because the three papers target different ones:

| Regime | What it stresses | Methods that live here |
|---|---|---|
| **A. Long-prompt, short-gen** | post-prefill compression quality | RDKV, eviction baselines |
| **B. Short-prompt, long-gen** | decode-time error accumulation | KVarN, RAQ-MHA, online quant |

Every method is evaluated on **both** regimes (a quantization method just quantizes the prompt cache too; an eviction method just evicts once post-prefill and appends FP16 during decode — RDKV's Zone-C semantics).

## Tasks (priority order)

### Core (fast iteration, run per commit)
1. **GSM8K** (subsample 200) — cheap math sanity. Metric: exact-match final number. Greedy. max_new=256.
2. **MATH500** — 500 problems, `\boxed{}` extraction. Greedy (vLLM-free parity with KVarN's Avg@3 approximated by greedy + note). max_new=4096 (Llama-3.1-8B non-thinking; paper used 8192 for Qwen3 thinking).
3. **Line-Retrieval** (KVarN App. D protocol, implemented in `src/dartkv/tasks/line_retrieval.py`): L∈{100,200,300,400,500,600} numbered lines (`line i: <10-char code>`), codes uniform over {A..Z}∪{0..9}; query = random line k; template per paper; 100 samples/L; max_new=32; exact match on the **last** 10-char alphanumeric token in the output. Decoding per model preset (Qwen3 non-thinking: temp 0.7/top-p 0.8/top-k 20; Llama-3.1: temp 0.6/top-p 0.9). Alignment targets: KVarN Tab.4 (Qwen3-4B: FP16 100/99/98/98/96/90, KIVI 98/97/88/84/84/74, Hadamard 98/98/91/98/92/83, KVarN 100/99/98/97/97/85; Llama-3.1-8B: FP16 100/99/99/96/93/92, KIVI 98/97/90/84/80/83, KVarN 99/98/98/95/91/89).
4. **RULER-NIAH subset**: S-NIAH-1/3 + MW-NIAH at 4k/16k/32k, 100 samples each (start 50). Metric: substring accuracy. → regime A.

### Full (alignment + reporting)
5. **HumanEval** (164, pass@1, execution-graded, greedy). 
6. **AIME24** (30 problems, Avg@3, temp 0.6/top-p 0.95 for Qwen3-thinking; Llama greedy). max_new=16384. Expensive — run once per method config.
7. **LongBench subset** (alignment with RDKV): all 16 tasks, official prompts, @budgets {64,128,512,1024}L. Chinese tasks (Zh.QA etc.) included.
8. **IFEval** (541, strict/loose, greedy, max_new=1280).

Dropped: InfiniteBench (2M-token tasks infeasible on 32GB + low marginal info vs RULER@128k), MultiQA, GPQA (MLA-paper specific, phase 2).

## Unified reporting axes

Every result row carries:
- `method`, `model`, `task`, `seed(s)`
- **Memory axis**: `bpe_k`, `bpe_v` (effective bits per cached element incl. scale/zp metadata, RAQ-MLA Eq.16 accounting) AND `fp16_eq_tokens_per_layer_head` (RDKV budget n). Both reported so budget- and bits-driven methods compare.
- **Timing** (phase P5 only): TTFT overhead, ms/tok decode.

Fairness rule: methods are compared at **matched memory** (e.g. 2.3 BPE ≈ 7× token budget ≈ …), plus each method's own paper operating point.

## Engine & numerics

- HuggingFace `transformers==5.20.1`, `attn_implementation="sdpa"`, bf16 compute, `device=cuda:1` (`CUDA_VISIBLE_DEVICES=1`).
- Compressed caches are **dequantized to bf16 before dense attention** — numerically equivalent to fused-dequant kernels up to fp rounding; this is the accuracy-only phase. Real packed storage is implemented (for BPE accounting + later speed), but attention reads reconstructed tensors.
- Sampling: greedy everywhere unless stated (deterministic across methods; paper parity runs use the paper's params).
- Chat template per model's default; task prompts as in each task's official repo.

## Models

| Model | Role | Notes |
|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | primary (all three papers use it) | GQA, 32 layers, 8 KV heads, d=128; 128k ctx. Needs HF token (gated). |
| `Qwen/Qwen3-4B` | secondary (decode-accumulation: thinking mode) | GQA 32/8; KVarN main model; fp16/bf16 ~8GB. |
| Fallback if Llama gated: `mistralai/Mistral-7B-Instruct-v0.3` or `Qwen/Qwen2.5-7B-Instruct` | RDKV also evaluates Mistral-7B | decide at P0.5 |

## Output format

`results/<date>_<method>_<model>_<task>.json`:
```json
{"method":"kvarn-2bit", "model":"...", "task":"math500", "seed":0,
 "bpe_k":2.25, "bpe_v":2.25, "fp16_eq_tokens": null,
 "acc": 0.792, "n": 500, "runtime_s": 1234, "config_hash": "..."}
```
Aggregation script produces `results/summary.csv` + markdown tables for docs/status/.
