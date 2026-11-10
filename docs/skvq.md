# SKVQ: Analysis & Integration

Source: [SKVQ: Sliding-window Key and Value Cache Quantization for LLMs](https://arxiv.org/abs/2405.06219) (Duanmu et al., v3 2024-11). Official code lives in [`external/SKVQ/`](../external/SKVQ/) (kept pristine as ground truth; **untracked local snapshot** — like external/KVarN, it is not in git; the snapshot came from the collaborator's tree and is recoverable from git history `HEAD:SKVQ/`). Our port: `src/dartkv/quant/skvq.py` + `src/dartkv/methods/skvq.py`, proven numerically identical to the official engine (see `tests/test_skvq.py`; tests skip when the snapshot is absent).

## Method in one paragraph

SKVQ quantizes the KV cache to ~2 bits by attacking **channel outliers with grouping, not transforms**: an offline calibration KMeans-clusters channels by their (min, max) statistics and permutes them ("reorder") so each quantization group holds similar-magnitude channels. On top it uses (a) **clipped dynamic asymmetric quantization** (group min/max × clip ≈ 0.92) per token, (b) a **sliding window**: the most recent W=128 tokens stay FP16 (exactly W at all times during decode — one token is evicted into the quantized body per step), (c) a tiny FP16 **attention sink** (S=5). Paper claims 2-bit K + 1.5-bit V with minimal loss; scales/zeros stored fp8 in the headline configs.

## What makes it different from the others

| Axis | KIVI | KVarN | **SKVQ** | RDKV |
|---|---|---|---|---|
| Outlier handling (K) | per-channel scale over time | Hadamard rotation + VarN (mix outliers into all dims) | **offline channel permutation → similar channels share a group** | n/a (eviction) |
| K quant unit | per channel, temporal groups of G=128 tokens | per channel, per temporal tile, rotated space | **per token, channel groups (g=128, may span heads, variable-size after reorder)** | n/a |
| V quant unit | per token, per head | per token, per tile, rotated | **per token, channel groups + reorder + clip; supports 1.5-bit (3 levels)** | n/a |
| FP16 regions | residual tail (sawtooth <G) | sink S + tail (sawtooth <G) | **sink S=5 + exact sliding window W=128 (evict-one-at-a-time)** | keep-set of n tokens |
| Calibration | none | none (transform is calibration-free) | **required**: wikitext2 min/max per channel → KMeans reorder file | none |
| Extra knobs | — | varn_iters, rtn_quantile | clip (0.91–0.97, sensitive), fp8 scales, sink | budget n |
| bpe @2-bit | 2.25 (fp16 scale+zp /G) | 2.25 (2×fp8 + fp16 zp /G) | **2.25 fp16 / 2.125 fp8 scales** + (S+W) FP16 tokens | 16 × n/T |
| Decode-time quant cost | seal per G tokens | per tile (VarN iters) | **per evicted token (trivial)** | one-shot prefill end |

Positioning: KIVI solves K outliers with per-channel temporal scales but pays with coarse 4-level steps along time; KVarN solves them by rotating the outlier structure away entirely (transform per tile, inverted at readout); SKVQ solves them by **rearranging channels so outliers quantize together** — no runtime transform, but needs an offline calibration artifact per model+gsize and a sensitive clip hyperparameter. The exact sliding window (vs our KIVI/KVarN sawtooth tail) is a structural difference: SKVQ always keeps the most recent W tokens exact, which matters most for decode-heavy workloads (their regime: 32k-1M contexts, needle + LongBench).

## Integration (bench spec `skvq`)

```
python scripts/run_bench.py --model Qwen/Qwen3-4B --method skvq \
    --task line_retrieval --lines 100 --n 30 --greedy --sink 5 \
    --window 128 --group 128 --skvq-clip 0.92 \
    --reorder-file configs/skvq_calib/qwen3-4b_wikitext2_rod_g128_postrod.pt
```

Use the `_postrod.pt` files on this harness (stats of the actually-cached post-RoPE tensors); the plain files replicate the official pre-RoPE calibration exactly and pair with their pre_rope=True engine.

- `--group` doubles as the channel group size g; `--window` = W (0 disables the sliding window, official w0 ablation); `--reorder-file` absent = contiguous groups (official no-reorder ablation).
- `--bits-v 1.5` supported (3 levels); `--skvq-fp8` for e4m3 scale/zp storage.
- Calibration: `scripts/calibrate_skvq.py` (official recipe: wikitext-2, seed 42, per-channel min/max hooks → KMeans n_init=10; writes official-format `.pt` to `configs/skvq_calib/`).

### What is intentionally NOT ported
- `pre_rope` — their engine can cache pre-RoPE keys and apply RoPE per step (paper's main config). Our harness caches post-RoPE keys: we run the official `pre_rope=False` setting (their ablation shows pre-RoPE helps K mildly).
- `KIVI_mode` — their engine's KIVI reproduction; our bench has native `kivi`.
- CUDA packed kernels — accuracy phase uses fake quant by design (same policy as kivi/kvarn rows in `docs/benchmark.md`); official README confirms paper numbers are fake-quant too.
- SmoothQuant path (commented out upstream, not in the paper mainline) and the experimental `turboquant_backend` hook (TQ is a separate integration workitem).

### Alignment evidence (对齐实验)
1. **Codec**: our `quant_dequant_channels` vs official `SKVQuantProcessor.quant_pytorch` — fp32 allclose, **bf16 bitwise-equal**, across {no-reorder, reorder+variable clusters, fp8, 1.5-bit}.
2. **Layout**: our `SkvqMethod` vs official `SlidingKVCacheManager` driven through prefill + every decode step — caches agree at every step (plain & reorder configs).
3. **End-task** (Qwen3-4B, L100, n=30, greedy; full table in `docs/experiments.md` 2026-11-10):
   - 16/16 = 93.3 ≈ full → integration exact.
   - V@2 nearly free (83.3), **beats KIVI's V@2 (76.7)** — the reorder+clip V mechanism works.
   - SKVQ 2/2 > KIVI 2/2 (paper direction ✓): 63.3 (g32) / 23.3 (g128) vs 13.3.
   - **Calibration must match the cached tensor**: the official k_proj hook is PRE-RoPE (their engine caches pre-RoPE keys); on our post-RoPE harness use `--hook cache` reorder files (`*_postrod.pt`), which rescues K@2 from 0.0 → 23.3 (g128) / 83.3 K2/V16 (g32).
   - Qwen3's QK-norm channel outliers demand fine channel groups: paper's g128 leaves K@2 weak post-RoPE; g32 (3.0 bpe) works.
