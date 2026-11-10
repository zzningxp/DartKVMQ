"""KVarN iterative dual-axis variance normalization (paper Alg. 1).

Aligned with the official pure-PyTorch reference in
external/KVarN/vllm/model_executor/layers/quantization/kvarn/sinkhorn.py:
  - std (not var) clamped to [1e-3, 1e3]; log-scales clamped to [-0.3, 10.0]
  - update: log_s += log(std)  (== paper's 0.5*log(var))
  - default 16 iterations; best-so-far per-tile tracking on imbalance metric
  - imbalance uses torch .std() (unbiased), matching the reference.

Tile orientation (see reference docstring):
  K tile is [D, group] rows=channels; V tile is [group, D] rows=tokens.
Here we take tiles as (N, R, C) and return (balanced, s_col (N,1,C),
s_row (N,R,1)) with balanced = tile / s_col / s_row.
"""
from __future__ import annotations

import torch

_ITERATIONS = 16
_CLIP_STD_MIN = 1e-3
_CLIP_STD_MAX = 1e3
_LOG_S_MIN = -0.3
_LOG_S_MAX = 10.0


def imbalance(X: torch.Tensor) -> torch.Tensor:
    sc = X.float().std(dim=-2)  # per-column std
    sr = X.float().std(dim=-1)  # per-row std
    return sc.amax(dim=-1) / sc.amin(dim=-1).clamp_min(1e-8) + (
        sr.amax(dim=-1) / sr.amin(dim=-1).clamp_min(1e-8)
    )


def variance_normalize(
    tiles: torch.Tensor,
    iterations: int = _ITERATIONS,
    clip_std: tuple[float, float] = (_CLIP_STD_MIN, _CLIP_STD_MAX),
    log_lo: float = _LOG_S_MIN,
    log_hi: float = _LOG_S_MAX,
):
    """Returns (balanced (N,R,C) fp32, s_col (N,1,C), s_row (N,R,1))."""
    m = tiles.float()
    N, R, C = m.shape
    smin, smax = clip_std

    log_sc = torch.zeros(N, 1, C, device=m.device)
    log_sr = torch.zeros(N, R, 1, device=m.device)

    cur = m / log_sc.exp() / log_sr.exp()
    imb_best = imbalance(cur)
    sc_best = log_sc.exp().clone()
    sr_best = log_sr.exp().clone()

    for _ in range(iterations):
        col_std = cur.std(dim=1, keepdim=True).clamp(smin, smax)  # (N,1,C)
        log_sc = (log_sc + col_std.log()).clamp(log_lo, log_hi)
        cur = m / log_sc.exp() / log_sr.exp()

        row_std = cur.std(dim=2, keepdim=True).clamp(smin, smax)  # (N,R,1)
        log_sr = (log_sr + row_std.log()).clamp(log_lo, log_hi)
        cur = m / log_sc.exp() / log_sr.exp()

        imb = imbalance(cur)
        better = imb <= imb_best
        mask = better.view(N, 1, 1).to(log_sc.dtype)
        sc_best = mask * log_sc.exp() + (1 - mask) * sc_best
        sr_best = mask * log_sr.exp() + (1 - mask) * sr_best
        imb_best = torch.where(better, imb, imb_best)

    return m / sc_best / sr_best, sc_best, sr_best
