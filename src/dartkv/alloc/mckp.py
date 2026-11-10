"""RDKV bit allocation: MCKP via Lagrangian bisection + reverse water-filling.

Units u (tokens for V, channels for K) get bit-widths b_u in B = {0,2,4,8,16}
minimizing  sum_u w_u * eps_u(b_u)  s.t. sum_u b_u <= budget, where eps is a
per-unit distortion table measured per model (RDKV Sec. 3.1 / Alg. 2).
"""
from __future__ import annotations

import torch


BIT_SET = [0, 2, 4, 8, 16]


def mckp_lagrangian(
    w: torch.Tensor,
    eps_table: dict[int, float] | torch.Tensor,
    budget: float,
    bit_set: list[int] | None = None,
    tol: float = 1e-2,
    max_iter: int = 64,
) -> torch.Tensor:
    """Allocate bits to units with weights w (any shape) under a total budget.

    eps_table: mapping bit-width -> per-coordinate distortion (relative),
        including eps(0)=1.0 (eviction) and eps(16)=0.0 (full precision).
    budget: max of sum(b_u) over all units (in bits).

    Returns long tensor of chosen bit-widths with same shape as w.
    """
    bits = bit_set if bit_set is not None else BIT_SET
    if isinstance(eps_table, dict):
        eps = torch.tensor([float(eps_table[b]) for b in bits],
                           device=w.device, dtype=torch.float32)
    else:
        eps = eps_table.to(w.device, torch.float32)
    bts = torch.tensor(bits, device=w.device, dtype=torch.float32)
    B, = bts.shape
    assert (bts.sum(dim=0) * 0 + budget) >= 0

    wf = w.float().reshape(-1, 1)  # (U, 1)
    # cost[u, j] = w_u * eps(bits[j]); rate[j] = bits[j]
    cost = wf * eps.unsqueeze(0)  # (U, B)
    rate = bts  # (B,)

    # lambda bounds: cost/Rate slope; lambda=0 -> pick min-cost (usually b=16,
    # eps=0); lambda huge -> pick min rate with min cost tie.
    lo = torch.zeros((), device=w.device)
    hi = (cost / rate.clamp_min(1e-12)).max() * 2 + 1.0

    best_alloc = None
    for _ in range(max_iter):
        lam = (lo + hi) / 2
        obj = cost + lam * rate.unsqueeze(0)  # (U, B)
        choice = obj.argmin(dim=1)  # (U,)
        total = rate[choice].sum()
        best_alloc = choice
        if abs(total - budget) / max(budget, 1e-9) < tol:
            break
        if total > budget:
            lo = lam  # need higher lambda to reduce bits
        else:
            hi = lam
    alloc_bits = bts[best_alloc].long().reshape(w.shape)

    # feasibility polish: if still over budget after max_iter, greedily demote
    # the units with the smallest marginal cost increase.
    total = alloc_bits.sum().item()
    if total > budget:
        order = torch.argsort(w.float().reshape(-1), descending=False)
        flat = alloc_bits.reshape(-1)
        for u in order.tolist():
            if total <= budget:
                break
            j = bits.index(int(flat[u].item()))
            if j > 0:
                total -= bits[j] - bits[j - 1]
                flat[u] = bits[j - 1]
        alloc_bits = flat.reshape(w.shape)
    return alloc_bits


def bennett_reverse_waterfilling(
    w: torch.Tensor, sigma: torch.Tensor, budget: float
) -> torch.Tensor:
    """Continuous relaxation (Thm 3.3): b* = [log2(ln2 * w * sigma / lam)]_+,
    lam chosen so sum b* == budget. Returns float bit-widths (can be <0->0)."""
    c = (w.float() * sigma.float()).clamp_min(1e-30).reshape(-1)
    import math

    def total_bits(lam: float) -> float:
        return torch.clamp(torch.log2(math.log(2) * c / lam), min=0).sum().item()

    lo, hi = 1e-30, (math.log(2) * c.max()).item() + 1e-30
    for _ in range(100):
        mid = (lo * hi) ** 0.5  # geometric bisection (log-space)
        if total_bits(mid) > budget:
            lo = mid
        else:
            hi = mid
    lam = (lo * hi) ** 0.5
    return torch.clamp(torch.log2(math.log(2) * c / lam), min=0).reshape(w.shape)
