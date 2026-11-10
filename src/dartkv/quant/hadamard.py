"""Hadamard transform (normalized, symmetric) for incoherence processing.

H is built via Sylvester construction for power-of-2 sizes, normalized by
2^(-k/2), giving H = H^T and H @ H = I. For non power-of-2 head dims we pad to
the next power of two and slice (approximation; note slicing breaks exact
orthogonality, so we only support power-of-2 dims here — Llama/Qwen d=128 OK).
"""
from __future__ import annotations

import torch


def hadamard_matrix(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
        )
    return H / (n**0.5)


def hadamard(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Apply H along `dim`: out = x * H (contract over dim)."""
    n = x.shape[dim]
    H = hadamard_matrix(n, device=x.device, dtype=torch.float32)
    return (x.float() @ H).to(x.dtype)


def hadamard_transpose(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Apply H^T along `dim` (H symmetric => same as `hadamard`)."""
    return hadamard(x, dim)
