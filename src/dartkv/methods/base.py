"""Method interface. Every compression method is a per-run object that owns
the *storage* of all layers and can serve dequantized (K, V) for attention.

Lifecycle (driven by CompressedCache):
  init_run(...)          once per generation run (model config known)
  ingest(l, k, v, first_pass)   every forward chunk, per layer, in layer order
  on_pass_end(kind)      'prefill' or 'decode' — RDKV compresses on prefill end
  get_cache(l) -> (K, V) dequantized bf16 tensors [B, H_kv, T_total, D]
  stats() -> dict        memory accounting (BPE etc.)

Storage is method-owned; CompressedCache only tracks sequence bookkeeping.
Tensors: k, v arrive as [B, H_kv, T_new, D] bf16 (HF layout).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunContext:
    num_layers: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    num_q_heads: int = 0
    dtype: str = "bfloat16"
    device: str = "cuda"


class CompressionMethod:
    name = "base"
    # online methods quantize groups as they seal during decode; offline
    # methods compress once at prefill end
    online = True

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.ctx: RunContext | None = None

    def init_run(self, ctx: RunContext):
        self.ctx = ctx

    def ingest(self, layer_idx: int, k, v, is_first_pass: bool):
        raise NotImplementedError

    def on_pass_end(self, kind: str):  # 'prefill' | 'decode'
        pass

    def get_cache(self, layer_idx: int):
        raise NotImplementedError

    def stats(self) -> dict:
        return {"method": self.name}

    # helper -------------------------------------------------------------
    def bits_report(self) -> dict:
        """Override: {'bpe_k': float|None, 'bpe_v': float|None,
        'fp16_eq_tokens_per_head': float|None}"""
        return {"bpe_k": None, "bpe_v": None, "fp16_eq_tokens_per_head": None}
