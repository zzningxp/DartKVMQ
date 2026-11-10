"""FullKV baseline: no compression; stores fp16/bf16 as-is."""
from __future__ import annotations

from .base import CompressionMethod


class FullMethod(CompressionMethod):
    name = "full"
    online = True

    def init_run(self, ctx):
        super().init_run(ctx)
        self.ks: list = []
        self.vs: list = []

    def ingest(self, layer_idx, k, v, is_first_pass):
        if layer_idx >= len(self.ks):
            self.ks.append(k); self.vs.append(v)
        else:
            self.ks[layer_idx] = torch.cat([self.ks[layer_idx], k], dim=2)
            self.vs[layer_idx] = torch.cat([self.vs[layer_idx], v], dim=2)

    def get_cache(self, layer_idx):
        return self.ks[layer_idx], self.vs[layer_idx]

    def bits_report(self):
        return {"bpe_k": 16.0, "bpe_v": 16.0, "fp16_eq_tokens_per_head": None}


import torch  # noqa: E402  (kept late to keep the docstring first)
