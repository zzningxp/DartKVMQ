"""KIVI baseline: per-channel K / per-token V asymmetric quantization,
FP16 open tail (no sink). 2-bit default, G-token groups along the quant axis.

Chunk storage (M = number of G-token groups in the chunk):
  K codes [B,H,D,M,G] with scale/zero [B,H,D,M,1]   (per channel per group)
  V codes [B,H,M*G,D] with scale/zero [B,H,M*G,1]   (per token)
"""
from __future__ import annotations

import torch

from ..quant.uniform import quantize_asym, dequantize_asym
from .grouped import GroupedOnlineMethod, TileStorage


class KiviMethod(GroupedOnlineMethod):
    name = "kivi"
    online = True

    def __init__(self, bits_k: int = 2, bits_v: int = 2, group_size: int = 128,
                 meta_bits: int = 32, sink_tokens: int = 0, **kw):
        super().__init__(group_size=group_size, sink_tokens=sink_tokens, **kw)
        self.bits_k, self.bits_v = bits_k, bits_v
        self.meta_bits = meta_bits

    def encode_tiles(self, k, v, layer_idx):
        G = self.G
        M = k.shape[2] // G
        # K per-channel: [B,H,D,M*G] grouped over tokens
        kq, kscale, kzero = quantize_asym(
            k.transpose(2, 3), self.bits_k, dim=-1, group_size=G)
        # V per-token: one scale per token over all D channels
        vq, vscale, vzero = quantize_asym(
            v, self.bits_v, dim=-1, group_size=None)
        return [TileStorage(
            k_codes=kq, k_params={"scale": kscale, "zero": kzero, "m": M,
                                  "out_dtype": k.dtype},
            v_codes=vq, v_params={"scale": vscale, "zero": vzero, "m": M,
                                  "out_dtype": v.dtype},
            bits_k=self.bits_k, bits_v=self.bits_v)]

    def decode_tiles(self, sts, layer_idx):
        assert len(sts) == 1
        st = sts[0]
        kd = dequantize_asym(st.k_codes, st.k_params["scale"],
                             st.k_params["zero"])          # [B,H,D,M,G]
        B, Hh, D, M, G = kd.shape
        kd = kd.permute(0, 1, 3, 4, 2).reshape(B, Hh, M * G, D)
        vd = dequantize_asym(st.v_codes, st.v_params["scale"],
                             st.v_params["zero"])           # [B,H,M*G,D]
        return kd.to(st.k_params["out_dtype"]), vd.to(st.v_params["out_dtype"])

    def bits_report(self):
        G = self.G
        bpe_k = self.bits_k + 2 * 16 / G        # fp16 scale+zero per group
        bpe_v = self.bits_v + 2 * 16 / G
        return {"bpe_k": bpe_k, "bpe_v": bpe_v,
                "fp16_eq_tokens_per_head": None}
