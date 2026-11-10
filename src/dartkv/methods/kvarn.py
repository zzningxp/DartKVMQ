"""KVarN: Hadamard (channel) + dual-axis variance normalization + 2-bit RTN.
Three-region layout: FP16 sink / quantized G-token groups / FP16 open tail.
Decode reconstructs original space: (dequant * scales) then @H — exact math.

Chunk storage (M groups of G tokens per chunk, tiles per (b,h,group)):
  K: rotated tiles [D, G] (channels=rows, per official reference);
     per-channel RTN with the VarN channel scale folded into scale/zero
     -> codes [B,H,M,D,G], scale/zero [B,H,M,D,1], token scale [B,H,M,1,G]
  V: tiles [G, D] (tokens=rows); per-token RTN with the VarN token scale
     folded -> codes [B,H,M,G,D], scale/zero [B,H,M,G,1], chan [B,H,M,1,D]
"""
from __future__ import annotations

import torch

from ..quant.uniform import quantize_asym, dequantize_asym
from ..quant.hadamard import hadamard_matrix
from ..quant.varn import variance_normalize
from .grouped import GroupedOnlineMethod, TileStorage


class KvarnMethod(GroupedOnlineMethod):
    name = "kvarn"
    online = True

    def __init__(self, bits_k: int = 2, bits_v: int = 2, group_size: int = 128,
                 sink_tokens: int = 128, varn_iters: int = 8,
                 rtn_quantile: float = 0.0, **kw):
        super().__init__(group_size=group_size, sink_tokens=sink_tokens, **kw)
        self.bits_k, self.bits_v = bits_k, bits_v
        self.varn_iters = varn_iters
        self.rtn_quantile = rtn_quantile or None

    def encode_tiles(self, k, v, layer_idx):
        B, Hh, T, D = k.shape                    # T = M*G
        G = self.G
        M = T // G
        out_dtype = k.dtype
        # hadamard_matrix is ALREADY orthonormal (H @ H = I, entries +-1/sqrt(D))
        # — same convention as the official runtime's H/sqrt(d)
        Hm = hadamard_matrix(D, device=k.device)
        kr = k.float() @ Hm                      # [B,H,T,D]
        vr = v.float() @ Hm

        # -- K: tiles [N, D, G] (channels as rows, tokens as cols) --
        kt = (kr.reshape(B, Hh, M, G, D).permute(1, 0, 2, 4, 3)
              .reshape(Hh * B * M, D, G))
        kr_n, kSc_tok, kSr_chan = variance_normalize(
            kt, iterations=self.varn_iters)      # (N,D,G) (N,1,G) (N,D,1)
        kr_n = (kr_n.reshape(Hh, B, M, D, G).permute(1, 0, 2, 4, 3)
                .reshape(B, Hh, M, G, D))
        kSr_chan = (kSr_chan.reshape(Hh, B, M, D, 1).permute(1, 0, 2, 3, 4))
        kSc_tok = (kSc_tok.reshape(Hh, B, M, 1, G).permute(1, 0, 2, 3, 4))

        # per-channel RTN over the G tokens of each (group, channel)
        kq, ks, kz = quantize_asym(kr_n.transpose(3, 4), self.bits_k,
                                   dim=-1, group_size=None,
                                   range_quantile=self.rtn_quantile)
        ks = ks * kSr_chan                        # fold channel scale
        kz = kz * kSr_chan
        k_params = {"scale": ks, "zero": kz, "tok_scale": kSc_tok,
                    "hadamard": Hm, "out_dtype": out_dtype, "m": M}

        # -- V: tiles [N, G, D] (tokens as rows) --
        vt = (vr.reshape(B, Hh, M, G, D).permute(1, 0, 2, 3, 4)
              .reshape(Hh * B * M, G, D))
        vr_n, vSc_chan, vSr_tok = variance_normalize(
            vt, iterations=self.varn_iters)      # (N,G,D) (N,1,D) (N,G,1)
        vr_n = vr_n.reshape(Hh, B, M, G, D).permute(1, 0, 2, 3, 4)
        vSc_chan = vSc_chan.reshape(Hh, B, M, 1, D).permute(1, 0, 2, 3, 4)
        vSr_tok = vSr_tok.reshape(Hh, B, M, G, 1).permute(1, 0, 2, 3, 4)

        vq, vs, vz = quantize_asym(vr_n, self.bits_v, dim=-1, group_size=None,
                                   range_quantile=self.rtn_quantile)
        vs = vs * vSr_tok                         # fold token scale
        vz = vz * vSr_tok
        v_params = {"scale": vs, "zero": vz, "chan_scale": vSc_chan,
                    "hadamard": Hm, "out_dtype": out_dtype, "m": M}

        return [TileStorage(k_codes=kq, k_params=k_params, v_codes=vq,
                            v_params=v_params, bits_k=self.bits_k,
                            bits_v=self.bits_v)]

    def decode_tiles(self, sts, layer_idx):
        assert len(sts) == 1
        st = sts[0]
        kp, vp = st.k_params, st.v_params
        kd = dequantize_asym(st.k_codes, kp["scale"], kp["zero"])  # [B,H,M,D,G]
        kd = kd * kp["tok_scale"]
        B, Hh, M, D, G = kd.shape
        kd = kd.transpose(3, 4) @ kp["hadamard"]   # [B,H,M,G,D]
        kd = kd.reshape(B, Hh, M * G, D)
        vd = dequantize_asym(st.v_codes, vp["scale"], vp["zero"])   # [B,H,M,G,D]
        vd = vd * vp["chan_scale"]
        vd = (vd @ vp["hadamard"]).reshape(B, Hh, M * G, D)
        od = kp["out_dtype"]
        return kd.to(od), vd.to(od)

    def bits_report(self):
        # paper App. D accounting: 2 bits + two FP8 scales (one absorbs the
        # VarN scale into the RTN scale) + one FP16 zero-point, per group of
        # G elements -> 2 + 32/G (2.25 at G=128; paper headline "2.3")
        bpe = 2 + (8 * 2 + 16) / self.G
        return {"bpe_k": bpe, "bpe_v": bpe, "fp16_eq_tokens_per_head": None}
