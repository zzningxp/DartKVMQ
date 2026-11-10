"""SKVQ: sliding-window KV cache quantization (arXiv 2405.06219 official port).

Layout replicates the official SlidingKVCacheManager (fake quant,
full_prefill=True): FP16 attention sink [0:S] | clipped dynamic
per-token channel-group quantized body | FP16 sliding window (last W).

Sliding-window semantics are exact (NOT the G-token sawtooth of
KIVI/KVarN): prefill quantizes [S : T-W] in one shot; during decode the
window holds exactly W tokens — each new token evicts the oldest window
token into the quantized body one at a time. Eager eviction per ingest
call is equivalent to the official prefill/decode split because tokens
quantize independently.

Quantization groups are CHANNEL groups (gsize channels of the flat H*D
axis per token), optionally reordered by offline calibration into
variable-size KMeans clusters (reorder_file, official format:
{"reorder_indices": [(k_idx, v_idx) per layer],
 "cluster_st_inds": [(k_bounds, v_bounds) per layer]}).

Codec math lives in quant/skvq.py (official KV_process.quant_pytorch
replicated in the input dtype: clip, fp8 e4m3 scales, 1.5-bit V).

Not ported from the official engine: KIVI_mode (native bench method
`kivi` covers it) and pre_rope (our harness caches post-RoPE keys ==
the official pre_rope=False setting).
"""
from __future__ import annotations

import torch

from ..quant.skvq import group_bounds, quant_dequant_channels
from .grouped import GroupedOnlineMethod, TileStorage


class SkvqMethod(GroupedOnlineMethod):
    name = "skvq"
    online = True

    def __init__(self, bits_k: float = 2, bits_v: float = 2,
                 group_size: int = 128, window: int = 128,
                 sink_tokens: int = 5, clip: float = 0.92,
                 fp8_scales: bool = False, reorder_file: str | None = None,
                 **kw):
        super().__init__(group_size=group_size, sink_tokens=sink_tokens, **kw)
        self.bits_k, self.bits_v = float(bits_k), float(bits_v)
        self.window = window
        self.clip = clip
        self.fp8_scales = fp8_scales
        self.reorder_file = reorder_file
        self._rod: list | None = None   # per-layer reorder metadata

    # -- lifecycle --------------------------------------------------------
    def init_run(self, ctx):
        super().init_run(ctx)
        for st in self.state:  # grouped.py slots + sliding window slots
            st.update(win_k=None, win_v=None, body_k=None, body_v=None,
                      n_quant=0)
        self.hidden = ctx.num_kv_heads * ctx.head_dim
        self._rod = self._load_reorder(ctx)

    def _load_reorder(self, ctx):
        if not self.reorder_file:
            if self.G and self.hidden % self.G:
                raise ValueError(
                    f"hidden {self.hidden} not divisible by gsize {self.G}")
            return None
        meta = torch.load(self.reorder_file, map_location="cpu")
        rod_idx, group_st = meta["reorder_indices"], meta["cluster_st_inds"]
        assert len(rod_idx) == ctx.num_layers, (
            f"reorder_file has {len(rod_idx)} layers, run has "
            f"{ctx.num_layers}")
        out = []
        for (ki, vi), (kb, vb) in zip(rod_idx, group_st):
            assert ki.shape[0] == self.hidden, (ki.shape, self.hidden)
            out.append({
                "k": {"idx": ki.long(), "bounds": kb.long()},
                "v": {"idx": vi.long(), "bounds": vb.long()},
            })
        return out

    def _meta(self, ttype: str, layer_idx: int):
        if self._rod is None:
            return None, group_bounds(self.hidden, self.G)
        m = self._rod[layer_idx][ttype]
        return m["idx"], m["bounds"]

    # -- codec ------------------------------------------------------------
    def encode_tiles(self, k, v, layer_idx: int) -> list[TileStorage]:
        """Fake-quantize a chunk of body tokens (per token, channel groups).

        In fake-quant mode codes == dequantized values (official returns
        the dequantized tensor directly), so decode is the identity.
        """
        B, H, T, D = k.shape
        flat_k = k.transpose(1, 2).reshape(B * T, H * D)
        flat_v = v.transpose(1, 2).reshape(B * T, H * D)
        kidx, kbounds = self._meta("k", layer_idx)
        vidx, vbounds = self._meta("v", layer_idx)
        kq = quant_dequant_channels(flat_k, self.bits_k, kbounds, self.clip,
                                    kidx, self.fp8_scales)
        vq = quant_dequant_channels(flat_v, self.bits_v, vbounds, self.clip,
                                    vidx, self.fp8_scales)
        kq = kq.reshape(B, T, H, D).transpose(1, 2)
        vq = vq.reshape(B, T, H, D).transpose(1, 2)
        return [TileStorage(k_codes=kq, k_params={"out_dtype": k.dtype},
                            v_codes=vq, v_params={"out_dtype": v.dtype},
                            bits_k=self.bits_k, bits_v=self.bits_v)]

    def decode_tiles(self, sts, layer_idx):
        assert len(sts) == 1
        st = sts[0]
        return (st.k_codes.to(st.k_params["out_dtype"]),
                st.v_codes.to(st.v_params["out_dtype"]))

    # -- sliding-window layout --------------------------------------------
    def ingest(self, layer_idx: int, k, v, is_first_pass: bool):
        st = self.state[layer_idx]
        k, v = self._absorb_sink(st, k, v)
        # append to the FP16 window, then evict the overflow into the body
        st["win_k"] = k if st["win_k"] is None else torch.cat([st["win_k"], k], dim=2)
        st["win_v"] = v if st["win_v"] is None else torch.cat([st["win_v"], v], dim=2)
        n = st["win_k"].shape[2]
        if n > self.window:
            m = n - self.window
            kd, vd = self.decode_tiles(
                self.encode_tiles(st["win_k"][:, :, :m].contiguous(),
                                  st["win_v"][:, :, :m].contiguous(),
                                  layer_idx), layer_idx)
            st["win_k"] = st["win_k"][:, :, m:]
            st["win_v"] = st["win_v"][:, :, m:]
            st["body_k"] = (kd if st["body_k"] is None
                            else torch.cat([st["body_k"], kd], dim=2))
            st["body_v"] = (vd if st["body_v"] is None
                            else torch.cat([st["body_v"], vd], dim=2))
            st["n_quant"] += m

    def get_cache(self, layer_idx: int):
        st = self.state[layer_idx]
        parts_k, parts_v = [], []
        if st["sink_k"] is not None:
            parts_k.append(st["sink_k"]); parts_v.append(st["sink_v"])
        if st["body_k"] is not None:
            parts_k.append(st["body_k"]); parts_v.append(st["body_v"])
        if st["win_k"] is not None and st["win_k"].shape[2]:
            parts_k.append(st["win_k"]); parts_v.append(st["win_v"])
        return (torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2))

    def total_tokens(self, layer_idx: int = 0) -> int:
        st = self.state[layer_idx]
        n = 0 if st["sink_k"] is None else st["sink_k"].shape[2]
        n += st["n_quant"]
        n += 0 if st["win_k"] is None else st["win_k"].shape[2]
        return n

    def bits_report(self):
        # scale+zero per channel group; fp16 pairs, or fp8 e4m3 with
        # -fp8 (official longbench configs store scales in fp8)
        scale_bits = 8 if self.fp8_scales else 16
        meta = 2 * scale_bits / self.G
        return {"bpe_k": self.bits_k + meta, "bpe_v": self.bits_v + meta,
                # persistent FP16 regions: S sink tokens + W window tokens
                "fp16_eq_tokens_per_head": float(self.S + self.window)}
