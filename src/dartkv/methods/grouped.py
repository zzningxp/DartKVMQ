"""Grouped online-quantization layout shared by KIVI / KVarN / RAQ-MHA.

Cache per (layer) = optional FP16 sink [0:S] + sealed quantized groups of G
tokens + FP16 open tail (< G, sealed when full; sawtooth like KVarN/RAQ).

Subclasses provide `process_tile(k_tile, v_tile, layer)` which must return a
**codec**: a callable mapping the raw tile -> (stored codes+params) and back.
Here we factor it as:

  encode_tile(tile_kv) -> TileStorage   (transform + quantize)
  decode_tile(st)      -> tile in ORIGINAL space (dequant + inverse transform)

Tiles are [H_kv, G, D] per layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import CompressionMethod, RunContext


@dataclass
class TileStorage:
    """Quantized tile + whatever readout params the method needs."""
    k_codes: torch.Tensor     # (H, G, ?) packed codes or fp16 if bits==16
    k_params: dict            # scales/zeropoints/extra (e.g. VarN row scales)
    v_codes: torch.Tensor
    v_params: dict
    bits_k: int
    bits_v: int


class GroupedOnlineMethod(CompressionMethod):
    """Common three-region layout. Subclass implements encode/decode_tile."""

    def __init__(self, group_size: int = 128, sink_tokens: int = 0, **kw):
        super().__init__(**kw)
        self.G = group_size
        self.S = sink_tokens
        # per layer: dict with sink/tail tensors and list of TileStorage
        self.state: list[dict] = []

    def init_run(self, ctx: RunContext):
        super().init_run(ctx)
        self.state = [
            {"sink_k": None, "sink_v": None, "tail_k": None, "tail_v": None,
             "groups": [], "n_group_tokens": 0,
             "body": None, "body_groups": -1, "body_sink": -1}
            for _ in range(ctx.num_layers)
        ]

    # -- tile codec (override) ------------------------------------------
    def encode_tile(self, k_tile: torch.Tensor, v_tile: torch.Tensor,
                    layer_idx: int) -> TileStorage:
        """Encode a single G-token tile (kept for tests / small paths)."""
        return self.encode_tiles(k_tile, v_tile, layer_idx)[0]

    def decode_tile(self, st: TileStorage, layer_idx: int):
        """Decode one storage chunk; returns ([B,H,n,·] K, V)."""
        return self.decode_tiles([st], layer_idx)

    def encode_tiles(self, k, v, layer_idx: int) -> list[TileStorage]:
        """Encode a sealed chunk of M*G tokens at once (batched hot path).
        Default: fall back to per-tile encode_tile via subclass."""
        M = k.shape[2] // self.G
        return [self.encode_tile(k[:, :, g * self.G:(g + 1) * self.G],
                                 v[:, :, g * self.G:(g + 1) * self.G],
                                 layer_idx)
                for g in range(M)]

    def decode_tiles(self, sts: list[TileStorage], layer_idx: int):
        outs = [self.decode_tile(st, layer_idx) for st in sts]
        ks = torch.cat([o[0] for o in outs], dim=2)
        vs = torch.cat([o[1] for o in outs], dim=2)
        return ks, vs

    # -- layout logic -----------------------------------------------------
    def ingest(self, layer_idx: int, k, v, is_first_pass: bool):
        st = self.state[layer_idx]
        # sink: fp16 prefix, topped up across calls until S tokens collected
        if self.S > 0:
            have = (st["sink_k"].shape[2]
                    if isinstance(st["sink_k"], torch.Tensor) else 0)
            if have < self.S:
                take = min(self.S - have, k.shape[2])
                if have:
                    st["sink_k"] = torch.cat([st["sink_k"], k[:, :, :take]], dim=2)
                    st["sink_v"] = torch.cat([st["sink_v"], v[:, :, :take]], dim=2)
                else:
                    st["sink_k"] = k[:, :, :take].clone()
                    st["sink_v"] = v[:, :, :take].clone()
                k, v = k[:, :, take:], v[:, :, take:]
        # append to tail
        st["tail_k"] = k if st["tail_k"] is None else torch.cat([st["tail_k"], k], dim=2)
        st["tail_v"] = v if st["tail_v"] is None else torch.cat([st["tail_v"], v], dim=2)
        # seal full groups (batched encode of the whole sealed chunk)
        n = st["tail_k"].shape[2]
        n_groups = n // self.G
        if n_groups:
            sealed_k = st["tail_k"][:, :, : n_groups * self.G]
            sealed_v = st["tail_v"][:, :, : n_groups * self.G]
            st["tail_k"] = st["tail_k"][:, :, n_groups * self.G:]
            st["tail_v"] = st["tail_v"][:, :, n_groups * self.G:]
            st["groups"].extend(self.encode_tiles(sealed_k, sealed_v,
                                                  layer_idx))
            st["n_group_tokens"] += n_groups * self.G

    def get_cache(self, layer_idx: int):
        st = self.state[layer_idx]
        # Decoded (sink + sealed groups) body is cached and only rebuilt when
        # a new group seals or the sink grows; each decode step then just
        # concatenates the fp16 tail. (Accuracy-repro speed cache: storage
        # accounting stays on codes; the dense body exists only to feed
        # SDPA, which needs dense K/V every step anyway.)
        sink_len = (st["sink_k"].shape[2]
                    if st["sink_k"] is not None else 0)
        body = st["body"]
        if (body is None or st["body_groups"] != len(st["groups"])
                or st["body_sink"] != sink_len):
            ks, vs = [], []
            if self.S > 0 and st["sink_k"] is not None:
                ks.append(st["sink_k"]); vs.append(st["sink_v"])
            for g in st["groups"]:
                kd, vd = self.decode_tiles([g], layer_idx)
                ks.append(kd); vs.append(vd)
            body = (torch.cat(ks, dim=2), torch.cat(vs, dim=2)) if ks else None
            st["body"], st["body_groups"], st["body_sink"] = (
                body, len(st["groups"]), sink_len)
        if body is None:
            return st["tail_k"], st["tail_v"]
        if st["tail_k"] is not None and st["tail_k"].numel():
            return (torch.cat([body[0], st["tail_k"]], dim=2),
                    torch.cat([body[1], st["tail_v"]], dim=2))
        return body

    def total_tokens(self, layer_idx: int = 0) -> int:
        st = self.state[layer_idx]
        n = 0 if st["sink_k"] is None else st["sink_k"].shape[2]
        n += st["n_group_tokens"]
        n += 0 if st["tail_k"] is None else st["tail_k"].shape[2]
        return n

    def bits_report(self):
        # default accounting: bits + fp16 scale&zero per group per row/col
        raise NotImplementedError
