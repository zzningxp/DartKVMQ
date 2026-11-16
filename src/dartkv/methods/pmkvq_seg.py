"""PmKvqSegMethod: segmented research variant of the PM-KVQ port.

Two knobs the official design does not have (F3/F1 follow-ups):

  shrink='layer'      official: on trigger every middle segment halves
                      (whole-layer sudden degradation)
  shrink='oldest'     gradual: halve OLDEST segments first, `shrink_frac`
                      of middle tokens per trigger event (lifecycle: young
                      tokens keep bits, old tokens pay)
  shrink='newest'     gradual, opposite end: halve NEWEST segments first

  k_axis='token'      official: K per-token (group = head_dim)
  k_axis='channel'    KIVI-style: K per-channel over G=128-token sealed
                      groups (raw staging buffer until 128 tokens, open
                      tail stays fp16 — KIVI open-group semantics); V stays
                      per-token. Tests F1's axis attribution on the same
                      harness: if this rescues the 2-bit Llama collapse,
                      the axis (not the ladder) is the cause.

Storage: middle = ordered segments; token-axis segments merge when they
share bits; channel-axis segments are fixed 128-token groups (never merged
— each carries its own channel params). Dequant needs no bits knowledge
(per-token/group scale already carries the halving factors); bits only
drive accounting and the next halving choice.
"""
from __future__ import annotations

import torch

from ..quant.uniform import quantize_asym
from .base import CompressionMethod, RunContext
from .pmkvq import _halve_codes


class _Seg:
    __slots__ = ("kq", "vq", "ks", "kz", "vs", "vz", "bits", "t",
                 "abs_start", "abs_end", "err")

    def __init__(self, kq, vq, ks, kz, vs, vz, bits):
        self.kq, self.vq = kq, vq
        self.ks, self.kz, self.vs, self.vz = ks, kz, vs, vz
        self.bits, self.t = bits, int(kq.shape[2])
        self.err = None          # per-token quant-error proxy (damage sched)


class PmKvqSegMethod(CompressionMethod):
    name = "pmkvq_seg"
    online = True

    def __init__(self, budget_tokens: int = 128, n_sink: int = 1,
                 n_window: int = 128, init_bits: int = 16, min_bits: int = 2,
                 shrink: str = "layer", shrink_frac: float = 0.25,
                 k_axis: str = "token", seg_g: int = 128,
                 beta: float = 1.0, evict: bool = False, **kw):
        super().__init__(**kw)
        self.G = seg_g   # channel-axis seal size / gradual segment cap
        self.n = budget_tokens
        self.n_sink, self.n_window = n_sink, n_window
        self.init_bits, self.min_bits = init_bits, min_bits
        assert shrink in ("layer", "oldest", "newest", "read")
        self.shrink, self.shrink_frac = shrink, shrink_frac
        assert k_axis in ("token", "channel")
        self.k_axis = k_axis
        # beta: trigger at beta*budget (proactive/anticipatory scheduling);
        # evict: allow the ladder's bottom rung (0-bit = drop segment),
        # unifying RDKV eviction with the quant ladder under one allocator
        assert 0.5 <= beta <= 1.0
        self.beta = float(beta)
        self.evict = bool(evict)
        self._sink: dict[int, tuple] = {}
        self._win: dict[int, tuple] = {}
        self._stage: dict[int, tuple] = {}     # channel-axis raw staging
        self._segs: dict[int, list[_Seg]] = {}
        self._seen: dict[int, int] = {}
        self.events: list[dict] = []

    # -- helpers ----------------------------------------------------------
    def _budget_bits(self) -> int:
        return 32 * self.n * self.ctx.head_dim * self.ctx.num_kv_heads

    def _seg_bits(self, s: _Seg) -> int:
        """Per-head stored bits of one segment (K params axis-dependent)."""
        D = self.ctx.head_dim
        kp = s.t * 2 * 16 if self.k_axis == "token" else D * 2 * 16
        return 2 * s.t * D * s.bits + kp + s.t * 2 * 16

    def _layer_bits(self, l: int) -> int:
        H = self.ctx.num_kv_heads
        D = self.ctx.head_dim
        total = 0
        if l in self._sink:
            total += H * 2 * self._sink[l][0].shape[2] * D * 16
        if l in self._win:
            total += H * 2 * self._win[l][0].shape[2] * D * 16
        if l in self._stage:
            total += H * 2 * self._stage[l][0].shape[2] * D * 16
        for s in self._segs.get(l, []):
            total += H * self._seg_bits(s)
        return total

    def _entry_bits(self, l: int) -> int:
        segs = self._segs.get(l)
        if not segs:
            return self.init_bits
        if self.shrink == "layer":
            return min(s.bits for s in segs)      # official: match layer bits
        return max(s.bits for s in segs)          # lifecycle: young stay high

    def _ladder(self, kq, vq, ks, vs, bits: int, tgt: int):
        b = bits
        while b > tgt:
            kq, f = _halve_codes(kq, b)
            vq, _ = _halve_codes(vq, b)
            ks, vs = ks * f, vs * f
            b //= 2
        return kq, vq, ks, vs, b

    def _capture_token(self, k, v, bits: int) -> _Seg:
        kq, ks, kz = quantize_asym(k, bits, dim=-1, group_size=None)
        vq, vs, vz = quantize_asym(v, bits, dim=-1, group_size=None)
        return _Seg(kq.to(torch.int32), vq.to(torch.int32), ks, kz, vs, vz,
                    bits)

    def _capture_channel(self, k, v, bits: int) -> _Seg:
        kq, ks, kz = quantize_asym(k, bits, dim=2, group_size=None)
        vq, vs, vz = quantize_asym(v, bits, dim=-1, group_size=None)
        return _Seg(kq.to(torch.int32), vq.to(torch.int32), ks, kz, vs, vz,
                    bits)

    def _append_seg(self, l: int, seg: _Seg):
        if self.k_axis == "channel":
            self._segs.setdefault(l, []).append(seg)   # fixed G groups
            return
        segs = self._segs.setdefault(l, [])
        last = segs[-1] if segs else None
        # gradual modes cap merge size so oldest/newest picking can select
        # a FRACTION of the middle (a single merged segment would force
        # whole-middle halving — found in overnight 2026-11-15 run)
        cap = 10**9 if self.shrink == "layer" else self.G
        if (last is not None and last.bits == seg.bits
                and last.t + seg.t <= cap):
            last.kq = torch.cat([last.kq, seg.kq], dim=2)
            last.vq = torch.cat([last.vq, seg.vq], dim=2)
            last.ks = torch.cat([last.ks, seg.ks], dim=2)
            last.kz = torch.cat([last.kz, seg.kz], dim=2)
            last.vs = torch.cat([last.vs, seg.vs], dim=2)
            last.vz = torch.cat([last.vz, seg.vz], dim=2)
            last.t += seg.t
        else:
            segs.append(seg)

    def _to_middle(self, l: int, k, v):
        tgt = max(self._entry_bits(l), self.min_bits)
        if self.k_axis == "channel":
            st = self._stage.get(l)
            k = k if st is None else torch.cat([st[0], k], dim=2)
            v = v if st is None else torch.cat([st[1], v], dim=2)
            while k.shape[2] >= self.G:                 # seal full groups
                seg = self._capture_channel(k[:, :, :self.G],
                                            v[:, :, :self.G], 16)
                seg.kq, seg.vq, seg.ks, seg.vs, seg.bits = self._ladder(
                    seg.kq, seg.vq, seg.ks, seg.vs, 16, tgt)
                self._append_seg(l, seg)
                k, v = k[:, :, self.G:], v[:, :, self.G:]
            self._stage[l] = (k, v)                     # open tail stays raw
        else:
            # gradual modes slice big chunks into G-sized segments so the
            # oldest/newest picker can take a fraction (single 1k-token
            # prefill segment would force whole-block halving)
            step = self.G if self.shrink != "layer" else k.shape[2]
            for i0 in range(0, k.shape[2], step):
                seg = self._capture_token(k[:, :, i0:i0 + step],
                                          v[:, :, i0:i0 + step], 16)
                seg.kq, seg.vq, seg.ks, seg.vs, seg.bits = self._ladder(
                    seg.kq, seg.vq, seg.ks, seg.vs, 16, tgt)
                self._append_seg(l, seg)

    def _halve_seg(self, s: _Seg):
        s.kq, f = _halve_codes(s.kq, s.bits)
        s.vq, _ = _halve_codes(s.vq, s.bits)
        s.ks, s.vs = s.ks * f, s.vs * f
        s.bits //= 2

    def _evict_order(self, segs):
        """Default eviction order: oldest first (newest for 'newest')."""
        order = list(range(len(segs)))
        return order[::-1] if self.shrink == "newest" else order

    def _evict_step(self, l: int, segs) -> bool:
        """Drop shrink_frac of middle tokens from the bottom of the ladder."""
        T = sum(s.t for s in segs)
        quota = max(1, int(self.shrink_frac * max(T, 1)))
        dropped, acc = [], 0
        for i in self._evict_order(segs):
            dropped.append(i)
            acc += segs[i].t
            if acc >= quota:
                break
        if not dropped:
            return False
        for i in dropped:
            s = segs[i]
            self.events.append({"at_token": self._seen[l], "layer": l,
                                "from": s.bits, "to": 0, "n_tok": s.t,
                                "mode": "evict"})
        for i in sorted(dropped, reverse=True):
            del segs[i]
        return True

    def _trigger(self, l: int):
        segs = self._segs.get(l, [])
        T = sum(s.t for s in segs)
        guard = 0
        while self._layer_bits(l) > self.beta * self._budget_bits():
            cand = [i for i, s in enumerate(segs) if s.bits > self.min_bits]
            if not cand:
                # ladder bottom: either evict (0-bit rung) or stop honestly
                if self.evict and segs and self._evict_step(l, segs):
                    continue
                break
            if self.shrink == "layer":
                picked = cand
            else:
                order = cand if self.shrink == "oldest" else cand[::-1]
                picked, acc = [], 0
                for i in order:
                    picked.append(i)
                    acc += segs[i].t
                    if acc >= self.shrink_frac * max(T, 1):
                        break
            for i in picked:
                pre = segs[i].bits
                self._halve_seg(segs[i])
                self.events.append({"at_token": self._seen[l], "layer": l,
                                    "from": pre, "to": segs[i].bits,
                                    "n_tok": segs[i].t, "mode": self.shrink})
            if self.k_axis == "token":               # re-merge same-bits runs
                cap = 10**9 if self.shrink == "layer" else self.G
                merged: list[_Seg] = []
                for s in segs:
                    last = merged[-1] if merged else None
                    if (last is not None and last.bits == s.bits
                            and last.t + s.t <= cap):
                        last.kq = torch.cat([last.kq, s.kq], dim=2)
                        last.vq = torch.cat([last.vq, s.vq], dim=2)
                        last.ks = torch.cat([last.ks, s.ks], dim=2)
                        last.kz = torch.cat([last.kz, s.kz], dim=2)
                        last.vs = torch.cat([last.vs, s.vs], dim=2)
                        last.vz = torch.cat([last.vz, s.vz], dim=2)
                        last.t += s.t
                    else:
                        merged.append(s)
                self._segs[l] = segs = merged
            guard += 1
            assert guard < 200, "seg shrink runaway"

    # -- CompressionMethod API --------------------------------------------
    def ingest(self, layer_idx: int, k, v, is_first_pass: bool):
        self._seen[layer_idx] = self._seen.get(layer_idx, 0) + k.shape[2]
        s = self._sink.get(layer_idx)
        if s is None:
            n = min(self.n_sink, k.shape[2])
            self._sink[layer_idx] = (k[:, :, :n], v[:, :, :n])
            k, v = k[:, :, n:], v[:, :, n:]
        if k.shape[2] > 0:
            w = self._win.get(layer_idx)
            kall = k if w is None else torch.cat([w[0], k], dim=2)
            vall = v if w is None else torch.cat([w[1], v], dim=2)
            t = kall.shape[2]
            if t > self.n_window:
                self._to_middle(layer_idx, kall[:, :, :t - self.n_window],
                                vall[:, :, :t - self.n_window])
                self._win[layer_idx] = (kall[:, :, t - self.n_window:],
                                        vall[:, :, t - self.n_window:])
            else:
                self._win[layer_idx] = (kall, vall)
        self._trigger(layer_idx)

    def get_cache(self, layer_idx: int):
        dt = (self._win[layer_idx][0].dtype if layer_idx in self._win
              else torch.bfloat16)
        parts_k, parts_v = [], []
        if layer_idx in self._sink:
            parts_k.append(self._sink[layer_idx][0])
            parts_v.append(self._sink[layer_idx][1])
        for s in self._segs.get(layer_idx, []):
            parts_k.append((s.ks * s.kq.float() + s.kz).to(dt))
            parts_v.append((s.vs * s.vq.float() + s.vz).to(dt))
        if layer_idx in self._stage:                  # newer than sealed
            parts_k.append(self._stage[layer_idx][0])
            parts_v.append(self._stage[layer_idx][1])
        if layer_idx in self._win:
            parts_k.append(self._win[layer_idx][0])
            parts_v.append(self._win[layer_idx][1])
        return torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)

    def stats(self) -> dict:
        from collections import Counter
        h = Counter()
        for segs in self._segs.values():
            for s in segs:
                h[s.bits] += s.t
        return {"method": self.name, "shrink": self.shrink,
                "k_axis": self.k_axis,
                "middle_tokens_by_bits": dict(h),
                "halving_events": len(self.events),
                "events": self.events[:2048]}

    def bits_report(self) -> dict:
        tot_bits = sum(self._layer_bits(l) for l in self._seen)
        tot_elems = sum(t * self.ctx.head_dim * 2 * self.ctx.num_kv_heads
                        for t in self._seen.values())
        return {"bpe_k": tot_bits / tot_elems if tot_elems else None,
                "bpe_v": None, "fp16_eq_tokens_per_head": self.n}
