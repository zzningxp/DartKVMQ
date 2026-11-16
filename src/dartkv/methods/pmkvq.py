"""PM-KVQ port (arXiv 2505.18610, ICLR 2026): memory-pressure progressive
re-quantization of the KV cache. Re-implementation of the official
ProgressiveQuantizedCache semantics inside our method framework.

Mechanism (faithful to external/PM-KVQ progressive_quantized_cache.py):
  layout   sink (n_sink tokens, 16b) | middle (progressively degraded) |
           sliding window (n_window tokens, 16b)
  quant    per-token asymmetric along head_dim (zero=-min, recon=s*q+min,
           params frozen at first capture, never re-fit)
  trigger  per-layer bit budget exceeded after an ingest -> whole-layer
           middle bits halve: 16 -> 8 -> 4 -> 2 (min_bits floor)
  shrinking ON THE CODES, fp16 never reconstructed:
           q' = round(q * (2^b-1) / (2^B-1)), scale *= 2^b + 1  (B = 2b)
  window   tokens leaving the window are captured at 16b and ladder-halved
           down to the layer's current middle bits

Differences vs official (documented, both benign for exams):
  * budget: uniform per-layer budget on the RDKV axis (fp16-equivalent
    tokens n per (layer,head); 32*n*D bits/layer*heads) instead of their
    sensitivity-allocated per-layer MB list — puts pmkvq128 on the same
    axis as rdkv128.
  * accounting counts honest bits (codes + bf16 scale/zero + sink/window).
  * at min_bits the cache stops shrinking (official raises MemoryError).
  * requant_mode='direct' (research variant, E1): keep a bf16 source copy
    of the middle; on trigger re-quantize from source straight to the
    target bits in ONE step (fresh min/max) — the cascade-vs-direct
    comparison nobody has measured on KV cache.
"""
from __future__ import annotations

import torch

from ..quant.uniform import quantize_asym
from .base import CompressionMethod, RunContext


def _halve_codes(q: torch.Tensor, bits: int) -> tuple[torch.Tensor, float]:
    """q' = round(q*(2^b-1)/(2^B-1)), B=2b. Returns (codes, scale_factor)."""
    assert bits in (16, 8, 4, 2) and bits > 1
    b = bits // 2
    qmax_old = 2 ** bits - 1
    qmax_new = 2 ** b - 1
    q = torch.round(q.float() * qmax_new / qmax_old).clamp_(0, qmax_new)
    return q, float(qmax_old) / qmax_new          # = 2^b + 1


class PmKvqMethod(CompressionMethod):
    name = "pmkvq"
    online = True

    def __init__(self, budget_tokens: int = 128, n_sink: int = 1,
                 n_window: int = 128, init_bits: int = 16, min_bits: int = 2,
                 requant_mode: str = "cascade", **kw):
        super().__init__(**kw)
        self.n = budget_tokens
        self.n_sink = n_sink
        self.n_window = n_window
        self.init_bits = init_bits
        self.min_bits = min_bits
        assert requant_mode in ("cascade", "direct")
        self.requant_mode = requant_mode
        # per-layer state
        self._sink: dict[int, tuple] = {}        # raw (k, v) [B,H,S,D]
        self._win: dict[int, tuple] = {}         # raw (k, v) [B,H,W,D]
        self._mid: dict[int, dict] = {}          # codes/scale/zero (+src)
        self._bits: dict[int, int] = {}          # current middle bits
        self._seen: dict[int, int] = {}          # tokens seen per layer
        self.events: list[dict] = []             # halving event log (E2)

    # -- helpers ----------------------------------------------------------
    def _budget_bits(self) -> int:
        """Per-layer bit budget: 32*n*D*H (RDKV axis)."""
        return 32 * self.n * self.ctx.head_dim * self.ctx.num_kv_heads

    def _layer_bits(self, l: int) -> int:
        """True stored bits of layer l over ALL kv heads."""
        H = self.ctx.num_kv_heads
        D = self.ctx.head_dim
        total = 0
        if l in self._sink:
            total += H * 2 * self._sink[l][0].shape[2] * D * 16
        if l in self._win:
            total += H * 2 * self._win[l][0].shape[2] * D * 16
        if l in self._mid:
            m = self._mid[l]
            t = m["kq"].shape[2]
            total += H * 2 * (t * D * self._bits[l] + t * 2 * 16)
        return total

    def _capture(self, k, v, bits: int) -> dict:
        """Per-token asym quant along head_dim at `bits` (params frozen)."""
        kq, ks, kz = quantize_asym(k, bits, dim=-1, group_size=None)
        vq, vs, vz = quantize_asym(v, bits, dim=-1, group_size=None)
        return {"kq": kq.to(torch.int32), "ks": ks, "kz": kz,
                "vq": vq.to(torch.int32), "vs": vs, "vz": vz}

    def _to_middle(self, l: int, k, v):
        """Capture at 16b (lossless-ish), ladder-halve to current bits."""
        st = self._capture(k, v, 16)
        b = 16
        while b > max(self._bits[l], 1):
            st["kq"], f = _halve_codes(st["kq"], b)
            st["vq"], _ = _halve_codes(st["vq"], b)
            st["ks"] = st["ks"] * f
            st["vs"] = st["vs"] * f
            b //= 2
        m = self._mid.get(l)
        if m is None:
            self._mid[l] = st
            if self.requant_mode == "direct":
                st["src_k"], st["src_v"] = k, v
            return
        for kk in ("kq", "vq"):
            m[kk] = torch.cat([m[kk], st[kk]], dim=2)
        for kk in ("ks", "kz", "vs", "vz"):
            m[kk] = torch.cat([m[kk], st[kk]], dim=2)
        if self.requant_mode == "direct":
            m["src_k"] = torch.cat([m["src_k"], k], dim=2)
            m["src_v"] = torch.cat([m["src_v"], v], dim=2)

    def _shrink(self, l: int):
        """One halving step (cascade) or direct re-quant from source."""
        b = self._bits[l]
        m = self._mid[l]
        if self.requant_mode == "direct" and "src_k" in m:
            nb = b // 2
            st = self._capture(m["src_k"], m["src_v"], nb)
            m.update(st)                              # fresh params
        else:
            m["kq"], f = _halve_codes(m["kq"], b)
            m["vq"], _ = _halve_codes(m["vq"], b)
            m["ks"] = m["ks"] * f
            m["vs"] = m["vs"] * f
        self._bits[l] = b // 2
        self.events.append({"at_token": self._seen[l], "layer": l,
                            "from": b, "to": b // 2})

    # -- CompressionMethod API --------------------------------------------
    def init_run(self, ctx: RunContext):
        super().init_run(ctx)
        for l in range(ctx.num_layers):
            self._bits[l] = self.init_bits

    def ingest(self, layer_idx: int, k, v, is_first_pass: bool):
        self._seen[layer_idx] = self._seen.get(layer_idx, 0) + k.shape[2]
        # 1. fill sink from the front
        s = self._sink.get(layer_idx)
        if s is None:
            n = min(self.n_sink, k.shape[2])
            self._sink[layer_idx] = (k[:, :, :n], v[:, :, :n])
            k, v = k[:, :, n:], v[:, :, n:]
        # 2. window overflow -> middle; window keeps last n_window tokens
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
        # 3. memory-pressure trigger: halve whole-layer middle until fits
        if layer_idx in self._mid:
            guard = 0
            while (self._layer_bits(layer_idx) > self._budget_bits()
                   and self._bits[layer_idx] > self.min_bits):
                self._shrink(layer_idx)
                guard += 1
                assert guard < 8, "shrink loop runaway"

    def get_cache(self, layer_idx: int):
        parts_k, parts_v = [], []
        if layer_idx in self._sink:
            parts_k.append(self._sink[layer_idx][0])
            parts_v.append(self._sink[layer_idx][1])
        if layer_idx in self._mid:
            m = self._mid[layer_idx]
            kd = m["ks"] * m["kq"].float() + m["kz"]
            vd = m["vs"] * m["vq"].float() + m["vz"]
            parts_k.append(kd.to(self._win[layer_idx][0].dtype))
            parts_v.append(vd.to(self._win[layer_idx][0].dtype))
        if layer_idx in self._win:
            parts_k.append(self._win[layer_idx][0])
            parts_v.append(self._win[layer_idx][1])
        return torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)

    def stats(self) -> dict:
        from collections import Counter
        bits_hist = Counter(self._bits.values())
        # full event log (capped) for the gradual-vs-sudden (E2) analysis
        ev = self.events[:256]
        return {"method": self.name,
                "middle_bits_per_layer": dict(bits_hist),
                "halving_events": len(self.events),
                "events": ev}

    def bits_report(self) -> dict:
        """Current (end-of-run) average bpw over the whole cache (K+V,
        all heads), including sink/window and params."""
        tot_bits = sum(self._layer_bits(l) for l in self._bits)
        tot_elems = 0
        for l in self._bits:
            t = self._seen.get(l, 0)
            tot_elems += t * self.ctx.head_dim * 2 * self.ctx.num_kv_heads
        bpe = tot_bits / tot_elems if tot_elems else None
        return {"bpe_k": bpe, "bpe_v": None,
                "fp16_eq_tokens_per_head": self.n}
