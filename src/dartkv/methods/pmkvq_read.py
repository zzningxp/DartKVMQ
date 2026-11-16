"""ReadAwareSegMethod: read-aware degradation schedule (2026-11-15).

The night's failure controls showed age is the wrong scheduler signal:
oldest-first degraded the prefix BEFORE it was read (factcover 0.0),
newest-first degraded unread tail (0.236). The right signal is *has it
been read?* — tokens already attended/copied into the output can be
degraded safely (their content lives on in the fp16 generation window).

Mechanism: wrap F.scaled_dot_product_attention (same trick as the RDKV
probe) to accumulate per-token attention received, per layer, absolute
token indexing. On a memory trigger, halve the segments with the HIGHEST
mean cumulative read first (most-consumed first), `shrink_frac` of middle
tokens per event; unread segments keep their bits until read.

Design round 1 (2026-11-15 night) adds four scheduler axes on top:

  half_life=S     read counter decays with half-life S decode steps
                  (recency-weighted reads). Tests the importance-drift
                  question: if future reads follow recent reads, decayed
                  ordering should match or beat raw cumulative reads.
  damage          order by remaining-value x quant-error instead of raw
                  reads: value proxy rv = 1/(1+read), error proxy e =
                  max|rms-normalized| amplitude captured once at 16-bit.
                  Cheapest expected damage degrades first; protects
                  outlier-heavy tokens that read ordering would blur.
  beta            trigger at beta*budget (proactive): degradation happens
                  earlier in smaller steps instead of one forced burst.
  evict           unified ladder bottom rung: when every segment sits at
                  min_bits and budget still binds, drop shrink_frac of
                  segments (0-bit rung) in the same order — RDKV eviction
                  and PM-KVQ quantization under one allocator.

On sequential-listing exams read-order coincides with 'oldest-first
DELAYED to actual reading time' — precisely the fix for old500's mode.
"""
from __future__ import annotations

import torch

from .base import RunContext
from .pmkvq_seg import PmKvqSegMethod, _Seg


class ReadAwareSegMethod(PmKvqSegMethod):
    name = "pmkvq_read"
    online = True

    def __init__(self, budget_tokens: int = 128, n_sink: int = 1,
                 n_window: int = 128, init_bits: int = 16, min_bits: int = 2,
                 shrink_frac: float = 0.25, k_axis: str = "token",
                 seg_g: int = 128, half_life: float = 0.0,
                 damage: bool = False, invert: bool = False,
                 uniform_depth: bool = False, beta: float = 1.0,
                 evict: bool = False, **kw):
        super().__init__(budget_tokens=budget_tokens, n_sink=n_sink,
                         n_window=n_window, init_bits=init_bits,
                         min_bits=min_bits, shrink="read",
                         shrink_frac=shrink_frac, k_axis=k_axis, seg_g=seg_g,
                         beta=beta, evict=evict, **kw)
        self.half_life = float(half_life)
        self.damage = bool(damage)
        self.invert = bool(invert)
        # uniform_depth: spread before deepening. Within a bits-tier,
        # degrade most-used first; but a segment may not fall a second
        # rung until every segment has fallen one. Attacks the CoT loss:
        # concentrated deep damage on heavily-used segments loses to the
        # official layer schedule's spread-shallow damage there, while
        # usage info still decides order (2026-11-16, design round 2).
        self.uniform_depth = bool(uniform_depth)
        self._read: dict[int, torch.Tensor] = {}      # [T_seen] cumulative
        self._rd: dict[int, torch.Tensor] = {}        # decayed reads
        self._r_snap: dict[int, torch.Tensor] = {}    # r at last decay
        self._rd_step: dict[int, int] = {}
        self._err: dict[int, torch.Tensor] = {}       # per-token err by abs
        self._mid_end: dict[int, int] = {}            # abs end of middle
        self._sdpa_orig = None
        self._call_ctr = 0
        self._n_layer_calls = 0

    # -- SDPA read tracking (class-level global patch, never stacks) -------
    _ACTIVE = None    # current run's method (repointed per attach)
    _ORIG_SDPA = None

    def attach(self, model):
        cls = type(self)
        cls._ACTIVE = self
        if cls._ORIG_SDPA is not None:
            return                                  # patch already armed
        import torch.nn.functional as Fn
        cls._ORIG_SDPA = Fn.scaled_dot_product_attention

        def wrapped(q, k, *a, **kw):
            m = cls._ACTIVE
            if m is not None and q.ndim == 4:
                try:
                    m._record_reads(q, k)
                except Exception:
                    pass                            # never break generation
            return cls._ORIG_SDPA(q, k, *a, **kw)

        Fn.scaled_dot_product_attention = wrapped
        torch.nn.functional.scaled_dot_product_attention = wrapped

    def detach(self):
        """Deactivate recording (patch stays armed but inert)."""
        type(self)._ACTIVE = None

    @torch.no_grad()
    def _record_reads(self, q, k):
        """q [1,Hq,Tq,D], k [1,Hq(=expanded),Tk,D]; layer = call order."""
        if not self._read:
            return                             # ctx not ready yet
        L = self.ctx.num_layers
        layer = self._call_ctr % L
        self._call_ctr += 1
        Hq, Hkv = q.shape[1], k.shape[1]
        if Hkv != Hq:      # transformers 5.x SDPA passes un-expanded K (GQA)
            k = k.repeat_interleave(Hq // Hkv, dim=1)
        T_k = k.shape[2]
        r = self._read[layer]
        if r.shape[0] < T_k:                   # cache grew (new tokens)
            r = torch.cat([r, torch.zeros(T_k - r.shape[0], device=r.device)])
            self._read[layer] = r
        scale = q.shape[-1] ** -0.5
        att = torch.softmax((q * scale) @ k.transpose(-1, -2), dim=-1)
        # attention received per key token, summed over heads & queries
        inc = att.sum(dim=(0, 1))[0]           # [T_k]
        r[:T_k] += inc

    def _updated_reads(self, l: int):
        """Read counter to order by: raw cumulative, or decayed (half_life)."""
        r = self._read.get(l)
        if r is None or self.half_life <= 0:
            return r
        step = self._call_ctr // max(self.ctx.num_layers, 1)
        rd = self._rd.get(l)
        if rd is None:                         # first trigger: adopt as-is
            self._rd[l] = r.clone()
            self._r_snap[l] = r.clone()
            self._rd_step[l] = step
            return self._rd[l]
        dt = step - self._rd_step[l]
        if dt > 0:
            snap = self._r_snap[l]
            if snap.numel() < r.numel():       # pad snapshot to current len
                snap = torch.cat([snap, torch.zeros(
                    r.numel() - snap.numel(), device=snap.device)])
                self._r_snap[l] = snap
            gamma = 0.5 ** (1.0 / self.half_life)
            rd.mul_(gamma ** dt)
            if r.numel() > rd.numel():         # grow for new tokens
                rd = torch.cat([rd, torch.zeros(
                    r.numel() - rd.numel(), device=rd.device)])
                self._rd[l] = rd
            rd.add_(r - snap)
            self._r_snap[l] = r.clone()
            self._rd_step[l] = step
        return rd

    # -- quant-error proxy (damage scheduler) --------------------------------
    @staticmethod
    def _tok_rms(x) -> torch.Tensor:
        """Per-token absolute RMS over (heads, head_dim); x [1,H,t,D].

        For per-token asymmetric quant the step is (max-min)/2^b = O(rms)
        of the token row, so the ABSOLUTE quantization error of a token is
        proportional to its rms (a ratio proxy would be scale-invariant
        and blind to amplitude — found in the 2026-11-15 smoke test).
        V error enters the attention output additively; K error perturbs
        logits ∝ |k| — both scale with absolute rms.
        """
        return x.float().pow(2).mean(dim=(1, 3)).sqrt()[0]

    def _quant_err(self, k, v) -> torch.Tensor:
        return 0.5 * (self._tok_rms(k) + self._tok_rms(v))

    def _capture_token(self, k, v, bits: int) -> _Seg:
        seg = super()._capture_token(k, v, bits)
        if self.damage:
            seg.err = self._quant_err(k, v)
        return seg

    def _capture_channel(self, k, v, bits: int) -> _Seg:
        seg = super()._capture_channel(k, v, bits)
        if self.damage:
            seg.err = self._quant_err(k, v)
        return seg

    # -- absolute-position bookkeeping --------------------------------------
    def _append_seg(self, l: int, seg: _Seg):
        """Keep abs/err bookkeeping consistent when merging into `last`."""
        segs = self._segs.setdefault(l, [])
        last = segs[-1] if segs else None
        cap = 10**9 if self.shrink == "layer" else self.G
        will_merge = (self.k_axis == "token" and last is not None
                      and last.bits == seg.bits and last.t + seg.t <= cap)
        if will_merge and getattr(last, "abs_start", None) is not None:
            last.abs_end += seg.t              # merged tokens extend the span
            if last.err is not None and seg.err is not None:
                last.err = torch.cat([last.err, seg.err])
        super()._append_seg(l, seg)

    def _fill_err(self, l: int, s: _Seg):
        e = s.err
        if e is None:
            return
        a, b = s.abs_start, s.abs_end
        cur = self._err.get(l)
        if cur is None or cur.numel() < b:
            pad_n = b - (cur.numel() if cur is not None else 0)
            pad = torch.zeros(pad_n, device=e.device)
            cur = torch.cat([cur, pad]) if cur is not None else pad
            self._err[l] = cur
        cur[a:b] = e.to(cur.dtype)

    def _to_middle(self, l: int, k, v):
        t0 = self._mid_end.get(l, self.n_sink)
        super()._to_middle(l, k, v)
        self._mid_end[l] = t0 + k.shape[2]
        # annotate segments with absolute span (created in order); pos only
        # advances over NEW segments — annotated ones sit before t0 (and
        # evicted tokens leave gaps that abs indexing must survive)
        pos = t0
        for s in self._segs[l]:
            if getattr(s, "abs_start", None) is None:
                s.abs_start, s.abs_end = pos, pos + s.t
                pos += s.t
                self._fill_err(l, s)

    def _seg_abs_range(self, l: int, i: int):
        s = self._segs[l][i]
        a = getattr(s, "abs_start", None)
        if a is not None:
            return a, a + s.t
        # fallback: reconstruct from order
        pos = self.n_sink
        for j, seg in enumerate(self._segs[l]):
            if j == i:
                return pos, pos + seg.t
            pos += seg.t
        return 0, s.t

    # -- ordering ------------------------------------------------------------
    def _seg_err(self, l: int, i: int) -> float:
        e = getattr(self._segs[l][i], "err", None)
        return float(e.mean()) if e is not None and e.numel() else 1.0

    def _order_key(self, l: int, i: int, r) -> float:
        """Ascending = cheapest-to-sacrifice first.

        damage=False: -mean_read (most-read first, baseline; +rd = least-
            read first when invert — CoT-direction control: re-read anchors
            may be the tokens to PROTECT, the opposite of the copy-task
            consumption model).
        damage=True : remaining-value x quant-error (expected damage).
        """
        a, b = self._seg_abs_range(l, i)
        rd = (r[a:min(b, r.numel())].mean().item()
              if (r is not None and a < r.numel()) else 0.0)
        if self.damage:
            return (1.0 / (1.0 + rd)) * self._seg_err(l, i)
        return rd if self.invert else -rd

    def _updated_state(self, l: int):
        return self._updated_reads(l)

    # -- read-ordered trigger ------------------------------------------------
    def _trigger(self, l: int):
        segs = self._segs.get(l, [])
        T = sum(s.t for s in segs)
        r = self._updated_state(l)
        guard = 0
        while self._layer_bits(l) > self.beta * self._budget_bits():
            cand = [i for i, s in enumerate(segs) if s.bits > self.min_bits]
            if not cand:
                # ladder bottom: evict (0-bit rung) or stop honestly
                if self.evict and segs and self._evict_step(l, segs):
                    continue
                break
            has_reads = r is not None and r.numel() > 0 and r.sum() > 0
            if has_reads:
                if self.uniform_depth:
                    # highest tier first; usage orders WITHIN a tier only
                    cand.sort(key=lambda i: (-segs[i].bits,
                                             self._order_key(l, i, r)))
                else:
                    cand.sort(key=lambda i: self._order_key(l, i, r))
            else:
                # no reads recorded yet (prefill-end trigger fires before
                # this layer's first SDPA call) -> uniform layer semantics
                cand = list(cand)
            picked, acc = [], 0
            for i in cand:
                picked.append(i)
                if has_reads:                     # frac applies only when
                    acc += segs[i].t              # read info can guide it
                    if acc >= self.shrink_frac * max(T, 1):
                        break
            for i in picked:
                pre = segs[i].bits
                self._halve_seg(segs[i])
                a, b = self._seg_abs_range(l, i)
                self.events.append({"at_token": self._seen[l], "layer": l,
                                    "from": pre, "to": segs[i].bits,
                                    "n_tok": segs[i].t, "mode": "read",
                                    "abs": [a, b]})
            if self.k_axis == "token":         # keep same-bits runs merged
                cap = self.G
                merged: list[_Seg] = []
                for s in segs:
                    last = merged[-1] if merged else None
                    if (last is not None and last.bits == s.bits
                            and last.t + s.t <= cap):
                        last.abs_end = getattr(s, "abs_end",
                                               last.abs_end + s.t)
                        if last.err is not None and s.err is not None:
                            last.err = torch.cat([last.err, s.err])
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
            assert guard < 400, "read shrink runaway"

    def _evict_step(self, l: int, segs) -> bool:
        """Read-ordered eviction: most-consumed segments drop first."""
        r = self._updated_reads(l)
        T = sum(s.t for s in segs)
        quota = max(1, int(self.shrink_frac * max(T, 1)))
        has_reads = r is not None and r.numel() > 0 and r.sum() > 0
        if has_reads:
            idx = sorted(range(len(segs)),
                         key=lambda i: self._order_key(l, i, r))
        else:
            idx = self._evict_order(segs)
        dropped, acc = [], 0
        for i in idx:
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

    # -- lifecycle -----------------------------------------------------------
    def _usage_stats(self) -> dict:
        """Attention-use structure snapshot from the usage counters.

        top10_share: usage mass held by the top-decile tokens (0.10 =
        perfectly flat; high = concentrated on few tokens).
        last512_share: usage mass on the trailing 512 absolute positions
        (high = recency-dominant structure, i.e. self-generated context;
        low = spread over the whole prompt, i.e. consumption structure).
        These are the discriminative features between task families —
        the basis for any structure-adaptive scheduler.
        """
        tops, lasts, n_layers = [], [], 0
        for l, r in self._read.items():
            if r.numel() < 2 or float(r.sum()) <= 0:
                continue
            n_layers += 1
            v, _ = torch.sort(r)
            k = max(1, r.numel() // 10)
            tops.append(float(v[-k:].sum() / r.sum()))
            last = r[-min(512, r.numel()):]
            lasts.append(float(last.sum() / r.sum()))
        if not n_layers:
            return {}
        return {"top10_share": sum(tops) / n_layers,
                "last512_share": sum(lasts) / n_layers}

    def stats(self) -> dict:
        out = super().stats()
        out["usage_stats"] = self._usage_stats()
        return out

    def init_run(self, ctx: RunContext):
        super().init_run(ctx)
        self._read = {l: torch.zeros(self.n_sink, device=ctx.device
                                     if ctx.device == "cpu" else "cuda")
                      for l in range(ctx.num_layers)}
        self._mid_end = {l: self.n_sink for l in range(ctx.num_layers)}
        self._rd, self._r_snap, self._rd_step = {}, {}, {}
        self._err = {}
