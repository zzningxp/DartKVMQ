"""RDKV: post-prefill joint eviction+quantization via rate-distortion
bit allocation (arXiv 2605.08317, full reimplementation — no official code).

Pipeline (paper Alg. 1 + Sec. 3):
  Stage 1  probe weights from a trailing observation window of S_w queries:
     w_t (V, per token)    = sum over window queries tau and group heads g
                             of attention prob a_{tau,t}, AvgPool1d(k=5)
                             smoothed along t (SnapKV convention)
     w_c (K, per channel)  = 1/sqrt(d) * sum_g ||Q_win[:,c]||_2 ||K[:,c]||_2
                             (ThinK score; window Q, full-T K)
  Stage 2  V token bits {0,2,4,8,16} by MCKP (Lagrangian bisection) over
           per-(layer,head) distortion tables eps^V(b); b=0 evicts.
  Stage 3  K channel bits over kept tokens, budget B_K/|T_kept|.
  Decode   new tokens append FP16 (Zone C).

Distortion tables are measured on-the-fly from the actual prefill K/V
(subsampled), matching the paper's per-(l,h) calibration intent; a cached
table can be supplied instead. Budget: n FP16-equivalent tokens per layer
=> per (l,h): sum_t b_t <= 16*n (V), sum_c b_c <= 16*n/|T_kept| (K);
K:V budget split r_k=0.5 (paper: best).

Probe Q capture: wraps F.scaled_dot_product_attention between attach() and
prefill end — SDPA's q argument is post-RoPE, exactly what the probe needs.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import CompressionMethod, RunContext
from ..alloc.mckp import mckp_lagrangian


class RdkvMethod(CompressionMethod):
    name = "rdkv"
    online = False  # compresses once, at prefill end

    def __init__(self, budget_tokens: int = 1024, window: int = 32,
                 pool: int = 5, r_k: float = 0.5,
                 eps_sub: int = 512,
                 eps_cache: dict | None = None,
                 bit_set: tuple = (0, 2, 4, 8, 16), tol: float = 1e-2):
        super().__init__()
        self.n = budget_tokens
        self.S_w = window
        self.pool_k = pool
        self.r_k = r_k
        self.eps_sub = eps_sub
        self.eps_cache = eps_cache  # optional {(kind, l, h): {b: eps}}
        self.bit_set = list(bit_set)
        self.tol = tol
        # filled at runtime
        self.state: list[dict] = []
        self._sdpa_orig = None
        self._probe_q: list[torch.Tensor] = []  # post-RoPE q per layer
        self._n_calls = 0

    # -- probe capture -----------------------------------------------------
    def attach(self, model):
        """Patch F.scaled_dot_product_attention to record q (post-RoPE)."""
        import torch.nn.functional as Fn

        if self._sdpa_orig is not None:
            return
        self._sdpa_orig = Fn.scaled_dot_product_attention
        method = self
        L = self.ctx.num_layers if self.ctx else None

        def wrapped(q, *a, **kw):
            if method._sdpa_orig is not None and q.ndim == 4:
                if L is None or method._n_calls < L:  # prefill pass only
                    method._probe_q.append(
                        (q.detach()[0, :, -method.S_w:].float().cpu(),
                         q.shape[2]))                  # (q_win, T_total)
                    method._n_calls += 1
            return method._sdpa_orig(q, *a, **kw)

        Fn.scaled_dot_product_attention = wrapped
        torch.nn.functional.scaled_dot_product_attention = wrapped

    def detach(self):
        if self._sdpa_orig is not None:
            torch.nn.functional.scaled_dot_product_attention = self._sdpa_orig
            self._sdpa_orig = None

    # -- lifecycle ---------------------------------------------------------
    def init_run(self, ctx: RunContext):
        super().init_run(ctx)
        self.state = [{"k": None, "v": None, "compressed": None,
                       "new_k": None, "new_v": None, "stats": None}
                      for _ in range(ctx.num_layers)]
        self._probe_q = []
        self._n_calls = 0

    def ingest(self, layer_idx: int, k, v, is_first_pass: bool):
        st = self.state[layer_idx]
        if not is_first_pass or st["compressed"] is not None:
            self.ingest_decode(layer_idx, k, v)   # Zone C append
            return
        if st["k"] is None:
            st["k"], st["v"] = k.clone(), v.clone()
        else:  # chunked prefill (rare) — append
            st["k"] = torch.cat([st["k"], k], dim=2)
            st["v"] = torch.cat([st["v"], v], dim=2)

    def on_pass_end(self, kind: str):
        if kind != "prefill" or self.state[0]["compressed"] is not None:
            return
        try:
            self._compress()
        finally:
            self.detach()  # never leave the global patch armed

    # -- core --------------------------------------------------------------
    def _probe_weights(self, l: int, k: torch.Tensor):
        """w_t [H_kv, T], w_c [H_kv, D] for layer l from the captured q."""
        q_win, T_q = self._probe_q[l]
        T = k.shape[2]
        if T_q != T:
            raise RuntimeError(
                f"RDKV probe expects single-chunk prefill (probe saw "
                f"{T_q} queries, cache holds {T} tokens)")
        q = q_win.to(k.device)                       # [H_q, win, D]
        H_q, win, D = q.shape
        H_kv = k.shape[1]
        assert H_q % H_kv == 0
        r = H_q // H_kv
        win = min(win, T)
        kw = k[0].float()                            # [H_kv, T, D]
        scale = D ** -0.5
        # scores per kv-head group: [r, win, T]
        w_t = torch.zeros(H_kv, T, device=k.device)
        w_c = torch.zeros(H_kv, D, device=k.device)
        for h in range(H_kv):
            g = q[h * r:(h + 1) * r]                 # [r, win, D]
            att = torch.softmax(g @ kw[h].T * scale, dim=-1)  # [r, win, T]
            # causal within the window (SnapKV uses causal obs scores)
            pos = torch.arange(T - win, T, device=k.device)
            mask = torch.arange(T, device=k.device)[None, :] > pos[:, None]
            att = att.masked_fill(mask[None], 0.0)
            att = att / att.sum(-1, keepdim=True).clamp_min(1e-9)
            s = att.sum((0, 1))                      # [T] sum over r, tau
            if self.pool_k > 1:                      # smooth along T axis
                s = F.avg_pool1d(s.view(1, 1, -1), kernel_size=self.pool_k,
                                 stride=1,
                                 padding=self.pool_k // 2).view(-1)
            w_t[h] = s
            # ThinK channel score: window-Q norm x full-K norm, summed over
            # group heads (GQA extension of Prop. 3.2)
            qn = g.norm(dim=1).sum(0)                # [D] sum over r heads
            kn = kw[h].norm(dim=0)                   # [D]
            w_c[h] = scale * qn * kn
        return w_t, w_c

    def _eps_tables_head(self, l: int, h: int, k: torch.Tensor,
                         v: torch.Tensor):
        """On-the-fly eps^V(b), eps^K(b) per (l,h); or from cache."""
        if self.eps_cache is not None:
            e = self.eps_cache
            return e[("V", l, h)], e[("K", l, h)]
        from ..quant.uniform import quantize_asym, dequantize_asym

        def nmse_rows(x):
            """mean relative error quantizing rows of x at each bit width."""
            out = {}
            T = x.shape[0]
            idx = torch.randperm(T, device=x.device)[: self.eps_sub]
            xs = x[idx].float()
            ref = xs.pow(2).mean().clamp_min(1e-12)
            for b in (2, 4, 8):
                q, s, z = quantize_asym(xs, b, dim=-1, group_size=None)
                xh = dequantize_asym(q, s, z)
                out[b] = ((xs - xh).pow(2).mean() / ref).item()
            return out

        v_h = v[0, h].float()                        # [T, D] tokens=rows
        k_h = k[0, h].float()                        # [T, D] channels=cols
        eV = nmse_rows(v_h)                          # per-token rows
        eK = nmse_rows(k_h.T.contiguous())           # per-channel rows
        for e in (eV, eK):
            e[0] = 1.0
            e[16] = 0.0
        return eV, eK

    def _compress(self):
        """Dense-repro semantics:
        - V token set is LAYER-level (w_t summed across kv-heads, one MCKP
          per layer) — dense attention cannot express per-head ragged token
          sets; SnapKV's own convention aggregates across heads the same
          way. Deviation from the paper's per-(l,h) V allocation, required
          for the dequant->dense path; TriZone's per-head packing would be
          the P5 speed path.
        - K channel bits stay per-(l,h): zeroing a channel is exactly
          removable from the dot product, so dense SDPA expresses it.
        - Evicted tokens are physically REMOVED (rows dropped), not zeroed.
        """
        from ..quant.uniform import quantize_asym, dequantize_asym
        ctx = self.ctx
        n = self.n
        for l in range(ctx.num_layers):
            st = self.state[l]
            k, v = st["k"], st["v"]                  # [1, H_kv, T, D] (orig dtype)
            _, H, T, D = k.shape
            w_t, w_c = self._probe_weights(l, k)
            # budgets (paper Sec. 3.1): B_head = 32*n*d bits; split r_k
            # V: sum_t b_t <= B_V/d = (1-r_k)*32*n   (16n at r_k=0.5)
            # K: sum_c b_c*|T_kept| <= B_K = r_k*32*n*d
            budget_v = (1.0 - self.r_k) * 32.0 * n
            budget_k = self.r_k * 32.0 * n * D

            tables = [self._eps_tables_head(l, h, k, v) for h in range(H)]
            eV = {b: sum(t[0][b] for t in tables) / H for b in (0, 2, 4, 8, 16)}

            # Stage 2: layer-level V token allocation (eviction at b=0)
            bV = mckp_lagrangian(w_t.sum(0), eV, budget_v,
                                 bit_set=self.bit_set, tol=self.tol)
            keep = bV > 0
            n_kept = int(keep.sum())
            st["stats"] = {
                "T": T, "kept": n_kept, "retention": n_kept / max(1, T),
                "bpe_v_payload": int(bV.sum()) / (16.0 * T),
                "bpe_k_payload": None,
            }
            if n_kept == 0:
                st["compressed"] = (
                    torch.zeros(1, H, 0, D, dtype=self._dtype()),
                    torch.zeros(1, H, 0, D, dtype=self._dtype()))
                st["k"] = st["v"] = None
                continue

            Kq = torch.empty(H, n_kept, D, device=k.device)
            Vq = torch.empty(H, n_kept, D, device=v.device)
            sum_bk = 0
            bits_row = bV[keep]
            for h in range(H):
                eK_h = tables[h][1]
                bK = mckp_lagrangian(w_c[h], eK_h,
                                     max(budget_k / n_kept, 1e-6),
                                     bit_set=self.bit_set, tol=self.tol)
                sum_bk += int(bK.sum())
                # V: per-token asym along D at b_t bits (kept rows only)
                vb = v[0, h][keep].float()           # [Tk, D]
                vq = torch.empty_like(vb)
                for b in torch.unique(bits_row):
                    m = bits_row == b
                    if int(b) == 16:
                        vq[m] = vb[m]
                        continue
                    q, s, z = quantize_asym(vb[m], int(b), dim=-1,
                                            group_size=None)
                    vq[m] = dequantize_asym(q, s, z)
                Vq[h] = vq.to(Vq.dtype)
                # K: per-channel asym over kept tokens at b_c bits;
                # b_c = 0 -> channel zeroed (exact removal from q.k)
                kb = k[0, h][keep].float()           # [Tk, D]
                kq = torch.zeros_like(kb)
                for b in torch.unique(bK):
                    m = bK == b
                    if int(b) == 0:
                        continue
                    if int(b) == 16:
                        kq[:, m] = kb[:, m]
                        continue
                    q, s, z = quantize_asym(kb[:, m].T.contiguous(), int(b),
                                            dim=-1, group_size=None)
                    kq[:, m] = dequantize_asym(q, s, z).T
                Kq[h] = kq.to(Kq.dtype)
            st["stats"]["bpe_k_payload"] = (sum_bk * n_kept) / (16.0 * D * T)
            st["compressed"] = (Kq[None].to(self._dtype()),   # [1,H,Tk,D]
                                Vq[None].to(self._dtype()))
            st["k"] = st["v"] = None  # free raw stash

    def _dtype(self):
        d = self.ctx.dtype
        d = d.split(".")[-1]  # tolerate "torch.bfloat16" from str(k.dtype)
        return {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32}[d]

    # -- serving -----------------------------------------------------------
    def get_cache(self, layer_idx: int):
        st = self.state[layer_idx]
        if st["compressed"] is None:
            return st["k"].to(self._dtype()), st["v"].to(self._dtype())
        Kc, Vc = st["compressed"]
        if st["new_k"] is not None and st["new_k"].numel():
            return (torch.cat([Kc, st["new_k"]], dim=2),
                    torch.cat([Vc, st["new_v"]], dim=2))
        return Kc, Vc

    def ingest_decode(self, layer_idx: int, k, v):
        """Zone C: decode tokens append FP16, uncompressed (paper Stage 4)."""
        st = self.state[layer_idx]
        kd, vd = k, v
        if st["new_k"] is not None:
            kd = torch.cat([st["new_k"], k], dim=2)
            vd = torch.cat([st["new_v"], v], dim=2)
        st["new_k"], st["new_v"] = kd, vd

    def bits_report(self):
        if self.state and self.state[0]["stats"]:
            s = [st["stats"] for st in self.state if st["stats"]]
            return {"bpe_k": sum(x["bpe_k_payload"] for x in s) / len(s),
                    "bpe_v": sum(x["bpe_v_payload"] for x in s) / len(s),
                    "fp16_eq_tokens_per_head": float(self.n),
                    "retention": sum(x["retention"] for x in s) / len(s)}
        return {"bpe_k": None, "bpe_v": None,
                "fp16_eq_tokens_per_head": float(self.n), "retention": None}
