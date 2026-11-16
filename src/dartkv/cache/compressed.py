"""CompressedCache: a transformers Cache that routes K/V through a
CompressionMethod and returns fully dequantized (K, V) to the model each
step. Numerically equivalent to fused-dequant attention kernels (up to fp
rounding) while storage actually stays compressed.

Pass detection: transformers calls update() layer-by-layer. When layer_idx
wraps (new call has layer_idx <= previous), a pass boundary occurred:
  first pass = prefill, later = decode steps.
"""
from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

from ..methods.base import CompressionMethod, RunContext


class CompressedCache(DynamicCache):
    def __init__(self, method: CompressionMethod, model_config=None,
                 skip_layers: int = 0):
        super().__init__(config=model_config)
        self._config_ref = model_config
        self.method = method
        self.skip_layers = skip_layers  # boundary skip: first+last N layers
        self._skip: set[int] = set()
        self._raw: dict[int, tuple] = {}
        self._last_layer = -1
        self._pass_count = 0
        self._seen: dict[int, int] = {}
        self._ctx_ready = False

    def _ensure_ctx(self, k):
        if self._ctx_ready:
            return
        cfg = getattr(self, "_config_ref", None)
        num_layers = getattr(cfg, "num_hidden_layers", 0) if cfg else 0
        ctx = RunContext(
            num_layers=num_layers or 32,
            num_kv_heads=k.shape[1],
            head_dim=k.shape[-1],
            dtype=str(k.dtype),
            device=str(k.device),
        )
        self.method.init_run(ctx)
        if hasattr(self.method, "attach"):  # e.g. RDKV probe capture
            self.method.attach(None)
        if self.skip_layers > 0:
            # mirrors official KVARN_BOUNDARY_SKIP_LAYERS / vLLM
            # kv_cache_dtype_skip_layers (TurboQuant PR #38479 usage):
            # first N and last N layers stay full precision
            n, L = self.skip_layers, ctx.num_layers
            self._skip = set(range(n)) | set(range(L - n, L))
        self._ctx_ready = True

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        self._ensure_ctx(key_states)
        if layer_idx <= self._last_layer:
            # pass boundary
            kind = "prefill" if self._pass_count == 0 else "decode"
            self.method.on_pass_end(kind)
            self._pass_count += 1
        self._last_layer = layer_idx
        self._seen[layer_idx] = self._seen.get(layer_idx, 0) + key_states.shape[2]
        if layer_idx in self._skip:
            prev = self._raw.get(layer_idx)
            if prev is None:
                self._raw[layer_idx] = (key_states, value_states)
            else:
                self._raw[layer_idx] = (
                    torch.cat([prev[0], key_states], dim=2),
                    torch.cat([prev[1], value_states], dim=2))
            return self._raw[layer_idx]
        self.method.ingest(layer_idx, key_states, value_states,
                           is_first_pass=self._pass_count == 0)
        return self.method.get_cache(layer_idx)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen.get(layer_idx, 0)

    # transformers 5.x sometimes probes these; keep them cheap/consistent
    @property
    def seen_tokens(self) -> int:
        return self.get_seq_length(0)
