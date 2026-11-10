"""Unit tests for CompressedCache: pass-boundary detection, seq tracking,
delegation to the method — using a mock forward loop, no model download."""
import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dartkv.cache.compressed import CompressedCache  # noqa: E402
from dartkv.methods.kvarn import KvarnMethod  # noqa: E402
from dartkv.methods.base import RunContext  # noqa: E402
from transformers import PretrainedConfig  # noqa: E402


def _cfg():
    # DynamicCache requires a real config (get_text_config)
    return PretrainedConfig(num_hidden_layers=2, num_key_value_heads=2,
                            head_dim=32)


def _pass(cache, layers, chunk):
    """Mimic HF: one update call per layer with the whole chunk (as HF does:
    prefill = full prompt per layer; decode step = 1 token per layer)."""
    return [cache.update(chunk, chunk, l) for l in range(layers)]


def test_pass_boundary_and_lengths():
    m = KvarnMethod(bits_k=2, bits_v=2, group_size=8, sink_tokens=2,
                    varn_iters=4)
    cache = CompressedCache(m, model_config=_cfg())
    fired = []
    m.on_pass_end = lambda kind: fired.append(kind)      # record hooks
    _pass(cache, 2, torch.randn(1, 2, 10, 32))          # prefill
    assert fired == []                     # boundary deferred to next pass
    assert cache._pass_count == 0
    assert cache.get_seq_length(0) == 10
    assert cache.get_seq_length(1) == 10
    for _ in range(3):                                   # decode steps
        _pass(cache, 2, torch.randn(1, 2, 1, 32))
    assert fired == ["prefill", "decode", "decode"]
    assert cache._pass_count == 3
    assert cache.get_seq_length(0) == 13
    # update returns the method's full dequantized cache
    k, v = cache.update(torch.randn(1, 2, 1, 32), torch.randn(1, 2, 1, 32), 0)
    assert k.shape == (1, 2, 14, 32)
    assert m.total_tokens(0) == 14
    # ctx inferred from config + tensor
    assert m.ctx.num_layers == 2 and m.ctx.num_kv_heads == 2


def test_no_config_still_infers():
    m = KvarnMethod(bits_k=2, bits_v=2, group_size=8, sink_tokens=2)
    cache = CompressedCache(m)             # model_config=None
    cache.update(torch.randn(1, 2, 5, 32), torch.randn(1, 2, 5, 32), 0)
    # falls back to a default layer count; first pass still works
    assert cache.get_seq_length(0) == 5
    assert m.ctx.head_dim == 32


def test_seen_tokens_property():
    m = KvarnMethod(bits_k=2, bits_v=2, group_size=8, sink_tokens=2)
    cache = CompressedCache(m, model_config=_cfg())
    cache.update(torch.randn(1, 2, 7, 32), torch.randn(1, 2, 7, 32), 0)
    assert cache.seen_tokens == 7


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                fails += 1
                import traceback; traceback.print_exc()
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
