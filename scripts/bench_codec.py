"""Micro-bench: KVarN/KIVI encode_tiles + decode_tiles on realistic shapes.
Estimates prefill-side overhead before the real Line-Retrieval runs.
"""
import os
import sys
import pathlib
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from dartkv.methods.base import RunContext  # noqa: E402
from dartkv.methods.kivi import KiviMethod  # noqa: E402
from dartkv.methods.kvarn import KvarnMethod  # noqa: E402

DEV = "cuda"


def bench(cls, kw, T, label):
    Hh, D, G = 8, 128, 128
    m = cls(group_size=G, **kw)
    m.init_run(RunContext(num_layers=1, num_kv_heads=Hh, head_dim=D,
                          dtype="torch.bfloat16", device=DEV))
    k = torch.randn(1, Hh, T, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(1, Hh, T, D, device=DEV, dtype=torch.bfloat16)
    for _ in range(3):  # warmup
        sts = m.encode_tiles(k, v, 0)
        m.decode_tiles(sts, 0)
    torch.cuda.synchronize()
    t0 = time.time()
    N = 10
    for _ in range(N):
        sts = m.encode_tiles(k, v, 0)
    torch.cuda.synchronize()
    enc = (time.time() - t0) / N
    t0 = time.time()
    for _ in range(N):
        m.decode_tiles(sts, 0)
    torch.cuda.synchronize()
    dec = (time.time() - t0) / N
    print(f"{label:28s} T={T:5d} encode={enc*1000:7.1f}ms "
          f"decode={dec*1000:7.1f}ms  (x36 layers: enc={enc*36:.2f}s "
          f"dec={dec*36*1000:.0f}ms/step)")


for T in (1280, 8704):  # ~100-line, ~600-line prompts
    bench(KvarnMethod, dict(bits_k=2, bits_v=2, varn_iters=8), T, "kvarn-8it")
    bench(KvarnMethod, dict(bits_k=2, bits_v=2, varn_iters=0), T, "kvarn_h")
    bench(KiviMethod, dict(bits_k=2, bits_v=2), T, "kivi")
