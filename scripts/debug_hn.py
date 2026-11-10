"""Debug the orthonormal-H regression: 16-bit KVarN roundtrip must be exact.
Check each intermediate for a (B=1, H=4, T=128, D=64) imbalanced tile."""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from dartkv.methods.base import RunContext  # noqa: E402
from dartkv.methods.kvarn import KvarnMethod  # noqa: E402
from dartkv.quant.hadamard import hadamard_matrix  # noqa: E402
from dartkv.quant.varn import variance_normalize  # noqa: E402
from tests.test_methods import _ctx, _imbalanced_tile  # noqa: E402

torch.manual_seed(0)
k, v = _imbalanced_tile(1, 4, 128, 64, seed=2)
print("k norm", k.norm().item(), "std", k.float().std().item())

D = 64
H1 = hadamard_matrix(D)            # +-1
Hn = H1 / (D ** 0.5)

# raw rotation roundtrip sanity
kr1 = k.float() @ H1
krn = k.float() @ Hn
print("rot roundtrip +-1:", ((kr1 @ H1 / D) - k.float()).norm().item())
print("rot roundtrip  Hn:", ((krn @ Hn) - k.float()).norm().item())

# VarN directly on the rotated tile at both scales
for name, kr in (("+-1", kr1), ("Hn", krn)):
    B, Hh, T, Dd = kr.shape
    G = 128
    kt = (kr.reshape(B, Hh, 1, G, Dd).permute(1, 0, 2, 4, 3)
          .reshape(Hh * B, Dd, G))
    bal, sc, sr = variance_normalize(kt, iterations=8)
    rec = bal * sc * sr
    print(f"[{name}] varn roundtrip rel err:",
          ((rec - kt).norm() / kt.norm()).item(),
          "s_col range", sc.min().item(), sc.max().item(),
          "s_row range", sr.min().item(), sr.max().item())

# full method roundtrip at 16 bits with Hn
m = KvarnMethod(bits_k=16, bits_v=16, group_size=128, varn_iters=8)
m.init_run(_ctx(dim=64))
st = m.encode_tile(k, v, 0)
kd, vd = m.decode_tile(st, 0)
print("method rel_k:", ((k - kd).norm() / k.norm()).item())
print("stored hadamard norm:",
      st.k_params["hadamard"].norm().item(), "(sqrt(D)=8 -> orthonormal)")
