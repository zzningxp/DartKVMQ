"""Debug: kvarn batched (M=2) vs per-group (M=1) intermediate comparison."""
import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dartkv.methods.kvarn import KvarnMethod
from dartkv.methods.base import RunContext
from dartkv.quant.hadamard import hadamard_matrix
from dartkv.quant.varn import variance_normalize

torch.manual_seed(7)
B, Hh, T, D, G = 1, 2, 128, 64, 64
M = T // G
tok = torch.rand(B, Hh, T, 1) * 8 + 0.2
ch = torch.rand(B, Hh, 1, D) * 8 + 0.2
k = torch.randn(B, Hh, T, D) * tok * ch
v = torch.randn(B, Hh, T, D) * tok * ch

m = KvarnMethod(bits_k=16, bits_v=16, group_size=G, varn_iters=8)
m.init_run(RunContext(num_layers=1, num_kv_heads=Hh, head_dim=D,
                      dtype="torch.float32", device="cpu"))

kb, vb = m.decode_tiles(m.encode_tiles(k, v, 0), 0)
s1 = m.encode_tile(k[:, :, :G], v[:, :, :G], 0)
k1, v1 = m.decode_tiles([s1], 0)
print("K batched-vs-group1 err:", (kb[:, :, :G] - k1).abs().max().item())
print("K batched-vs-group2 err:", (kb[:, :, G:] - k[:, :, G:]).abs().max().item())
print("V batched-vs-original rel:",
      ((vb - v).norm() / v.norm()).item())

# step through intermediates manually for K
Hm = hadamard_matrix(D)
kr = k.float() @ Hm
kt2 = (kr.reshape(B, Hh, M, G, D).permute(1, 0, 2, 4, 3)
       .reshape(Hh * B * M, D, G))
kr1 = k[:, :, :G].float() @ Hm
kt1 = (kr1.reshape(B, Hh, 1, G, D).permute(1, 0, 2, 4, 3)
       .reshape(Hh * B, D, G))
print("tile match g0:", torch.allclose(kt2[0], kt1[0]),
      torch.allclose(kt2[2], kt1[0]))  # tile ordering check
n2, sc2, sr2 = variance_normalize(kt2, iterations=8)
n1, sc1, sr1 = variance_normalize(kt1, iterations=8)
print("varn balanced match:", torch.allclose(n2[0], n1[0]),
      torch.allclose(n2[2], n1[0]))
print("scales match:", torch.allclose(sc2[0], sc1[0]),
      torch.allclose(sr2[0], sr1[0]))
# after mapping back, what does single decode give?
print("batched kr_n reshaped back matches tiles:",
      torch.allclose(kr_n := n2.reshape(Hh, B, M, D, G).permute(1, 0, 2, 4, 3),
                     n2.reshape(Hh, B, M, D, G).permute(1, 0, 2, 4, 3)))
