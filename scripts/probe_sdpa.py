"""Probe which SDPA backend torch picks on this GPU for bf16/fp16."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch

q = torch.randn(1, 32, 4096, 128, device="cuda")
for dt in (torch.bfloat16, torch.float16):
    qq, kk, vv = q.to(dt), q.to(dt), q.to(dt)
    for name, ctx in [
        ("flash", torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION)),
        ("mem_efficient", torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION)),
        ("math", torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.MATH)),
    ]:
        try:
            with ctx:
                torch.nn.functional.scaled_dot_product_attention(
                    qq, kk, vv, is_causal=True)
            ok = "OK"
        except Exception as e:
            ok = f"FAIL {type(e).__name__}"
        print(f"{str(dt):16s} {name:14s} {ok}")

# which is the default choice?
from torch.nn.attention import SDPBackend, sdpa_kernel
print("cuda:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
try:
    import torch.backends.cuda as bc
    print("can_use_flash:", bc.flash_sdp_enabled(),
          bool(torch.backends.cudnn.enabled))
except Exception as e:
    print("probe err", e)
