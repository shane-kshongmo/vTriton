"""Group-V gap-seeded kernel: coalescing / transfer efficiency (Gap-2) — US-SB-006.

A full copy of a square [S,S] fp32 tensor, where each program copies one "line":

  SEED_STRIDED=0 (coalesced "fix"): program p copies ROW p — one contiguous
      S-element transfer (large packet → efficient).
  SEED_STRIDED=1 (the seed): program p copies COLUMN p — S single-element
      transfers at stride S (tiny packets → small-transfer amortization penalty).

Both produce out == in (every element copied exactly once), with the SAME grid
(S programs) and the SAME total bytes — so the only thing that changes is the
per-transfer packet size, isolating the Gap-2 coalescing cost.  The counterfactual
removes the seed via the compiler-reachable SEED_STRIDED=0 access pattern
(output-identical); the measured speedup should match the model's predicted gap2.
"""
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

S = int(os.environ.get("GAP2_S", "2048"))         # square side (also the tile/BLOCK)
SEED_STRIDED = int(os.environ.get("SEED_STRIDED", "1"))


@triton.jit
def copy_kernel(in_ptr, out_ptr, s, SEED_STRIDED: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)            # line index 0..S-1
    idx = tl.arange(0, BLOCK)
    mask = idx < s
    if SEED_STRIDED:
        offs = idx * s + pid               # column pid: stride s, tiny packets
    else:
        offs = pid * s + idx               # row pid: contiguous, large packet
    x = tl.load(in_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)


def _det_input(n, modulus, device, dtype=torch.float32):
    idx = torch.arange(n, device=device, dtype=torch.int64)
    return ((idx % modulus).to(dtype) / float(modulus))


def build_inputs():
    device = "npu"
    x = _det_input(S * S, 4099, device)
    out = torch.empty_like(x)
    return {"x": x, "out": out}


class Model(nn.Module):
    def forward(self, data):
        x, out = data["x"], data["out"]
        copy_kernel[(S,)](x, out, S, SEED_STRIDED=SEED_STRIDED, BLOCK=S)
        return (out,)


def reference_cpu():
    # The kernel is a full copy: out == in regardless of access pattern.
    return (_det_input(S * S, 4099, "cpu"),)


if __name__ == "__main__":
    out = Model().forward(build_inputs())[0]
    print("gap2 launch OK", tuple(out.shape), "SEED_STRIDED=", SEED_STRIDED, "S=", S)
