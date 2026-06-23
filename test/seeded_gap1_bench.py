"""Group-V gap-seeded kernel: wrong-unit placement (Gap-1) — for US-SB-006/008.

A vector-eligible floating-point arithmetic chain is deliberately expressed on a
SCALAR value, so bishengir lowers it to the scalar pipe (PIPE_S).  The model's
eligibility oracle says fp elementwise work is Vector-eligible, so a scalar
placement is a Gap-1 (wrong-unit) cause — and with the achievable per-op rates,
the scalar unit is ~395x slower than vector, making the predicted Gap-1 large.

  SEED_SCALAR=1 (default): the FMA chain runs on a single fp32 scalar `s`
      (reduced from the tile) → scalar pipe → slow.
  SEED_SCALAR=0 (the counterfactual "fix"): the identical arithmetic runs on a
      vector tile → vector pipe → fast.

Both produce the SAME result: the chain is affine (s ← s*m + a), so applying it
to `s` then broadcasting equals applying it elementwise to a broadcast of `s`.
The output is `x + chain(s)`; deterministic inputs make it CPU-reconstructable.

The counterfactual removes the seeded Gap-1 via the compiler-reachable
SEED_SCALAR=0 path (a DSL placement change, output-equivalent); the measured
speedup should match the model's predicted Gap-1 / compiler_headroom.
"""
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

N = int(os.environ.get("GAP1_N", str(1 * 1024 * 1024)))
BLOCK = int(os.environ.get("GAP1_BLOCK", "1024"))
NITER = int(os.environ.get("GAP1_NITER", "2048"))
SEED_SCALAR = int(os.environ.get("SEED_SCALAR", "1"))
_M = 1.0001
_A = 0.0001


@triton.jit
def gap1_kernel(
    x_ptr, out_ptr, n,
    SEED_SCALAR: tl.constexpr, BLOCK: tl.constexpr, NITER: tl.constexpr,
    M: tl.constexpr, A: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    if SEED_SCALAR:
        # Affine FMA chain on a single fp32 SCALAR → scalar pipe (the seed).
        s = tl.sum(x, axis=0) / BLOCK
        for _ in range(NITER):
            s = s * M + A
        out = x + s
    else:
        # Identical arithmetic on a VECTOR tile → vector pipe (the fix).
        v = tl.sum(x, axis=0) / BLOCK + tl.zeros([BLOCK], dtype=tl.float32)
        for _ in range(NITER):
            v = v * M + A
        out = x + v

    tl.store(out_ptr + offs, out, mask=mask)


def _det_input(n, modulus, device, dtype=torch.float32):
    idx = torch.arange(n, device=device, dtype=torch.int64)
    return ((idx % modulus).to(dtype) / float(modulus))


def build_inputs():
    device = "npu"
    x = _det_input(N, 4099, device)
    out = torch.empty_like(x)
    return {"x": x, "out": out}


class Model(nn.Module):
    def forward(self, data):
        x, out = data["x"], data["out"]
        grid = (triton.cdiv(N, BLOCK),)
        gap1_kernel[grid](
            x, out, N,
            SEED_SCALAR=SEED_SCALAR, BLOCK=BLOCK, NITER=NITER, M=_M, A=_A,
        )
        return (out,)


def reference(x):
    """CPU reference: per-tile mean → affine chain → broadcast-add."""
    xb = x.reshape(-1, BLOCK)
    s = xb.mean(dim=1, keepdim=True)
    for _ in range(NITER):
        s = s * _M + _A
    return (xb + s).reshape(-1)


def reference_cpu():
    return (reference(_det_input(N, 4099, "cpu")),)


if __name__ == "__main__":
    data = build_inputs()
    out = Model().forward(data)[0]
    print("gap1 launch OK", tuple(out.shape), "SEED_SCALAR=", SEED_SCALAR, "NITER=", NITER)
