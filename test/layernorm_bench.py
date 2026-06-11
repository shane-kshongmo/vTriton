"""Launcher-compatible LayerNorm forward kernel (US-SB-005 kernel #4).

Per-row mean/variance reduction + affine transform.  Memory-bound vector kernel
with a heavier compute mix than softmax (two reductions + rsqrt).
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl

ROWS = 8192
N = 2048
BLOCK = 2048
EPS = 1e-5


@triton.jit
def layernorm_kernel(X, Y, W, B, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    tl.store(Y + row * stride + cols, xc * rstd * w + b, mask=mask)


def build_inputs():
    torch.manual_seed(0)
    x = torch.randn(ROWS, N, device="npu", dtype=torch.float32)
    w = torch.randn(N, device="npu", dtype=torch.float32)
    b = torch.randn(N, device="npu", dtype=torch.float32)
    out = torch.empty_like(x)
    return {"x": x, "w": w, "b": b, "out": out}


class Model(nn.Module):
    def forward(self, data):
        x, w, b, out = data["x"], data["w"], data["b"], data["out"]
        layernorm_kernel[(ROWS,)](x, out, w, b, x.stride(0), N, EPS, BLOCK=BLOCK)
        return (out,)


def reference(x, w, b):
    return torch.nn.functional.layer_norm(x, (N,), w, b, EPS)


# HBM floor: read X + write Y (W/B negligible), fp32.
HBM_BYTES = 2 * ROWS * N * 4


if __name__ == "__main__":
    out = Model().forward(build_inputs())[0]
    print("layernorm launch OK", out.shape, out.dtype)
