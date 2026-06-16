"""Launcher-compatible RMSNorm forward kernel (US-SB-005 kernel #5).

Single mean-of-squares reduction + rsqrt scale + weight.  A lighter-reduction
cousin of layernorm (no mean subtraction), rounding out the n>=5 set.
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl

ROWS = 8192
N = 2048
BLOCK = 2048
EPS = 1e-6


@triton.jit
def rmsnorm_kernel(X, Y, W, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)
    w = tl.load(W + cols, mask=mask, other=0.0)
    tl.store(Y + row * stride + cols, x * rstd * w, mask=mask)


def build_inputs():
    torch.manual_seed(0)
    x = torch.randn(ROWS, N, device="npu", dtype=torch.float32)
    w = torch.randn(N, device="npu", dtype=torch.float32)
    out = torch.empty_like(x)
    return {"x": x, "w": w, "out": out}


class Model(nn.Module):
    def forward(self, data):
        x, w, out = data["x"], data["w"], data["out"]
        rmsnorm_kernel[(ROWS,)](x, out, w, x.stride(0), N, EPS, BLOCK=BLOCK)
        return (out,)


def reference(x, w):
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(ms + EPS) * w


# HBM floor: read X + write Y (W negligible), fp32.
HBM_BYTES = 2 * ROWS * N * 4


if __name__ == "__main__":
    out = Model().forward(build_inputs())[0]
    print("rmsnorm launch OK", out.shape, out.dtype)
