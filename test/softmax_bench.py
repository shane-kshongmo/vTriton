"""Launcher-compatible row-wise softmax kernel (US-SB-005 kernel #3).

A vector-reduction kernel (max + exp + sum over each row) — distinct from the
compute-bound chunk_kda and the pure-MTE vector_add.  Memory-bound: one fused
load + one store per element.
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl

ROWS = 8192
N_COLS = 2048
BLOCK = 2048  # next pow2 >= N_COLS


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_stride + cols, num / den, mask=mask)


def build_inputs():
    torch.manual_seed(0)
    x = torch.randn(ROWS, N_COLS, device="npu", dtype=torch.float32)
    out = torch.empty_like(x)
    return {"x": x, "out": out}


class Model(nn.Module):
    def forward(self, data):
        x, out = data["x"], data["out"]
        softmax_kernel[(ROWS,)](out, x, x.stride(0), out.stride(0), N_COLS, BLOCK=BLOCK)
        return (out,)


def reference(x):
    return torch.softmax(x, dim=-1)


# HBM floor bytes: fused single load + single store, fp32.
HBM_BYTES = 2 * ROWS * N_COLS * 4


if __name__ == "__main__":
    out = Model().forward(build_inputs())[0]
    print("softmax launch OK", out.shape, out.dtype)
