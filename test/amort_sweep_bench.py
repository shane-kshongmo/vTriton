"""Transfer-size (packet) amortization microbench for 910B3 — calibrates pkt_param.

Fixed grid of NPROG programs, fixed total volume.  Each program copies NCHUNK
runs of CHUNK contiguous elements, where the runs are interleaved across programs
(layout [NCHUNK, NPROG, CHUNK]) so consecutive runs are NPROG*CHUNK apart and the
compiler cannot coalesce them.  The hardware therefore moves CHUNK*4-byte packets.

Sweep CHUNK (via env) with NCHUNK = TOTAL_PER_PROG // CHUNK to hold total bytes
and grid constant; achieved bandwidth = total_bytes / time isolates the
small-packet amortization penalty (independent of the Gap-2 strided counterfactual
it will validate).
"""
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

NPROG = int(os.environ.get("AMORT_NPROG", "2048"))
TOTAL_PER_PROG = int(os.environ.get("AMORT_TOTAL_PER_PROG", "4096"))
CHUNK = int(os.environ.get("AMORT_CHUNK", "1"))
NCHUNK = TOTAL_PER_PROG // CHUNK
N = NPROG * TOTAL_PER_PROG
PACKET_BYTES = CHUNK * 4


@triton.jit
def amort_copy(in_ptr, out_ptr, nprog, NCHUNK: tl.constexpr, CHUNK: tl.constexpr):
    p = tl.program_id(axis=0)
    idx = tl.arange(0, CHUNK)
    for c in range(NCHUNK):
        offs = c * (nprog * CHUNK) + p * CHUNK + idx
        x = tl.load(in_ptr + offs)
        tl.store(out_ptr + offs, x)


def _det_input(n, modulus, device, dtype=torch.float32):
    i = torch.arange(n, device=device, dtype=torch.int64)
    return ((i % modulus).to(dtype) / float(modulus))


def build_inputs():
    x = _det_input(N, 4099, "npu")
    out = torch.empty_like(x)
    return {"x": x, "out": out}


class Model(nn.Module):
    def forward(self, data):
        x, out = data["x"], data["out"]
        amort_copy[(NPROG,)](x, out, NPROG, NCHUNK=NCHUNK, CHUNK=CHUNK)
        return (out,)


def reference_cpu():
    return (_det_input(N, 4099, "cpu"),)


if __name__ == "__main__":
    out = Model().forward(build_inputs())[0]
    print("amort launch OK", tuple(out.shape), "CHUNK=", CHUNK, "packet_bytes=", PACKET_BYTES, "NCHUNK=", NCHUNK)
