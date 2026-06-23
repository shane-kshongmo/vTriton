"""Optimal contiguous stream-copy microbench for 910B3 — M-1/M-2 calibration.

Unlike `amort_sweep_bench.py` (which *deliberately* interleaves runs to force
small packets and measure the amortization penalty), this bench is the **optimal
copy**: program ``p`` owns a single contiguous slab ``[p*BLOCK, (p+1)*BLOCK)`` and
streams it through UB in large ``TILE``-element loads.  Packets are therefore as
large as the hardware allows, so the achieved aggregate bandwidth approaches the
true HBM peak.

Two calibration sweeps are driven from this one kernel (see
`scripts/measure_hbm_bw.py`):

* **M-1 (peak):** fix a saturating grid, sweep ``TILE``/``BLOCK`` → max aggregate
  GB/s is the true peak.
* **M-2 (contention):** fix a large ``BLOCK``/``TILE``, sweep the program/core
  count ``n`` → per-core BW(n) = aggregate(n) / min(n, n_cores).

Inputs are deterministic (``arange % p / p``) so the copy is CPU-verifiable
without RNG (NPU ``randn`` != CPU ``randn``).
"""
from __future__ import annotations

import statistics
import time

import torch
import triton
import triton.language as tl


@triton.jit
def stream_copy(in_ptr, out_ptr, BLOCK: tl.constexpr, TILE: tl.constexpr):
    p = tl.program_id(axis=0)
    base = p * BLOCK
    for t in range(0, BLOCK, TILE):
        offs = base + t + tl.arange(0, TILE)
        x = tl.load(in_ptr + offs)
        tl.store(out_ptr + offs, x)


def _det_input(n, modulus, device, dtype=torch.float32):
    i = torch.arange(n, device=device, dtype=torch.int64)
    return (i % modulus).to(dtype) / float(modulus)


def run_config(
    nprog: int,
    block: int,
    tile: int,
    warmup: int = 8,
    iters: int = 40,
    verify: bool = False,
) -> dict:
    """Time one (nprog, block, tile) contiguous-copy config on the NPU.

    Returns aggregate bandwidth = (read+write bytes) / median_time.
    """
    assert block % tile == 0, f"block {block} not divisible by tile {tile}"
    n = nprog * block
    x = _det_input(n, 4099, "npu")
    out = torch.empty_like(x)

    def call():
        stream_copy[(nprog,)](x, out, BLOCK=block, TILE=tile)

    for _ in range(warmup):
        call()
    torch.npu.synchronize()

    samples_us = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        call()
        torch.npu.synchronize()
        samples_us.append((time.perf_counter() - t0) * 1e6)

    med = statistics.median(samples_us)
    bytes_moved = 2 * n * 4  # one read + one write, f32
    agg_gbps = bytes_moved / med / 1e3  # bytes/µs / 1e3 = GB/s

    verified = None
    if verify:
        # A pure copy must reproduce the source tensor bit-for-bit.  Compare
        # against x directly (not a host-regenerated reference): the input's
        # `/modulus` division rounds differently on NPU vs CPU, so a host
        # reference would mismatch by ~1 ulp even for a correct copy.
        verified = bool(torch.equal(out, x))

    return {
        "nprog": nprog,
        "block": block,
        "tile": tile,
        "packet_bytes": tile * 4,
        "N": n,
        "bytes_moved": bytes_moved,
        "median_us": med,
        "min_us": min(samples_us),
        "agg_gbps": agg_gbps,
        "verified": verified,
    }


if __name__ == "__main__":
    r = run_config(nprog=48, block=262144, tile=8192, verify=True)
    print(
        f"stream_copy OK nprog={r['nprog']} block={r['block']} tile={r['tile']} "
        f"median_us={r['median_us']:.1f} agg_gbps={r['agg_gbps']:.1f} "
        f"verified={r['verified']}"
    )
