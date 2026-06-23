"""Group-V gap-seeded kernel: avoidable serialization (Gap-3) — for US-SB-006/008.

Two fully INDEPENDENT streams run inside each program instance:

  Stream A (MTE/HBM-bound): load a large tile, trivially scale it, store it.
  Stream B (Vector-bound):  load a small tile, run a long FMA chain, store it.

A and B share no data (separate inputs, separate outputs), so an ordering
barrier between them is *provably redundant*. We deliberately SEED one via
``tl.debug_barrier()`` (gated on SEED_BARRIER). bishengir lowers that to a
``hivm.hir.pipe_barrier`` that fully drains A's MTE pipe before B's Vector pipe
starts — destroying the MTE↔Vector overlap that would otherwise hide one stream
behind the other.

The counterfactual (US-SB-006 / US-SB-008) removes exactly that barrier from the
compiled NPUIR via ``tritonsim-hivm --remove-pipe-barrier-index=K
--edited-npuir-file=...``. Because the streams are independent, the edited kernel
is bitwise-equivalent; because the barrier sat between two different pipes on the
critical path, removal yields a measurable speedup whose size the model's Gap-3 /
two-limit ``compiler_headroom`` predicts.

Sizes / iteration counts are env-tunable so the seeded gap can be driven well
above hardware measurement noise on the box.
"""
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl


# Tunables (env-overridable for gap-magnitude calibration on 910B3).
N_A = int(os.environ.get("SEED_N_A", str(8 * 1024 * 1024)))   # MTE-stream elements (big copy)
N_B = int(os.environ.get("SEED_N_B", str(64 * 1024)))         # Vector-stream elements (small, resident)
BLOCK_A = int(os.environ.get("SEED_BLOCK_A", "2048"))
BLOCK_B = int(os.environ.get("SEED_BLOCK_B", "2048"))
NITER = int(os.environ.get("SEED_NITER", "512"))              # FMA-chain depth (Vector work)
SEED_BARRIER = int(os.environ.get("SEED_BARRIER", "1"))       # 1 = seed avoidable serialization
_FMA_MUL = 1.0001
_FMA_ADD = 0.0001


@triton.jit
def seeded_serial_kernel(
    xa_ptr, outa_ptr, xb_ptr, outb_ptr,
    n_a, n_b,
    SEED_BARRIER: tl.constexpr,
    BLOCK_A: tl.constexpr, BLOCK_B: tl.constexpr,
    NITER: tl.constexpr,
    FMA_MUL: tl.constexpr, FMA_ADD: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # ── Stream A: MTE/HBM-bound copy ─────────────────────────────────────
    off_a = pid * BLOCK_A + tl.arange(0, BLOCK_A)
    mask_a = off_a < n_a
    a = tl.load(xa_ptr + off_a, mask=mask_a, other=0.0)
    tl.store(outa_ptr + off_a, a, mask=mask_a)

    # ── Seeded avoidable serialization (provably redundant: A ⟂ B) ───────
    if SEED_BARRIER:
        tl.debug_barrier()

    # ── Stream B: Vector-bound FMA chain ─────────────────────────────────
    off_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = off_b < n_b
    b = tl.load(xb_ptr + off_b, mask=mask_b, other=0.0)
    for _ in range(NITER):
        b = b * FMA_MUL + FMA_ADD
    tl.store(outb_ptr + off_b, b, mask=mask_b)


def _grid():
    # One grid covers both streams; each program does its A-tile and B-tile.
    return (max(triton.cdiv(N_A, BLOCK_A), triton.cdiv(N_B, BLOCK_B)),)


def _det_input(n, modulus, device, dtype=torch.float32):
    """Deterministic, CPU-reconstructable input in [0, 1).

    Avoids RNG so the CPU ``reference`` reproduces the device inputs exactly
    (NPU torch.randn != CPU torch.randn for the same seed), making the
    counterfactual output-equivalence check unambiguous.
    """
    idx = torch.arange(n, device=device, dtype=torch.int64)
    return ((idx % modulus).to(dtype) / float(modulus))


def build_inputs():
    device = "npu"
    xa = _det_input(N_A, 4099, device)
    xb = _det_input(N_B, 4093, device)
    outa = torch.empty_like(xa)
    outb = torch.empty_like(xb)
    return {"xa": xa, "xb": xb, "outa": outa, "outb": outb}


class Model(nn.Module):
    def forward(self, data):
        xa, xb = data["xa"], data["xb"]
        outa, outb = data["outa"], data["outb"]
        seeded_serial_kernel[_grid()](
            xa, outa, xb, outb,
            N_A, N_B,
            SEED_BARRIER=SEED_BARRIER,
            BLOCK_A=BLOCK_A, BLOCK_B=BLOCK_B,
            NITER=NITER,
            FMA_MUL=_FMA_MUL, FMA_ADD=_FMA_ADD,
        )
        # outb (the post-barrier Vector stream) is returned FIRST so the
        # launcher's kernel_output.npy alias verifies the stream whose
        # scheduling the seeded barrier actually gates.
        return (outb, outa)


def reference(xa, xb):
    """CPU/torch reference for correctness checks (run_counterfactual reference_fn).

    Equivalent regardless of SEED_BARRIER — the barrier only affects scheduling.
    """
    outa = xa.clone()
    b = xb.clone()
    for _ in range(NITER):
        b = b * _FMA_MUL + _FMA_ADD
    # Match Model.forward output order: (outb, outa).
    return b, outa


def reference_cpu():
    """Rebuild the deterministic inputs on CPU and return (outb, outa).

    Usable without a device — used by the counterfactual reference_fn and the
    output-equivalence check.
    """
    xa = _det_input(N_A, 4099, "cpu")
    xb = _det_input(N_B, 4093, "cpu")
    return reference(xa, xb)


if __name__ == "__main__":
    data = build_inputs()
    outs = Model().forward(data)
    print("seeded_serial launch OK",
          [tuple(o.shape) for o in outs],
          "SEED_BARRIER=", SEED_BARRIER, "NITER=", NITER)
