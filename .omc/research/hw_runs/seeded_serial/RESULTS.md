# Seeded Serial Two-Limit Hardware Result

US-SB-008 is closed by a hand-edited compiler-IR realization of the seeded
barrier removal. The standalone edited NPUIR object still does not launch
through the current Triton path, but the equivalent TTAdapter compiler IR edit
does: source stays `SEED_BARRIER=1`, Triton is pointed at a hand-edited
TTAdapter file with exactly one `gpu.barrier` removed, and the normal
Triton-Ascend backend compiles and launches the resulting `npubin`.

The kernel has two independent streams: an MTE/HBM copy stream and a Vector FMA
stream. `SEED_BARRIER=1` inserts a redundant `tl.debug_barrier()` between them;
the barrier-free HIVM limit removes the four resulting `hivm.hir.pipe_barrier`
ops from the compiled NPUIR.

## Evidence

- ON DES: `seeded_on_des.json`
- OFF DES: `seeded_off_des.json`
- ON TTAdapter IR: `seeded_on_ttadapter.mlir`
- ON timing CSV: `seeded_on_op_summary.csv`
- OFF TTAdapter timing CSV: `seeded_off_ttadapter_op_summary.csv`
- OFF TTAdapter IR: `seeded_off_ttadapter.mlir`
- OFF source timing CSV: `seeded_off_src_op_summary.csv`
- Held-out calibration CSVs: `seeded_calib_a4m_on_op_summary.csv`,
  `seeded_calib_a4m_off_src_op_summary.csv`
- Outputs: `seeded_on_outa.npy`, `seeded_on_outb.npy`,
  `seeded_off_ttadapter_outa.npy`, `seeded_off_ttadapter_outb.npy`,
  `seeded_off_src_outa.npy`, `seeded_off_src_outb.npy`
- Machine: 910B3, `warmup=8`, `iters=40`

## Result

| Quantity | Value |
|---|---:|
| `T_bound_HIVM` | 1819.070 us |
| `T_bound_DSL` | 1840.007 us |
| `T_measured` (ON median) | 3289.604 us |
| OFF TTAdapter median | 3268.744 us |
| OFF source median | 3268.054 us |
| Predicted headroom | 20.937 us |
| Measured delta (TTAdapter) | 20.860 us |
| Quantification error | 0.368% |

Ordering holds:

```text
T_bound_HIVM = 1819.070 us
  <= T_bound_DSL = 1840.007 us
  <= T_measured = 3289.604 us
```

Both outputs are bitwise identical between ON and hand-edited TTAdapter OFF:

```text
outa: shape=[8388608], dtype=float32, equal=true, max_abs=0.0
outb: shape=[65536], dtype=float32, equal=true, max_abs=0.0
```

## Held-Out Calibration Note

The first local gate using raw DES `pipe_barrier.duration * loop_multiplier`
over-predicted headroom by about 21x. The current result uses a calibrated
`pipe_barrier_cycles_per_iter=0.18451333295482839` in
`calib_910b3_v1.json`, but the calibration is held out from the reported 8M
validation. It is derived from a smaller `SEED_N_A=4194304` run:

```text
10.54 us * 1850 cycles/us / (ceil(2048/20) * 1026 barrier loop-units)
```

The 8M validation then predicts 20.937 us against a measured 20.860 us
TTAdapter-edit delta, for 0.368% error. The source-level `SEED_BARRIER=0`
cross-check measures 21.550 us.

## Compiler-Reachability Note

`tritonsim-hivm --remove-pipe-barrier-index=0` was applied repeatedly until all
four barriers were removed, producing `seeded_off_clean.npuir.mlir` and
`seeded_off_des.json`.

For the hardware realization, the ON TTAdapter dump contains exactly one
`gpu.barrier`; the hand-edited TTAdapter removes that line:

```diff
-    gpu.barrier loc(#loc10)
```

The run uses `TRITON_KERNEL_OVERRIDE=1` and
`TRITON_ACCEPT_TTADAPTER_OVERRIDE=1` so Triton compiles the edited TTAdapter
through its normal backend path. The launcher log includes:

```text
Overriding kernel with file .../override_seed_hand_ttadapter/.../seeded_serial_kernel.ttadapter
```

On the remote 910B3, `bishengir-compile --enable-hivm-compile=true` accepts the
edited NPUIR and emits an ELF device object. Replacing Triton's cached `npubin`
with that object makes the launcher load the object, but the kernel fails during
device-to-host output copy. A manually recompiled barrier-ON NPUIR object fails
the same cache-replacement launch test, so the failure is in the standalone
NPUIR compile/launch path, not evidence that barrier removal is semantically
wrong.
