# Real 910B3 Hardware Runs — Caveat Closure (2026-06-10)

Closes the "hardware-gated, never run on real hardware" caveats flagged on
A.5/A.6/A.7. All runs executed on the remote **910B3** (ssh host `910B3`,
8× 910B3 NPUs, CANN **9.0.0 release**, conda env `triton_hxl`, triton 3.2.0,
aarch64) via `scripts/remote_bench.py` wiring.

## 1. bishengir chunk_kda compile — RESOLVED (was the A.6 Gap-1 blocker)

The blocker was: `bishengir-compile` crashes in `ConvertLinalgRToBinary`
(SmallVector assertion) compiling chunk_kda on **CANN 9.0.0-beta.2**.

**Result:** on **CANN 9.0.0 release**, chunk_kda compiles and runs cleanly:

```
$ python test/chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2.py
✅ Kernel launched successfully
torch.Size([1, 8192, 32, 128]) torch.float32   # dq
torch.Size([1, 8192, 32, 128]) torch.float32   # dk
torch.Size([1, 8192, 32, 128]) torch.bfloat16  # dv2
torch.Size([32, 1, 8192]) torch.float32         # db
torch.Size([1, 8192, 32, 128]) torch.float32   # dg
torch.Size([32, 1, 8192, 64]) torch.float32     # dA
```

The crash was a third-party compiler bug in the beta, fixed in the release.
The "dump-before-codegen" spike (`TestDumpBeforeCodegen`) is therefore moot:
codegen completes, no crash to survive. `test_chunk_kda_milestone.py` xfail
reasons updated to reflect this (local xfail now only because WSL has no NPU).

## 2. Real T_measured + initial naive-floor residual — POPULATED

chunk_kda profiled under `msprof` (10 iters → 6 recorded MIX_AIC invocations).
Raw CSV: `chunk_kda_op_summary.csv` (also fixtured at
`tests/perfbound/fixtures/chunk_kda_op_summary_910b3.csv`).

| Quantity | Value | Source |
|----------|-------|--------|
| T_measured | **104,326 µs** (~104 ms/launch) | msprof op_summary, 6× MIX_AIC rows ~104.3 ms each |
| T_bound (HBM floor) | 1,386 µs | analytic: 2.218e9 B / 1.6 TB/s |
| T_bound (compute floor) | 148 µs | analytic: 4.724e10 FLOP / 320 TFLOP/s bf16 |
| binding (predicted) | MTE_GM (memory) | max(floor) = HBM |
| Soundness status | **PASS** (T_bound ≤ T_measured) | `validate_from_csv` |
| Tightness | **75.3×** | T_measured / T_bound |
| naive-floor residual | **102,940 µs** | T_measured − naive 1,386 µs HBM floor |
| component_match | **False** | predicted `mte`, measured `aicore`-dominant |

### Findings
- This was the initial naive-floor residual, not an attainable-headroom
  estimate and not the current two-limit result.
- **The naive HBM-floor bound is sound but very loose (75×)** for chunk_kda,
  and **mispredicts the binding component**: it predicts memory-bound, but the
  kernel measures as AI compute-core dominated. chunk_kda's dots use tiny
  tiles (64×32×32), so it is issue/compute bound, not HBM bound — motivating
  the tighter component/two-limit analysis over the single grid floor.

## 3. Parser bug exposed by real data — FIXED

Real chunk_kda rows have Task Type **`MIX_AIC`** (mixed cube). The timing
filter `parse_kernel_time_us` only matched `AI_CORE`/`AICORE`, so it dropped
every kernel row → `ValueError` → EXECUTION_ERROR. Meanwhile
`parse_component_durations` already mapped `MIX_AIC`. Fixed by a shared
`_is_aicore_task()` helper recognising `AI_CORE`, `AICORE`, `MIX_AIC`,
`MIX_AIV`, `AI_VECTOR_CORE`, `AIV` — used by both functions so they cannot
drift again. Guarded by `tests/perfbound/test_chunk_kda_hw_validation.py`.

## 4. remote_bench wiring fixed to the real machine

`scripts/remote_bench.py` previously hardcoded CANN `/usr/local/Ascend/cann/
set_env.sh` + conda `tlx` and emitted literal `{{ }}` (invalid bash, would
break the `||` fallbacks on first real run), plus an unrecognised `msprof
--version` preflight and a non-existent default bishengir path. Now: CANN
`ascend-toolkit/set_env.sh` + conda `triton_hxl` (both env-overridable),
single-brace shell groups, `command -v msprof` preflight, `bishengir-compile`
on PATH. Config in `~/.vtriton_remote` (host `910B3`).

## 5. Gap 4 (repeat/mask) — IR-stage investigation + analytical derivation (2026-06-11)

**Question (US-SB-004 / Task 4b):** where do per-op CCE `repeat`/`mask` first
appear in the lowering pipeline, so the model can read them?

**Finding — they are NOT in any parseable MLIR stage we can reach:**
- `grep repeat` on the dumped `chunk_kda_kernel_clean.npuir.mlir`
  (post-`GraphSyncSolver` `hivm.hir`) → **0 hits**; the only `mask` token is
  `set_mask_norm`. Same for `test/hivm_add_kernel.npuir.mlir` and
  `hivm_mixed_cv_kernel.npuir.mlir` → repeat/mask are absent at the npuir level
  for **every** kernel, not just chunk_kda.
- They materialize only in later bishengir CCE/binary codegen. Attempting to
  dump that via `bishengir-compile --mlir-print-ir-after-all
  --mlir-print-ir-tree-dir` **times out (>600 s, even plain compile)** and the
  per-pass tree never flushes — not a viable extraction path.

**Resolution — derive `repeat` analytically (no device / no bishengir needed):**
the iteration count is a function of data already in the IR. The C++ emitter
(`HIVMAnalysis.cpp`, after element/byte inference) sets
`repeat = ceil(elements / (2048 / bitsPerElem))` — a 256-byte vector register
processes `2048/bits` elements per iteration (matches `AscendModelOps.cpp`).
On the real `kda_des.json` this fills **271 / 1378 ops with repeat>1** (8/16/32/64).

**Gap-4 model = per-instruction issue overhead** (`bound_combiner._compute_gap4`):
an op whose `repeat` already packs its iterations into the minimum number of
instructions has **0** avoidable overhead; a suboptimally-low repeat (the
paper's AvgPool `repeat=1` case) issues many instructions, each paying the
fixed startup latency. Bounded `inefficiency ∈ (0,1)` keeps the bound sound.

**Result on chunk_kda:** real Tier-2 bound `t_bound = 46,110 µs ≤ t_measured =
104,289 µs` (**2.26×** — vs the 75× HBM grid floor); Gap-4 is non-zero
(~2.6% of measured) and now the dominant attributed gap. `mask` is left 0 (the
per-instruction model does not use it; a lane-fill `mask` model is a future
refinement). The v1 database has no measured vector/cube startup latency, so
the current Gap-4 value uses explicitly reported diagnostic defaults.

## 6. HIVM IR optimization feasibility analysis (2026-06-11)

Post-closure of US-SB-008, analysis of whether HIVM IR can be directly edited
or hinted for performance optimization. The two-limit result for chunk_kda:

```
T_bound_HIVM = 46,065 µs  (idealized — Gap-1/3 relaxed)
T_bound_DSL  = 46,110 µs  (realized bishengir structure)
T_measured   = 104,289 µs (post-warmup median of 5 launches on 910B3 NPU:1)

Compiler headroom = T_bound_DSL − T_bound_HIVM = 44.99 µs  (0.04%)
Author residual   = T_measured  − T_bound_DSL  = 58,179 µs (55.8%)
```

The author residual is not a predicted attainable win. Without a
correctness-verified counterfactual, the pipeline reports only a low-confidence
diagnostic range from 0 to the residual.

### Intervention levels

| Level | Gap owned | Feasibility | Max win |
|-------|-----------|-------------|---------|
| HIVM IR (des.json) editing | Gap-1/3 | Model only — bishengir-compile cannot accept des.json | ~45 µs |
| bishengir MLIR hints (ascend-tiling-opt) | Gap-4 (2.6%) | Actionable — modify MLIR, standard compile | ~2.7 ms |
| Triton kernel rewrite (author level) | Author residual (55.8%) | Highest-impact search space — tiling, fusion, layout | ≤58 ms diagnostic cap |

### Gap attribution guide for hints

| Gap | Attribution (chunk_kda) | What to hint |
|-----|------------------------|-------------|
| Gap-1 (placement) | 0.16% | Memory level assignment for intermediates |
| Gap-2 (coalescing) | 0% | Already optimal for chunk_kda |
| Gap-3 (serialization) | 0% | Pipeline overlap structure |
| Gap-4 (intra-unit exec) | **2.6%** | Instruction scheduling, repeat count, vector width |

### Conclusion

The compiler-side structural residual is small for chunk_kda. The remaining
55.8% is an author-level residual, not a demonstrated performance opportunity.
Tiling strategy, op fusion, and memory access pattern are the primary
counterfactual search space. `ascend-tiling-opt` can rank candidates with the
white-box model, but only correctness-verified hardware measurements can
promote a diagnostic cap to attainable headroom. Direct HIVM IR editing has a
small modeled effect (~0.04%) and lacks a back-compilation path to hardware.

## 7. Multi-kernel validation set — US-SB-005 (2026-06-11)

Profiled on 910B3 via `msprof` + `kernel_launcher.py`, correctness verified
against torch references / mathematical invariants. The 5th DISTINCT kernel
(rmsnorm) was added 2026-06-11 to remove the vector_add double-count critique.

### Validation set summary (6 entries / 5 distinct kernels)

| Kernel | Bound kind | T_bound (µs) | T_measured (µs) | Tightness | Status |
|--------|-----------|-------------|-----------------|-----------|--------|
| chunk_kda | tier2_des | 46,109.91 | 104,326.00 | 2.26× | PASS |
| vector_add_16m | analytic_hbm_floor | 125.83 | 307.03 | 2.44× | PASS |
| vector_add_32m | analytic_hbm_floor | 251.66 | 605.25 | 2.41× | PASS |
| softmax_8kx2k | analytic_hbm_floor | 88.01 | 305.07 | 3.47× | PASS |
| layernorm_8kx2k | analytic_hbm_floor | 88.01 | 608.39 | 6.91× | PASS |
| rmsnorm_8kx2k | analytic_hbm_floor | 88.01 | 702.86 | 7.99× | PASS |

**soundness_rate = 1.0** (no BOUND_VIOLATION across all 6 entries). Five
distinct kernel families: chunk_kda, vector_add, softmax, layernorm, rmsnorm.
Tier-2 DES coverage remains chunk_kda-only; the vector kernels use the
conservative analytic HBM floor (a sound lower bound). The broader >=30-kernel
Part-B campaign with more Tier-2 bounds is tracked in
`stage_b_remainder_handoff.md`.

**rmsnorm_8kx2k**: RMSNorm forward (8192×2048, fp32). Mean-of-squares
reduction + rsqrt scale + weight (lighter reduction than layernorm; no mean
subtraction). AI_VECTOR_CORE; 11 msprof invocations; T_measured =
median(post-warmup) = 702.86 µs; HBM floor = 88.01 µs. Correctness verified vs
torch reference: max_abs_err=1.9e-6, rel=1.1e-7, finite.

**msprof path note (2026-06-11):** the triton/`kernel_launcher.py` + `msprof`
path produces a valid `op_summary` CSV (verified live for rmsnorm). The
separate AscendC `bench_launcher` path currently yields no `op_summary` on this
box (see §8) — that regression does not affect the triton-kernel measurements
in this table.

### Kernel details

**softmax_8kx2k**: Row-wise softmax (8192×2048, fp32). Pure AI_VECTOR_CORE
kernel: `aiv_scalar_ratio=0.45`, `aiv_vec_ratio=0.25`, `aiv_mte2_ratio=0.24`.
11 msprof invocations captured; T_measured = median(10 post-warmup) = 305.07 µs.
Correctness: all row sums = 1.0, all values positive and finite.

**layernorm_8kx2k**: LayerNorm forward (8192×2048, fp32). Pure AI_VECTOR_CORE
kernel: `aiv_scalar_ratio=0.52`, `aiv_vec_ratio=0.18`, `aiv_mte2_ratio=0.29`.
11 invocations; T_measured = 608.39 µs. Correctness: output finite, non-trivial
per-row variance (1.74–2.21).

### Kernel diversity

The set spans both compute-bound (chunk_kda, Tier-2 DES) and memory-bound
(analytic HBM floor) kernels, with tightness ranging from 2.26× to 6.91×.
All soundness checks pass — the model never predicts a bound above the
measured wall-clock time.

## 8. Scalar throughput (US-SB-007) — CLOSED with direct CCE measurement (2026-06-16)

**Goal:** replace the derived `P_scalar` (= P_vector/128) with a directly
measured value from a CCE (AscendC, per project convention) microbench on
910B3, as US-SB-007 requires.

**What was built (committed):**
- `perfbound/calibration/microbench/scalar_peak.cce` + `ScalarPeakKernel` in
  `vt_microbench_common.h`: an AIV kernel running a dependent scalar FMA chain
  (`acc = acc*c1 + c2`, 1e6 iterations) that the compiler cannot vectorise, so
  the work is forced onto the scalar issue path. Wired into `bench_launcher.cpp`
  (`--kernel scalar_peak`) and the CMake glob.
- It **compiles and runs** on 910B3 (`vt_a1_bench_launcher --kernel scalar_peak`
  returns rc=0; the `scalar_peak` symbol is linked into the launcher).

**Measurement:** reran the CCE path with the explicit CANN 8.2 profiler binary
that fixed Stage-A op-summary collection:

```bash
python3 perfbound/calibration/scripts/cce_remote_bench.py \
  --host 910B3 --direct-ssh \
  --remote-workdir /tmp/vtriton_scalar_cce \
  --skip-sync --skip-direct-compile \
  --n-repeat 45 --kernel-timeout-sec 300 \
  --soc-version Ascend910B1 \
  --cann-env /usr/local/Ascend/ascend-toolkit/8.2.RC1.alpha003/set_env.sh \
  --cann-package-path /usr/local/Ascend/ascend-toolkit/8.2.RC1.alpha003 \
  --msprof /usr/local/Ascend/ascend-toolkit/8.2.RC1.alpha003/tools/profiler/bin/msprof \
  --kernels scalar_peak \
  --output-dir /tmp/vtriton_scalar_calib
```

The runner built the launcher, executed 45 `scalar_peak` invocations, and synced
`scalar_peak.csv`. The fitter uses `aiv_scalar_time(us)` for the dependent
scalar FMA chain and discards the first 15 rows chronologically as warmup.

| Constant | Measured value | 95% CI | Steady samples | CV |
|----------|----------------|--------|----------------|----|
| `P_scalar_add_sustained` | **0.5998 GFLOPS** | 0.000019 GFLOPS | 30 | 0.009% |

**Resolution:** US-SB-007 is now closed. `calib_910b3_v1.json` promotes
`P_scalar_add_sustained` with `source="cce_microbench"` and sets
`vector.scalar_throughput_measured=true`, so scalar-bound floors now use the
measured scalar rate instead of the prior Vector-rate fallback.

## 9. US-SB-006 (live gap counterfactual) + US-SB-008 (two-limit HW reachability) — attempted, BLOCKED (re-audited 2026-06-16)

Both stories need a *measurable* gap removed on real hardware. Every concrete
path was tried; each hits a real blocker on the one kernel with a full
NPUIR→des.json pipeline (chunk_kda).

**Executable Gap-3 edit is not present in the current checkout, and would be
vacuous for chunk_kda anyway.** A prior note referenced
`tritonsim-hivm --remove-pipe-barrier-index=N --edited-npuir-file=...`, but the
current `tritonsim-hivm --help` output exposes no such option and the current
source tree has no matching implementation. Even if restored, the model's
predicted bound delta for chunk_kda is **0.0 µs** — chunk_kda's Gap-3
attribution is effectively zero, so removing a barrier does not move the
binding component floor. Nothing measurable to validate on this kernel.

**Gap-4 (`raise_repeat`) moves the attribution but not the bound.** After
fixing `raise_repeat` to match the real des.json pipe tokens (`PIPE_M/PIPE_V/
PIPE_S` — it previously only matched the synthetic `Cube/Vector/Scalar` names
and silently no-op'd on every real kernel), doubling repeat on chunk_kda drops
Gap-4 from 2.569%→1.555% (the model *does* quantify the change), but the bound
stays 46,109.91 µs: Gap-4 is an attribution fraction, not on chunk_kda's binding
core-compute path. Again no measurable bound delta.

**The DES-JSON edit → bishengir → hardware path is structurally invalid.**
`raise_repeat`/`insert_pingpong`/`merge_transfers` mutate **DES-graph JSON**,
which is analytical-only; `recompile_remote` would feed it to
`bishengir-compile`, which accepts MLIR, not des.json. The only executable edit
(`--remove-pipe-barrier-index`) is the pipe-barrier eraser above, and it lands
on chunk_kda's zero-Gap-3.

**msprof op-collection is currently broken for the AscendC-launcher path**
(§8); the triton/`kernel_launcher.py` path works (used for §7), but there is no
proven NPUIR→bishengir-binary→aclrtLaunch path to run an *edited* chunk_kda
npuir on device and fetch its output for equivalence checking.

**Fallback (simpler kernel with large, measurable compiler headroom) not
completed:** it requires a *second* gap-seeded kernel carried through the full
NPUIR→des.json pipeline plus a hardware npuir-recompile-and-launch harness that
is not proven end-to-end in this tree. Documented as the next concrete step.

**Accepted-evidence fixture:** `.omc/research/hw_runs/counterfactual_gap_results.json`
records the accepted counterfactual contract, all attempted paths, and explicitly
keeps `satisfies_us_sb_006=false` / `satisfies_us_sb_008=false`. The vector-add
2× work-scaling run remains a useful sanity check (`quantification_error=2.95%`,
outputs verified) but is intentionally excluded because it changes problem size
rather than removing a seeded Gap-1/2/3/4 cause from an equivalent kernel.

**Honest outcome:** US-SB-006 and US-SB-008 remain `passes:false`. The
analytical two-limit ordering (T_bound_HIVM ≤ T_bound_DSL ≤ T_measured) and the
model's gap-quantification mechanism are validated on real chunk_kda data, but
chunk_kda's compiler headroom is below measurement noise, no current
compiler-reachable edit exists in this checkout, and no alternative
large-headroom kernel was carried to a hardware measurement. The `raise_repeat`
real-pipe fix is kept as a genuine bug fix for analytical DES edits, not as a
hardware counterfactual closure.

## Reproduce

```bash
# Regenerate kda_des.json with analytical repeat (local, no device):
build/bin/tritonsim-hivm \
  --npuir-file=.omc/research/hw_runs/chunk_kda_kernel_clean.npuir.mlir \
  --des-graph-file=.omc/research/hw_runs/kda_des.json \
  --hardware-config=configs/ascend_910b.json --scheduler=static

# config: ~/.vtriton_remote -> [remote] host=910B3 path=/root/vTriton
rsync -az --exclude=.git --exclude=build --exclude=thirdparty ./ 910B3:/root/vTriton/
ssh 910B3 'source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
  conda activate triton_hxl && \
  msprof --application="python /root/vTriton/scripts/kernel_launcher.py \
    --kernel /root/vTriton/test/chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2.py \
    --output-dir /root/vTriton/kernel_outputs --iters 10" \
    --output=/root/vTriton/msprof_chunk_kda'
# fetch op_summary_*.csv, then:
pytest tests/perfbound/test_chunk_kda_hw_validation.py -q
```

## 10. Stage A P0 calibration closure (2026-06-15)

The final two P0 CCE microbenchmarks were reviewed, corrected, compiled, and
profiled on 910B3:

| Constant | Measured value | 95% CI | Steady samples | CV |
|----------|----------------|--------|----------------|----|
| `BW_l0c_to_gm_sustained` | **143.523 GB/s** | 1.141 GB/s | 30 | 2.2% |
| `BW_hbm_allcore_sustained` | **77.043 GB/s/core** | 0.027 GB/s/core | 30 | 0.1% |

The FixPipe benchmark initializes one valid L0C tile with MMAD, synchronizes
the L0C producer/consumer through the CO1 queue, then profiles 1280 repeated
FixPipe transfers using `aic_fixpipe_time(us)`. The all-core benchmark launches
with block dim 20 and assigns each core a disjoint 16 MiB region; the aggregate
320 MiB footprint exceeds the 192 MiB L2 capacity.

The CANN 9.0 `msprof` selected by PATH still emitted no AscendC op summary.
Using the explicit CANN 8.2 profiler binary produced 45 rows for each kernel.
The fitter discards the first 15 rows chronologically as warmup and computes
statistics over the final 30; it does not select fastest samples. Therefore the
broad profiler caveat in sections 8 and 9 does not apply to these two
measurements. The calibration database now reports 18 measured constants,
maximum relative 95% CI 1.07%, and no P0 violations.

Running the complete Stage A report with this promoted database gives:

```text
T_bound_HIVM = 46,064.92 us
T_bound_DSL  = 46,109.91 us
T_measured   = 104,288.94 us
binding      = vector
compiler headroom = 44.99 us
author residual   = 58,179.04 us
```

Profile diagnosis is `Insufficient Parallelism`, with scalar residency as the
measured dominant component. Until a correctness-verified counterfactual is
measured, realistic performance headroom remains an explicitly low-confidence
diagnostic range of **0..58,179.04 us**, not a point estimate.
