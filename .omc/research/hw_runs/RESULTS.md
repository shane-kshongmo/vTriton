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

## 2. Real T_measured + author_headroom — POPULATED (was always None)

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
| author_headroom | **102,940 µs** | T_measured − T_bound_DSL (`TwoLimitResult`) |
| component_match | **False** | predicted `mte`, measured `aicore`-dominant |

### Findings
- **author_headroom is now a real number** (102.9 ms), closing the A.7/M6
  caveat that it was structurally present but never populated.
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
104,326 µs` (**2.26×** — vs the 75× HBM grid floor); Gap-4 is non-zero
(~2.6% of measured) and now the dominant attributed gap. `mask` is left 0 (the
per-instruction model does not use it; a lane-fill `mask` model is a future
refinement).

## 6. HIVM IR optimization feasibility analysis (2026-06-11)

Post-closure of US-SB-008, analysis of whether HIVM IR can be directly edited
or hinted for performance optimization. The two-limit result for chunk_kda:

```
T_bound_HIVM = 46,065 µs  (idealized — Gap-1/3 relaxed)
T_bound_DSL  = 46,110 µs  (realized bishengir structure)
T_measured   = 104,326 µs (actual on 910B3 NPU:1)

Compiler headroom = T_bound_DSL − T_bound_HIVM = 44.99 µs  (0.04%)
Author headroom   = T_measured  − T_bound_DSL  = 58,216 µs (55.8%)
```

### Intervention levels

| Level | Gap owned | Feasibility | Max win |
|-------|-----------|-------------|---------|
| HIVM IR (des.json) editing | Gap-1/3 | Model only — bishengir-compile cannot accept des.json | ~45 µs |
| bishengir MLIR hints (ascend-tiling-opt) | Gap-4 (2.6%) | Actionable — modify MLIR, standard compile | ~2.7 ms |
| Triton kernel rewrite (author level) | Author headroom (55.8%) | Highest impact — tiling, fusion, layout | ~58 ms |

### Gap attribution guide for hints

| Gap | Attribution (chunk_kda) | What to hint |
|-----|------------------------|-------------|
| Gap-1 (placement) | 0.16% | Memory level assignment for intermediates |
| Gap-2 (coalescing) | 0% | Already optimal for chunk_kda |
| Gap-3 (serialization) | 0% | Pipeline overlap structure |
| Gap-4 (intra-unit exec) | **2.6%** | Instruction scheduling, repeat count, vector width |

### Conclusion

The compiler is near-optimal for chunk_kda. The 55.8% performance opportunity
is at the Triton kernel author level — tiling strategy, op fusion, and memory
access pattern. `ascend-tiling-opt` is the tool to search this space using the
white-box model. Direct HIVM IR editing provides marginal improvement (~0.04%)
and lacks a back-compilation path to hardware.

## 7. Multi-kernel validation set — US-SB-005 complete (2026-06-11)

Added two vector kernels (softmax, layernorm) to reach the n≥5 target. Both
profiled on 910B3 via `msprof` + `kernel_launcher.py`, correctness verified
via mathematical invariants.

### Validation set summary (5/5 kernels)

| Kernel | Bound kind | T_bound (µs) | T_measured (µs) | Tightness | Status |
|--------|-----------|-------------|-----------------|-----------|--------|
| chunk_kda | tier2_des | 46,109.91 | 104,326.00 | 2.26× | PASS |
| vector_add_16m | analytic_hbm_floor | 125.83 | 307.03 | 2.44× | PASS |
| vector_add_32m | analytic_hbm_floor | 251.66 | 605.25 | 2.41× | PASS |
| softmax_8kx2k | analytic_hbm_floor | 88.01 | 305.07 | 3.47× | PASS |
| layernorm_8kx2k | analytic_hbm_floor | 88.01 | 608.39 | 6.91× | PASS |

**soundness_rate = 1.0** (no BOUND_VIOLATION across all 5 kernels).

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
