# CCE Microbenchmarks for Ascend 910B3 Calibration

This directory contains CCE microbenchmark kernels used to measure sustained hardware rates for the two-tier analytical performance bound model.

**Purpose**: Populate `perfbound/calibration/data/calib_910b3_v1.json` with measured hardware constants (I_c values).

**Source spec**: `.omc/specs/performance_bound_model.md` §A.1

**Related**: `scripts/run_benchmarks.sh` (orchestration), `scripts/fit_constants.py` (extraction)

---

## Kernel List

### Compute (Cube)

| Kernel | Purpose | Template | Expected Metric |
|--------|---------|----------|-----------------|
| `cube_peak_fp16.cce` | Cube FP16 sustained throughput | M=128, N=128, K=4096, 30× repeat | P_cube[fp16] ≈ 280 TFLOPS |
| `cube_peak_int8.cce` | Cube INT8 sustained throughput | M=128, N=128, K=4096, 30× repeat | P_cube[int8] ≈ 560 TFLOPS |
| `cube_peak_bf16.cce` | Cube BF16 sustained throughput | M=128, N=128, K=4096, 30× repeat | P_cube[bf16] ≈ 280 TFLOPS |

### Compute (Vector)

| Kernel | Purpose | Template | Expected Metric |
|--------|---------|----------|-----------------|
| `vector_peak_elemwise_{add,mul,max,min}.cce` | Vector elementwise throughput per op | 256 elements, 10000× repeat | P_vector[op] ≈ 18 TFLOPS FP16 |
| `vector_peak_transcendental.cce` | Vector transcendental throughput (exp/log/sqrt/rsqrt) | 256 elements, 10000× repeat | P_vector[op] varies |

### Memory Transfer (MTE)

| Kernel | Purpose | Template | Expected Metric |
|--------|---------|----------|-----------------|
| `mte_gm_to_ub.cce` | GM→UB bandwidth (Vector load path) | 256 KiB transfer, 2048× repeat | BW[gm→ub] sustained |
| `mte_ub_to_gm.cce` | UB→GM bandwidth (Vector store path) | 256 KiB transfer, 2048× repeat | BW[ub→gm] sustained |
| `mte_gm_to_l1.cce` | GM→L1 bandwidth (Cube load path) | 256 KiB transfer, 2048× repeat | BW[gm→l1] sustained |
| `mte_l1_to_l0a.cce` | L1→L0A bandwidth (Cube input staging) | 256 KiB staged through 16 KiB L0A tiles, 2048× repeat | BW[l1→l0a] sustained |
| `mte_l0c_to_gm.cce` | L0C→GM FixPipe bandwidth (Cube output path) | 128×128 FP32 tile, 1280× FixPipe | BW[l0c→gm] sustained |
| `mte_hbm_allcore.cce` | HBM bandwidth under 20-core contention | 20 disjoint 16 MiB regions, 1280× GM→L1 | Per-core BW under full load |

### Pipeline Serialization

| Kernel | Purpose | Template | Expected Metric |
|--------|---------|----------|-----------------|
| `mandatory_handoff.cce` | Cube→Vector handoff cost isolation | K-sweep: 128/256/384/512/1024/2048, linear fit T(K)=α+βK | mandatory_handoff_cost ≈ 2000 cycles, R²>0.99 |

---

## Toolchain Requirements

### CCE Compiler
- **Tool**: `ccecom` (CCE compiler for Ascend)
- **Target**: Ascend 910B3 (20 AIC, 40 AIV, clock=1.85 GHz)
- **Version**: Matched to CANN toolkit on 910B3 server

### Compilation
```bash
# Compile single kernel
ccecom <kernel>.cce -o <kernel>.cce --npu-size=1

# Example:
ccecom cube_peak_fp16.cce -o cube_peak_fp16 --npu-size=1
```

---

## Running on Remote 910B3

### Prerequisites
1. SSH access to 910B3 server with CCE toolchain installed
2. Compiled `.cce` binaries in `~/cce_bench/` (or configured via `REMOTE_CCE_PATH`)
3. `hccl_run` command available (CANN runtime)

### Execution
Use the orchestration script from the parent directory:

```bash
# From vTriton root
cd perfbound/calibration

# Run the remote AscendC launcher and collect profiler CSVs
python3 scripts/cce_remote_bench.py \
  --host 910B3 \
  --n-repeat 45 \
  --output-dir bench_output
```

This runs all 15 default calibration kernels with 45 raw repeats and `msprof`,
producing one CSV per kernel plus per-K handoff CSVs in `./bench_output/`.
Use `--msprof PATH` when multiple CANN profiler versions are installed.

### Expected Output Format
Each kernel produces a CSV with msprof op_summary format:

```csv
#op_name,op_type,duration(us),cycles,task_id,core_id
cube,matrix,59.87,110763,0,0
```

`fit_constants.py` reads task duration for ordinary kernels and
`aic_fixpipe_time(us)` for the L0C→GM benchmark.

---

## Extracting Constants

After benchmarks complete, run the extraction script:

```bash
python scripts/fit_constants.py ./bench_output/ data/calib_910b3_v1.json
```

This outputs `calib_910b3_v1.json` with all P0 constants:

```json
{
  "version": "v1",
  "hardware_name": "Ascend 910B3",
  "constants": {
    "P_cube_fp16_sustained": {
      "name": "P_cube_fp16_sustained",
      "value": 280.5,
      "unit": "TFLOPS",
      "ci_95": 3.2,
      "source": "cce_microbench",
      "n_runs": 30,
      "notes": "CV=0.011"
    },
    ...
  }
}
```

---

## Validation

Run cross-validation against tilesim reference values:

```bash
python scripts/validate_vs_tilesim.py data/calib_910b3_v1.json
```

Checks:
1. `vec_cycle`: computing_cycles within 10% of 910B1 (clock-normalized)
2. `Cube FP16`: measured ≤ clock-scaled 910B4 × 1.05
3. `Bandwidth`: GM→UB ratio ≤ 1.3× 910B4 reference

---

## Reproducibility

To reproduce P0 constants within 10% on the same 910B3 unit:

1. **Same toolchain version**: Pin CCE toolkit version in compile commands
2. **Same thermal state**: Ensure chip is at steady-state temperature (avoid cold-start throttling)
3. **Same isolation**: No other workloads running on the 910B3 during measurement
4. **Same parameters**: Use template sizes (M=128, N=128, K=4096, etc.) as specified in kernel headers

### Run-to-Run Variance
- **Acceptable**: CV < 5% (coefficient of variation)
- **Target**: ≥30 runs per kernel (enforced in `fit_constants.py`)
- **If CV > 5%**: Increase `n_repeat` to 100, check for thermal throttling

---

## Troubleshooting

### "hccl_run: command not found"
- Ensure CANN toolkit is loaded: `source /usr/local/Ascend/ascend-toolkit/set_env.sh`

### "No 'cube' op found in CSV"
- Verify kernel compiled and ran successfully
- Check msprof profiling enabled: `--profile=msprof` flag required

### "CV > 5% for Cube FP16"
- Chip may be thermally throttling; allow cooldown between runs
- Increase `n_repeat` to 100 for better statistics

### "Cube FP16 exceeds scaled 910B4 max"
- Measurement artifact (e.g., measuring peak, not sustained)
- Re-run with longer warm-up to ensure steady-state measurement

---

## File Status

- **Status**: Template/stub implementations ready for Step 2 (remote 910B3 execution)
- **Blocker**: Requires CCE toolchain and 910B3 hardware access to compile and run
- **Next step**: Run `scripts/run_benchmarks.sh` on remote 910B3 to generate CSV data

---

## References

- **Spec**: `.omc/specs/performance_bound_model.md` §A.1 (Module 1 — Calibration Database)
- **Plan**: `.omc/plans/a1_calibration.md` (Implementation details)
- **Constants schema**: `perfbound/calibration/constants.py` (CalibrationConstant, CalibrationDB)
- **Hardware config**: `configs/ascend_910b3.json` (memory hierarchy, compute units)
