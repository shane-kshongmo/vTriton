#!/usr/bin/env python3
"""
fit_constants.py - Extract calibration constants from msprof CSV output.

Reads msprof op_summary CSVs from CCE microbenchmark runs, computes sustained
hardware rates with confidence intervals, and outputs calib_910b3_v1.json.

Usage:
    python fit_constants.py <input_dir> [output_json]

Args:
    input_dir: Directory containing msprof CSV outputs (from run_benchmarks.sh)
    output_json: Output JSON path (default: perfbound/calibration/data/calib_910b3_v1.json)

Outputs:
    calib_910b3_v1.json: Calibration database with all P0 constants
    bandwidth_910b3.csv: Bandwidth curve table (tilesim format)

Acceptance (per A.1 plan §6):
    - All P0 constants: n_runs ≥ 30, CV < 5%
    - mandatory_handoff_cost: R² > 0.99 on linear fit
    - All constants carry source="cce_microbench"

Source spec: .omc/specs/performance_bound_model.md §A.1
Related: perfbound/calibration/scripts/run_benchmarks.sh
Related: perfbound/calibration/constants.py (CalibrationConstant, CalibrationDB)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import statistics

# Add perfbound to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from perfbound.calibration.constants import (
    CalibrationConstant,
    CalibrationDB,
    CoreConfig,
    CubeConfig,
    VectorConfig,
    MemHierarchy,
    MemBandwidth,
    DType,
)


MTE_PATHS = [("gm", "ub"), ("ub", "gm"), ("gm", "l1"), ("l1", "l0a"), ("l0c", "gm")]


# ── CSV Parsing ─────────────────────────────────────────────────────────────

@dataclass
class MSProfRow:
    """One row from msprof op_summary CSV."""
    op_name: str
    op_type: str
    duration_us: float
    cycles: float
    task_id: int = 0
    core_id: int = 0
    task_type: str = ""          # "AI_CORE", "AI_CPU", "AIV", etc.
    start_time_us: float = 0.0   # Task Start Time(us)
    fixpipe_time_us: float = 0.0
    aiv_scalar_time_us: float = 0.0
    aicore_time_us: float = 0.0   # from aicore_time(us) column (MIX tasks)
    aiv_time_us: float = 0.0      # from aiv_time(us) column (MIX tasks)
    block_dim: int = 0


def read_msprof_csv(csv_path: Path) -> List[MSProfRow]:
    """Read msprof op_summary CSV into structured rows.

    Real op_summary CSVs interleave aggregate/host rows whose numeric columns
    are ``N/A``; these are skipped.  Rather than print one warning per skipped
    row (real CSVs can have dozens), the count is summarised once at the end.
    """
    rows = []
    skipped = 0
    first_skip_reason = ""
    with open(csv_path) as f:
        reader = csv.DictReader(line for line in f if not line.strip().startswith("#"))
        for line in reader:
            try:
                row = MSProfRow(
                    op_name=_first_present(line, ["op_name", "Op Name", "Name"], "unknown"),
                    op_type=_first_present(line, ["op_type", "Op Type", "OP Type", "Type"], "unknown"),
                    duration_us=float(_first_present(
                        line,
                        ["duration(us)", "Duration(us)", "Task Duration(us)", "duration_us"],
                        "0",
                    )),
                    cycles=float(_first_present(
                        line,
                        ["cycles", "Cycles", "aic_total_cycles", "aiv_total_cycles"],
                        "0",
                    )),
                    task_id=int(float(_first_present(line, ["task_id", "Task ID"], "0"))),
                    core_id=int(float(_first_present(line, ["core_id", "Core ID"], "0"))),
                    task_type=_first_present(
                        line,
                        ["task_type", "Task Type", "TaskType"],
                        "",
                    ),
                    start_time_us=float(_first_present(
                        line,
                        ["start_time(us)", "Task Start Time(us)", "StartTime(us)"],
                        "0",
                    )),
                    fixpipe_time_us=float(_first_present(
                        line,
                        ["aic_fixpipe_time(us)", "AIC FixPipe Time(us)"],
                        "0",
                    )),
                    aiv_scalar_time_us=float(_first_present(
                        line,
                        ["aiv_scalar_time(us)", "AIV Scalar Time(us)"],
                        "0",
                    )),
                    aicore_time_us=float(_first_present(
                        line,
                        ["aicore_time(us)", "aicore_time (us)"],
                        "0",
                    )),
                    aiv_time_us=float(_first_present(
                        line,
                        ["aiv_time(us)", "aiv_time (us)"],
                        "0",
                    )),
                    block_dim=int(float(_first_present(
                        line,
                        ["Block Dim", "Block Num", "block_dim"],
                        "0",
                    ))),
                )
                rows.append(row)
            except (ValueError, KeyError) as e:
                skipped += 1
                if not first_skip_reason:
                    first_skip_reason = str(e)
    if skipped:
        print(
            f"Warning: skipped {skipped} non-numeric/aggregate row(s) in "
            f"{csv_path} (e.g. {first_skip_reason})",
            file=sys.stderr,
        )
    return rows


def _first_present(row: dict[str, str], names: list[str], default: str) -> str:
    """Return the first non-empty CSV value among possible msprof column names."""
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value.strip()
    return default


# ── Statistics ───────────────────────────────────────────────────────────────

def compute_mean_ci(values: List[float], confidence: float = 0.95) -> tuple[float, float]:
    """Compute mean and confidence interval for a list of measurements.

    Returns:
        (mean, ci_half_width) where ci_half_width uses a normal 95% factor.
    """
    if not values:
        return 0.0, 0.0

    n = len(values)
    mean = statistics.fmean(values)

    if n < 2:
        return mean, 0.0

    z_value = 1.96 if confidence == 0.95 else 1.96
    ci_half = z_value * statistics.stdev(values) / (n ** 0.5)
    return mean, ci_half


def compute_cv(values: List[float]) -> float:
    """Coefficient of variation: sample stddev / mean."""
    if not values:
        return float("inf")
    mean = statistics.fmean(values)
    if mean == 0:
        return float("inf")
    return statistics.stdev(values) / mean if len(values) > 1 else 0.0


def select_steady_state_samples(values: List[float]) -> tuple[List[float], int]:
    """Discard the chronological warmup third when at least 45 runs exist.

    Calibration runs use 45 profiler-visible invocations so that the final 30
    are independent steady-state measurements. Preserve input order: sorting
    by duration would select best-case order statistics and overstate the
    sustained hardware rate.
    """
    if len(values) < 45:
        return values, 0
    warmup_count = len(values) // 3
    return values[warmup_count:], warmup_count


# ── Constant Extraction ───────────────────────────────────────────────────────

def extract_cube_constant(csv_path: Path, dtype: DType) -> CalibrationConstant:
    """Extract Cube throughput constant from msprof CSV.

    Formula: P = (2 * M * N * K * N_iter) / cube_time_us / 1e6  [TFLOPS]
    """
    rows = read_msprof_csv(csv_path)
    op_markers = {"cube", csv_path.stem.lower()}
    cube_rows = [r for r in rows if r.op_name.lower() in op_markers]

    if not cube_rows:
        raise ValueError(f"No cube op found in {csv_path}")

    durations = [r.duration_us for r in cube_rows if r.duration_us > 0]

    if len(durations) < 30:
        print(f"Warning: Only {len(durations)} measurements for Cube {dtype.value}")

    # Template parameters from kernel source
    M, N, K = 128, 128, 4096  # From cube_peak_*.cce template
    N_iter = 30  # Outer loop repeats

    flops_per_iter = 2 * M * N * K  # 2 for multiply-add
    total_flops = flops_per_iter * N_iter

    rates = [total_flops / duration / 1e6 for duration in durations]
    mean_tflops, ci_tflops = compute_mean_ci(rates)

    cv = compute_cv(rates)
    if cv > 0.05:
        print(f"Warning: Cube {dtype.value} CV={cv:.3f} > 0.05 (reject quality check)")

    return CalibrationConstant(
        name=f"P_cube_{dtype.value}_sustained",
        value=mean_tflops,
        unit="TFLOPS",
        ci_95=ci_tflops,
        source="cce_microbench",
        n_runs=len(durations),
        notes=f"M={M}, N={N}, K={K}, N_iter={N_iter}, CV={cv:.3f}",
    )


def extract_vector_constant(csv_path: Path, op_name: str) -> CalibrationConstant:
    """Extract Vector throughput constant for one operation type.

    Formula: P = (buffer_elements * N_iter) / vector_time_us / 1e3  [GFLOPS]
    """
    rows = read_msprof_csv(csv_path)
    op_markers = {op_name.lower(), f"vector_{op_name.lower()}", csv_path.stem.lower()}
    vec_rows = [r for r in rows if r.op_name.lower() in op_markers]
    if not vec_rows:
        vec_rows = [r for r in rows if r.op_name.lower() == "vector"]

    if not vec_rows:
        raise ValueError(f"No vector op for '{op_name}' found in {csv_path}")

    durations = [r.duration_us for r in vec_rows if r.duration_us > 0]

    if len(durations) < 30:
        print(f"Warning: Only {len(durations)} measurements for Vector {op_name}")

    # Template parameters from vector_peak_*.cce
    buffer_elements = 256  # UB buffer size
    N_iter = 10000  # Repeat count

    total_ops = buffer_elements * N_iter
    rates = [total_ops / duration / 1e3 for duration in durations]
    mean_gflops, ci_gflops = compute_mean_ci(rates)

    cv = compute_cv(rates)
    if cv > 0.05:
        print(f"Warning: Vector {op_name} CV={cv:.3f} > 0.05")

    return CalibrationConstant(
        name=f"P_vector_{op_name}_sustained",
        value=mean_gflops,
        unit="GFLOPS",
        ci_95=ci_gflops,
        source="cce_microbench",
        n_runs=len(durations),
        notes=f"buffer={buffer_elements}, N_iter={N_iter}, CV={cv:.3f}",
    )


def extract_mte_bandwidth(csv_path: Path, src: str, dst: str) -> CalibrationConstant:
    """Extract MTE bandwidth constant.

    Formula: BW = (transfer_bytes * N_iter) / mte_time_us / 1000.0  [GB/s]
    """
    rows = read_msprof_csv(csv_path)
    path_marker = f"mte_{src}_to_{dst}".lower()
    mte_rows = [r for r in rows if r.op_name.lower() == path_marker]
    if not mte_rows:
        mte_rows = [r for r in rows if "mte" in r.op_name.lower()]

    if not mte_rows:
        raise ValueError(f"No MTE op found in {csv_path}")

    raw_durations = [r.duration_us for r in mte_rows if r.duration_us > 0]
    if len(raw_durations) < 30:
        print(f"Warning: Only {len(raw_durations)} measurements for MTE {src}→{dst}")

    durations, warmup_samples = select_steady_state_samples(raw_durations)
    steady_samples = len(durations)

    # Template parameters from mte_*.cce
    transfer_bytes = 131072 * 2  # kMteElements * sizeof(half)
    N_iter = 1280  # kMteMeasured

    total_bytes = transfer_bytes * N_iter
    rates = [total_bytes / duration / 1000.0 for duration in durations]
    mean_bw, ci_bw = compute_mean_ci(rates)

    cv = compute_cv(rates)
    if cv > 0.05:
        print(f"Warning: MTE {src}→{dst} CV={cv:.3f} > 0.05")

    return CalibrationConstant(
        name=f"BW_{src}_to_{dst}_sustained",
        value=mean_bw,
        unit="GB/s",
        ci_95=ci_bw,
        source="cce_microbench",
        n_runs=len(durations),
        notes=(
            f"transfer={transfer_bytes/1024}KiB, N_iter={N_iter}, "
            f"warmup_samples={warmup_samples}, steady_samples={steady_samples}, "
            f"selection=chronological, CV={cv:.3f}"
        ),
    )


def extract_l0c_to_gm_bandwidth(csv_path: Path) -> CalibrationConstant:
    """Extract FixPipe (L0C→GM) sustained bandwidth.

    The FixPipe microbench runs a Cube MMAD pipeline with repeated
    Fixpipe (L0C→GM) transfers.  Each Fixpipe iteration transfers
    M * N * sizeof(float) bytes from L0C to GM.

    Formula: BW = (M * N * sizeof(float) * N_fixpipe_iters) / time_us / 1000.0  [GB/s]
    """
    rows = read_msprof_csv(csv_path)
    op_markers = {"mte_l0c_to_gm", csv_path.stem.lower()}
    fixpipe_rows = [r for r in rows if r.op_name.lower() in op_markers]
    if not fixpipe_rows:
        fixpipe_rows = [r for r in rows if "mte" in r.op_name.lower() or "fixpipe" in r.op_name.lower()]

    if not fixpipe_rows:
        raise ValueError(f"No FixPipe op found in {csv_path}")

    raw_durations = [
        r.fixpipe_time_us if r.fixpipe_time_us > 0 else r.duration_us
        for r in fixpipe_rows
        if r.fixpipe_time_us > 0 or r.duration_us > 0
    ]
    if len(raw_durations) < 30:
        print(f"Warning: Only {len(raw_durations)} measurements for L0C→GM FixPipe")

    durations, warmup_samples = select_steady_state_samples(raw_durations)
    steady_samples = len(durations)

    # Template parameters from mte_l0c_to_gm.cce / vt_microbench_common.h
    M, N = 128, 128
    bytes_per_fixpipe = M * N * 4  # sizeof(float) = 4
    N_iter = 1280  # kMteMeasured

    total_bytes = bytes_per_fixpipe * N_iter
    rates = [total_bytes / duration / 1000.0 for duration in durations]
    mean_bw, ci_bw = compute_mean_ci(rates)

    cv = compute_cv(rates)
    if cv > 0.05:
        print(f"Warning: L0C→GM FixPipe CV={cv:.3f} > 0.05")

    return CalibrationConstant(
        name="BW_l0c_to_gm_sustained",
        value=mean_bw,
        unit="GB/s",
        ci_95=ci_bw,
        source="cce_microbench",
        n_runs=len(durations),
        notes=(
            f"fixpipe M={M}, N={N}, bytes_per_iter={bytes_per_fixpipe}, "
            f"N_iter={N_iter}, profiler_metric=aic_fixpipe_time, "
            f"warmup_samples={warmup_samples}, steady_samples={steady_samples}, "
            f"selection=chronological, CV={cv:.3f}"
        ),
    )


def extract_scalar_constant(csv_path: Path) -> CalibrationConstant:
    """Extract sustained scalar FP32 FMA throughput from scalar_peak.

    scalar_peak runs a dependent scalar recurrence:
        acc = acc * c1 + c2
    for kScalarRepeat iterations. Each iteration is one multiply plus one add,
    and the dependency chain prevents vectorization. Use aiv_scalar_time(us)
    rather than whole task duration so the final GM store does not pollute the
    arithmetic throughput.
    """
    rows = read_msprof_csv(csv_path)
    scalar_rows = [r for r in rows if r.op_name.lower() == "scalar_peak"]
    if not scalar_rows:
        scalar_rows = [r for r in rows if "scalar" in r.op_name.lower()]
    if not scalar_rows:
        raise ValueError(f"No scalar_peak op found in {csv_path}")

    observed_block_dims = {r.block_dim for r in scalar_rows if r.block_dim > 0}
    if observed_block_dims and observed_block_dims != {1}:
        raise ValueError(
            f"scalar_peak requires Block Dim 1, got {sorted(observed_block_dims)}"
        )

    raw_durations = [
        r.aiv_scalar_time_us if r.aiv_scalar_time_us > 0 else r.duration_us
        for r in scalar_rows
        if r.aiv_scalar_time_us > 0 or r.duration_us > 0
    ]
    if len(raw_durations) < 30:
        print(f"Warning: Only {len(raw_durations)} measurements for scalar_peak")

    durations, warmup_samples = select_steady_state_samples(raw_durations)

    scalar_repeat = 1_000_000
    flops_per_iter = 2
    total_flops = scalar_repeat * flops_per_iter
    rates = [total_flops / duration / 1000.0 for duration in durations]
    mean_gflops, ci_gflops = compute_mean_ci(rates)
    cv = compute_cv(rates)
    if cv > 0.05:
        print(f"Warning: scalar_peak CV={cv:.3f} > 0.05")

    return CalibrationConstant(
        name="P_scalar_add_sustained",
        value=mean_gflops,
        unit="GFLOPS",
        ci_95=ci_gflops,
        source="cce_microbench",
        n_runs=len(durations),
        notes=(
            f"dependent scalar FMA chain, N_iter={scalar_repeat}, "
            f"flops_per_iter={flops_per_iter}, "
            f"profiler_metric=aiv_scalar_time, warmup_samples={warmup_samples}, "
            f"steady_samples={len(durations)}, selection=chronological, "
            f"CV={cv:.6f}"
        ),
    )


def extract_hbm_allcore_bandwidth(csv_path: Path) -> CalibrationConstant:
    """Extract all-core HBM sustained bandwidth.

    The all-core benchmark launches on all 20 AIC cores simultaneously.
    Each core reads a disjoint region via GM→L1 DataCopy.  The
    per-core sustained rate under full contention is:

    Formula: BW_per_core = (transfer_bytes * N_iter) / core_time_us / 1000.0  [GB/s]

    The aggregate chip bandwidth is BW_per_core * n_active_cores, but
    we store the per-core rate since the grid model uses per-core I_binding.
    """
    rows = read_msprof_csv(csv_path)
    op_markers = {"mte_hbm_allcore", csv_path.stem.lower()}
    hbm_rows = [r for r in rows if r.op_name.lower() in op_markers]
    if not hbm_rows:
        hbm_rows = [r for r in rows if "mte" in r.op_name.lower() or "hbm" in r.op_name.lower()]

    if not hbm_rows:
        raise ValueError(f"No all-core HBM op found in {csv_path}")

    observed_block_dims = {r.block_dim for r in hbm_rows if r.block_dim > 0}
    if observed_block_dims and observed_block_dims != {20}:
        raise ValueError(
            f"All-core HBM requires Block Dim 20, got {sorted(observed_block_dims)}"
        )

    raw_durations = [r.duration_us for r in hbm_rows if r.duration_us > 0]
    if len(raw_durations) < 30:
        print(f"Warning: Only {len(raw_durations)} measurements for all-core HBM")

    durations, warmup_samples = select_steady_state_samples(raw_durations)
    steady_samples = len(durations)

    # Template parameters from mte_hbm_allcore.cce / vt_microbench_common.h
    transfer_bytes = 131072 * 2  # kMteElements * sizeof(half)
    N_iter = 1280  # kMteMeasured
    n_active_cores = 20

    total_bytes = transfer_bytes * N_iter
    # Per-core rate under full contention
    rates = [total_bytes / duration / 1000.0 for duration in durations]
    mean_bw, ci_bw = compute_mean_ci(rates)

    cv = compute_cv(rates)
    if cv > 0.05:
        print(f"Warning: all-core HBM CV={cv:.3f} > 0.05")

    return CalibrationConstant(
        name="BW_hbm_allcore_sustained",
        value=mean_bw,
        unit="GB/s",
        ci_95=ci_bw,
        source="cce_microbench",
        n_runs=len(durations),
        notes=(
            f"transfer={transfer_bytes/1024}KiB, N_iter={N_iter}, "
            f"n_active_cores={n_active_cores}, warmup_samples={warmup_samples}, "
            f"steady_samples={steady_samples}, selection=chronological, "
            f"CV={cv:.3f}, per_core_rate_under_contention"
        ),
    )


def extract_mandatory_handoff(csv_dir: Path) -> CalibrationConstant:
    """Extract mandatory handoff cost via linear fit T(K) = α + β·K.

    Reads mandatory_handoff CSVs for each K value, performs linear regression,
    and returns the intercept (α) as the handoff cost in cycles.

    Acceptance: R² > 0.99
    """
    K_values = [128, 256, 384, 512, 1024, 2048]
    total_times = {}  # K → mean total time (μs)
    run_counts = []

    for K in K_values:
        csv_path = csv_dir / f"mandatory_handoff_K{K}.csv"
        if not csv_path.exists():
            print(f"Warning: No mandatory_handoff CSV for K={K}")
            continue

        rows = read_msprof_csv(csv_path)
        # The current handoff benchmark is a single mixed AIC/AIV kernel. Older
        # CSVs may contain separate producer/consumer rows; keep that fallback
        # for historical data only.
        durations = [r.duration_us for r in rows if r.duration_us > 0]
        if durations:
            has_vector_consumer = any(r.op_name.lower() == "mandatory_handoff_vector" for r in rows)
            if has_vector_consumer and len(durations) >= 2 and len(durations) % 2 == 0:
                per_run = [sum(durations[i:i + 2]) for i in range(0, len(durations), 2)]
            else:
                per_run = durations
            total_times[K] = statistics.fmean(per_run)
            run_counts.append(len(per_run))
        else:
            print(f"Warning: No data for K={K}")

    if len(total_times) < 3:
        raise ValueError(f"Need at least 3 K values for fit, got {len(total_times)}")

    K_arr = list(total_times.keys())
    T_arr = list(total_times.values())
    slope, intercept, r_squared, std_err = linear_regression(K_arr, T_arr)

    if r_squared < 0.99:
        print(f"Warning: mandatory_handoff R²={r_squared:.4f} < 0.99 (quality check failed)")

    clock_freq_mhz = 1850.0
    intercept_cycles = intercept * clock_freq_mhz

    return CalibrationConstant(
        name="mandatory_handoff_cost_l0c_to_gm_to_ub",
        value=intercept_cycles,
        unit="cycles",
        ci_95=std_err * clock_freq_mhz,
        source="cce_microbench",
        n_runs=min(run_counts) if run_counts else 0,
        notes=f"linear fit T(K)=α+βK, R²={r_squared:.4f}, slope={slope:.6f}",
    )


def linear_regression(x_values: List[float], y_values: List[float]) -> tuple[float, float, float, float]:
    """Return slope, intercept, R², and intercept standard error."""
    n = len(x_values)
    if n != len(y_values) or n < 2:
        raise ValueError("linear_regression requires matching x/y values")

    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    ss_xx = sum((x - x_mean) ** 2 for x in x_values)
    if ss_xx == 0:
        raise ValueError("linear_regression requires non-constant x values")

    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    fitted = [slope * x + intercept for x in x_values]
    ss_res = sum((y - y_hat) ** 2 for y, y_hat in zip(y_values, fitted))
    ss_tot = sum((y - y_mean) ** 2 for y in y_values)
    r_squared = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot

    if n > 2:
        residual_var = ss_res / (n - 2)
        intercept_stderr = (residual_var * (1.0 / n + x_mean ** 2 / ss_xx)) ** 0.5
    else:
        intercept_stderr = 0.0
    return slope, intercept, r_squared, intercept_stderr


# ── Main Extraction ───────────────────────────────────────────────────────────

def extract_all_constants(input_dir: Path) -> CalibrationDB:
    """Extract all P0 calibration constants from msprof CSVs.

    Args:
        input_dir: Directory containing CSV outputs from run_benchmarks.sh

    Returns:
        CalibrationDB with all measured constants
    """
    input_dir = Path(input_dir)
    constants = {}

    # Cube constants (P0)
    for dtype_name, dtype in [("fp16", DType.FP16), ("int8", DType.INT8), ("bf16", DType.BF16)]:
        csv_path = input_dir / f"cube_peak_{dtype_name}.csv"
        if csv_path.exists():
            try:
                const = extract_cube_constant(csv_path, dtype)
                constants[const.name] = const
                print(f"✓ {const.name}: {const.value:.2f} ± {const.ci_95:.2f} {const.unit}")
            except Exception as e:
                print(f"✗ cube_peak_{dtype_name}: {e}")

    # Vector constants (P0 ops)
    for op in ["add", "mul", "max", "min"]:
        csv_path = input_dir / f"vector_peak_elemwise_{op}.csv"
        if csv_path.exists():
            try:
                const = extract_vector_constant(csv_path, op)
                constants[const.name] = const
                print(f"✓ {const.name}: {const.value:.2f} ± {const.ci_95:.2f} {const.unit}")
            except Exception as e:
                print(f"✗ vector_{op}: {e}")

    # Transcendental ops
    for op in ["exp", "log", "sqrt", "rsqrt"]:
        csv_path = input_dir / f"vector_peak_transcendental_{op}.csv"
        if not csv_path.exists():
            csv_path = input_dir / "vector_peak_transcendental.csv"
        if csv_path.exists():
            try:
                const = extract_vector_constant(csv_path, op)
                constants[const.name] = const
                print(f"✓ {const.name}: {const.value:.2f} ± {const.ci_95:.2f} {const.unit}")
            except Exception as e:
                print(f"✗ vector_{op}: {e}")

    # MTE bandwidth constants (P0)
    for src, dst in [("gm", "ub"), ("ub", "gm"), ("gm", "l1"), ("l1", "l0a")]:
        csv_path = input_dir / f"mte_{src}_to_{dst}.csv"
        if csv_path.exists():
            try:
                const = extract_mte_bandwidth(csv_path, src, dst)
                constants[const.name] = const
                print(f"✓ {const.name}: {const.value:.2f} ± {const.ci_95:.2f} {const.unit}")
            except Exception as e:
                print(f"✗ mte_{src}_to_{dst}: {e}")

    hbm_ingress = constants.get("BW_gm_to_l1_sustained")
    if hbm_ingress is not None:
        for name, path in [
            ("BW_gm_to_ub_sustained", "gm→ub"),
            ("BW_ub_to_gm_sustained", "ub→gm"),
        ]:
            const = constants.get(name)
            if const is not None and const.value > 5000.0:
                constants[name] = CalibrationConstant(
                    name=name,
                    value=hbm_ingress.value,
                    unit=hbm_ingress.unit,
                    ci_95=hbm_ingress.ci_95,
                    source="cce_microbench",
                    n_runs=hbm_ingress.n_runs,
                    notes=(
                        f"{path} vector MTE msprof duration was implausible "
                        f"({const.value:.1f} GB/s); using GM→L1 HBM ingress "
                        f"measurement as conservative fallback. {hbm_ingress.notes}"
                    ),
                )
                print(f"↳ {name}: replaced implausible vector-MTE result with GM→L1 fallback")

    # L0C→GM FixPipe bandwidth (P0)
    csv_path = input_dir / "mte_l0c_to_gm.csv"
    if csv_path.exists():
        try:
            const = extract_l0c_to_gm_bandwidth(csv_path)
            constants[const.name] = const
            print(f"✓ {const.name}: {const.value:.2f} ± {const.ci_95:.2f} {const.unit}")
        except Exception as e:
            print(f"✗ mte_l0c_to_gm: {e}")

    # All-core HBM bandwidth (P0)
    csv_path = input_dir / "mte_hbm_allcore.csv"
    if csv_path.exists():
        try:
            const = extract_hbm_allcore_bandwidth(csv_path)
            constants[const.name] = const
            print(f"✓ {const.name}: {const.value:.2f} ± {const.ci_95:.2f} {const.unit}")
        except Exception as e:
            print(f"✗ mte_hbm_allcore: {e}")

    # Mandatory handoff (P0)
    try:
        const = extract_mandatory_handoff(input_dir)
        constants[const.name] = const
        print(f"✓ {const.name}: {const.value:.0f} ± {const.ci_95:.0f} {const.unit}")
    except Exception as e:
        print(f"✗ mandatory_handoff: {e}")

    scalar_measured = False
    csv_path = input_dir / "scalar_peak.csv"
    if csv_path.exists():
        try:
            const = extract_scalar_constant(csv_path)
            constants[const.name] = const
            scalar_measured = True
            print(f"✓ {const.name}: {const.value:.4f} ± {const.ci_95:.4f} {const.unit}")
        except Exception as e:
            print(f"✗ scalar_peak: {e}")

    # Preserve derived scalar-throughput evidence only when no direct scalar
    # benchmark is available. Direct scalar measurements are allowed to tighten
    # scalar-bound floors because they carry CCE provenance and confidence.
    vector_add = constants.get("P_vector_add_sustained")
    if vector_add is not None and "P_scalar_add_sustained" not in constants:
        scalar = CalibrationConstant(
            name="P_scalar_add_sustained",
            value=vector_add.value / 128.0,
            unit="GFLOPS",
            ci_95=vector_add.ci_95 / 128.0,
            source="derived_from_vector_microbench",
            n_runs=vector_add.n_runs,
            notes=(
                "DERIVED (NOT measured): P_vector_add_sustained / SIMD width "
                "128. Evidence only; scalar_throughput_measured=false keeps "
                "this estimate out of the performance bound."
            ),
        )
        constants[scalar.name] = scalar

    memory = MemHierarchy(
        gm_size_gb=32.0,
        l2_size_mb=192.0,
        l1_size_kb=1024.0,
        l0a_size_kb=64.0,
        l0b_size_kb=64.0,
        l0c_size_kb=256.0,
        ub_size_kb=256.0,
    )
    for src, dst in MTE_PATHS:
        name = f"BW_{src}_to_{dst}_sustained"
        const = constants.get(name)
        if const is not None:
            memory.bw[(src, dst, -1)] = MemBandwidth(src, dst, const.value)

    db = CalibrationDB(
        version="v1",
        hardware_name="Ascend 910B3",
        description="Calibration database from CCE microbenchmarks",
        core=CoreConfig(aic_core_num=20, aiv_core_num=40, clock_freq_ghz=1.85),
        cube=CubeConfig(
            throughput={
                DType.FP16: constants.get("P_cube_fp16_sustained", CalibrationConstant("missing", 0, "TFLOPS", 0, "missing", 0)).value,
                DType.INT8: constants.get("P_cube_int8_sustained", CalibrationConstant("missing", 0, "TFLOPS", 0, "missing", 0)).value,
                DType.BF16: constants.get("P_cube_bf16_sustained", CalibrationConstant("missing", 0, "TFLOPS", 0, "missing", 0)).value,
            },
            fractal_sizes={
                DType.FP16: (16, 16, 16),
                DType.INT8: (16, 32, 16),
                DType.BF16: (16, 16, 16),
            },
        ),
        vector=VectorConfig(
            vec_width_elements=128,
            throughput_fp16_tflops=constants.get("P_vector_add_sustained", CalibrationConstant("missing", 0, "GFLOPS", 0, "missing", 0)).value / 1000,
            scalar_throughput_fp16_tflops=constants.get(
                "P_scalar_add_sustained",
                CalibrationConstant("missing", 0, "GFLOPS", 0, "missing", 0),
            ).value / 1000,
            scalar_throughput_measured=scalar_measured,
        ),
        memory=memory,
        constants=constants,
    )

    # Add mandatory handoff if extracted
    if "mandatory_handoff_cost_l0c_to_gm_to_ub" in constants:
        db.mandatory_handoff_cycles = constants["mandatory_handoff_cost_l0c_to_gm_to_ub"].value

    return db


def write_bandwidth_csv(db: CalibrationDB, csv_path: Path) -> None:
    """Write measured MTE bandwidths in the tilesim-compatible CSV schema."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["src_mem", "dst_mem", "core_num", "pkt_size", "mode", "bandwidth(GB/s)"],
            lineterminator="\n",
        )
        writer.writeheader()
        for (src, dst, core_num), bw in sorted(db.memory.bw.items()):
            writer.writerow({
                "src_mem": src,
                "dst_mem": dst,
                "core_num": core_num,
                "pkt_size": bw.pkt_size,
                "mode": "sustained",
                "bandwidth(GB/s)": f"{bw.bw_gb_per_s:.6g}",
            })


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract calibration constants from msprof CSVs")
    parser.add_argument("input_dir", type=Path, help="Directory with msprof CSV outputs")
    parser.add_argument("output_json", type=Path, nargs="?",
                        default=PROJECT_ROOT / "perfbound/calibration/data/calib_910b3_v1.json",
                        help="Output JSON path")
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory {args.input_dir} does not exist")
        return 1

    print(f"Extracting constants from: {args.input_dir}")
    print("=" * 60)

    try:
        db = extract_all_constants(args.input_dir)

        # Write output JSON
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(db.to_dict(), f, indent=2)
        bandwidth_csv = args.output_json.parent / "bandwidth_910b3.csv"
        write_bandwidth_csv(db, bandwidth_csv)

        print("=" * 60)
        print(f"✓ Calibration DB written to: {args.output_json}")
        print(f"✓ Bandwidth CSV written to: {bandwidth_csv}")

        # Validate P0 quality
        violations = db.validate_p0_constants()
        if violations:
            print("⚠ P0 quality violations:")
            for v in violations:
                print(f"  - {v}")
            return 1
        else:
            print("✓ All P0 constants pass quality checks (n_runs≥30, CV<5%)")
            return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
