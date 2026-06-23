#!/usr/bin/env python3
# trace_calibrator.py — Main entry point for workload-based calibration.
#
# Orchestrates the four-layer calibration pipeline:
#   Layer 0: Core distribution (grid → core mapping, per-block → E2E scaling)
#   Layer 1: Per-core time calibration (startup latency, scalar overhead)
#   Layer 2: Pipeline overlap calibration (handoff, barrier cycles)
#   Layer 3: Multi-kernel convergence check and report generation
#
# Usage:
#   python perfbound/calibration/trace_calibrator.py \
#       --real-csv temp/real_op_summary.csv \
#       --model-trace temp/model_trace.json \
#       --calib-in perfbound/calibration/data/calib_910b3_v1.json \
#       --calib-out perfbound/calibration/data/calib_910b3_v2.json \
#       --report temp/calibration_report.json

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from perfbound.calibration.constants import CalibrationDB
from perfbound.calibration.calib_loader import load_calibration
from perfbound.calibration.msprof_trace_extractor import extract_real_trace, RealTrace
from perfbound.calibration.model_trace_extractor import extract_model_trace, ModelTrace
from perfbound.calibration.per_core_calibrator import (
    calibrate_per_core, apply_per_core_adjustments, PerCoreResult,
)
from perfbound.calibration.pipeline_calibrator import (
    calibrate_pipeline, apply_pipeline_adjustments, PipelineResult,
)
from perfbound.calibration.calib_reporter import (
    generate_report, save_report, print_report, CalibrationReport,
)
from perfbound.distribution.core_mapper import CoreMapper, DistributionResult


DEFAULT_CALIB_IN = "perfbound/calibration/data/calib_910b3_v1.json"
DEFAULT_CALIB_OUT = "perfbound/calibration/data/calib_910b3_v2.json"


def run_calibration(
    real_csvs: List[str | Path],
    model_traces: List[str | Path],
    calib_in: str | Path = DEFAULT_CALIB_IN,
    calib_out: Optional[str | Path] = None,
    report_out: Optional[str | Path] = None,
    convergence_threshold: float = 0.05,
    *,
    aic_cores: int = 20,
    aiv_cores: int = 40,
    grid: Optional[List[int]] = None,
) -> CalibrationReport:
    """Run the full calibration pipeline.

    Args:
        real_csvs: List of msprof op_summary.csv files from real hardware.
        model_traces: List of Perfetto trace JSON files from tritonsim-opt.
        calib_in: Path to input calibration JSON (v1).
        calib_out: Path to output calibrated JSON (v2).
        report_out: Path for calibration report JSON.
        convergence_threshold: Max allowed efficiency deviation.
        aic_cores: Number of AIC (Cube) physical cores (default 20).
        aiv_cores: Number of AIV (Vector) physical cores (default 40).
        grid: Override grid dimensions, e.g. ``(128, 32)``.
              When ``None``, inferred from msprof ``Block Num``.

    Returns:
        CalibrationReport with all results.
    """
    if len(real_csvs) != len(model_traces):
        raise ValueError(
            f"Mismatched inputs: {len(real_csvs)} real CSVs vs "
            f"{len(model_traces)} model traces"
        )

    # Load baseline calibration
    try:
        calib_db = load_calibration(calib_in)
        print(f"Loaded calibration DB: {calib_in}")
    except FileNotFoundError:
        print(f"Calibration DB not found at {calib_in}, using defaults")
        calib_db = None

    # ── Build core mapper ────────────────────────────────────────────────
    mapper = CoreMapper(aic_cores=aic_cores, aiv_cores=aiv_cores)

    per_core_results: List[PerCoreResult] = []
    pipeline_results: List[PipelineResult] = []
    distribution_results: List[DistributionResult] = []
    kernel_names: List[str] = []

    for real_csv, model_trace in zip(real_csvs, model_traces):
        print(f"\n--- Calibrating kernel: {Path(real_csv).stem} ---")

        # Extract traces
        real = extract_real_trace(real_csv)
        model = extract_model_trace(model_trace)
        kernel_names.append(real.kernel_name)

        # ── Layer 0: Core distribution ─────────────────────────────────
        total_blocks = grid if grid else real.block_dim
        if isinstance(total_blocks, (list, tuple)):
            total_blocks = total_blocks  # pass as grid tuple
        else:
            total_blocks = int(total_blocks)  # pass as scalar block count

        distribution = mapper.map(
            grid=total_blocks,
            per_block_span_aic_us=model.aic_total_time_us,
            per_block_span_aiv_us=model.aiv_total_time_us,
            task_type=real.task_type_raw or "MIX_AIC",
        )
        distribution_results.append(distribution)

        print(f"  Layer 0 (Core Dist): {real.block_dim} blocks × "
              f"{aic_cores}AIC+{aiv_cores}AIV")
        print(f"    Waves: AIC={distribution.waves_aic}, AIV={distribution.waves_aiv}")
        print(f"    Model E2E: AIC={distribution.e2e_aic_us:.1f}us  "
              f"AIV={distribution.e2e_aiv_us:.1f}us  "
              f"wall={distribution.e2e_wall_us:.1f}us")
        print(f"    Real  E2E: wall={real.kernel_wall_time_us:.1f}us")

        # Print per-block spans for diagnostics
        print(f"    Model per-block: AIC={model.aic_total_time_us:.1f}us  "
              f"AIV={model.aiv_total_time_us:.1f}us")
        print(f"    Real  per-core:  AIC={real.aic_total_time_us:.1f}us  "
              f"AIV={real.aiv_total_time_us:.1f}us")

        # ── Layer 1: Per-core calibration (use E2E model + real E2E)  ──
        # Build synthetic RealTrace / ModelTrace with E2E values so
        # existing calibrators work unchanged.
        real_e2e = RealTrace(
            kernel_name=real.kernel_name,
            csv_path=real.csv_path,
            n_cores=real.n_cores,
            block_dim=real.block_dim,
            # Align: real wall clock vs model E2E wall clock
            aic_total_time_us=real.kernel_wall_time_us,
            aiv_total_time_us=real.kernel_wall_time_us,
            kernel_wall_time_us=real.kernel_wall_time_us,
            task_type_raw=real.task_type_raw,
        )
        model_e2e = ModelTrace(
            kernel_name=model.kernel_name,
            trace_path=model.trace_path,
            aic_total_time_us=distribution.e2e_aic_us,
            aiv_total_time_us=distribution.e2e_aiv_us,
        )

        print(f"  Aligned: model_AIC={model_e2e.aic_total_time_us:.1f}us, "
              f"model_AIV={model_e2e.aiv_total_time_us:.1f}us, "
              f"real_wall={real_e2e.aic_total_time_us:.1f}us")

        per_core = calibrate_per_core(real_e2e, model_e2e, calib_db)
        per_core_results.append(per_core)
        print(f"  Per-core: AIC eff={per_core.aic_efficiency:.3f}  "
              f"AIV eff={per_core.aiv_efficiency:.3f}")

        # Layer 2: Pipeline calibration
        pipeline = calibrate_pipeline(real_e2e, model_e2e, calib_db)
        pipeline_results.append(pipeline)
        print(f"  Pipeline: eff={pipeline.pipeline_efficiency:.3f}")

    # Generate report
    report = generate_report(
        per_core_results, pipeline_results, kernel_names,
        description=f"Calibration vs {len(kernel_names)} kernel(s)",
        convergence_threshold=convergence_threshold,
    )

    # Inject core distribution metadata into report
    report.adjustments["_layer0_core_distribution"] = {
        "aic_cores": aic_cores,
        "aiv_cores": aiv_cores,
    }
    if distribution_results:
        dr = distribution_results[0]
        report.adjustments["_layer0_detail"] = {
            "waves_aic": dr.waves_aic,
            "waves_aiv": dr.waves_aiv,
            "bottleneck": dr.bottleneck,
            "per_block_span_aic_us": dr.per_block_span_aic_us,
            "per_block_span_aiv_us": dr.per_block_span_aiv_us,
        }

    print_report(report)

    # Save outputs
    if report_out:
        save_report(report, report_out)
        print(f"Report saved to {report_out}")

    if calib_out and calib_db is not None:
        # Apply adjustments to calibration DB
        calibrated_db = calib_db
        for pr in per_core_results:
            if pr.recommended_adjustments:
                calibrated_db = apply_per_core_adjustments(calibrated_db, pr)
        for pl in pipeline_results:
            if pl.recommendations:
                calibrated_db = apply_pipeline_adjustments(calibrated_db, pl)

        # Update version
        calibrated_db.version = "v2"
        calibrated_db.description = (
            f"Calibrated from {len(kernel_names)} kernel(s); "
            f"avg AIC eff={report.avg_aic_efficiency:.3f}, "
            f"avg AIV eff={report.avg_aiv_efficiency:.3f}"
        )

        calibrated_db.save(calib_out)
        print(f"Calibrated DB saved to {calib_out}")

    return report


def _cli():
    parser = argparse.ArgumentParser(
        description="TritonSim: workload-based hardware calibration",
    )
    parser.add_argument(
        "--real-csv", action="append", default=[],
        help="Path to msprof op_summary.csv (repeat for multiple kernels)",
    )
    parser.add_argument(
        "--model-trace", action="append", default=[],
        help="Path to model Perfetto trace JSON (repeat, same order as --real-csv)",
    )
    parser.add_argument(
        "--calib-in", default=DEFAULT_CALIB_IN,
        help=f"Input calibration JSON (default: {DEFAULT_CALIB_IN})",
    )
    parser.add_argument(
        "--calib-out", default=DEFAULT_CALIB_OUT,
        help=f"Output calibrated JSON (default: {DEFAULT_CALIB_OUT})",
    )
    parser.add_argument(
        "--report", default=None,
        help="Path for calibration report JSON",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.05,
        help="Convergence threshold (default: 0.05)",
    )
    parser.add_argument(
        "--aic-cores", type=int, default=20,
        help="Number of AIC (Cube) physical cores (default: 20)",
    )
    parser.add_argument(
        "--aiv-cores", type=int, default=40,
        help="Number of AIV (Vector) physical cores (default: 40)",
    )
    parser.add_argument(
        "--grid", type=int, nargs="+", default=None,
        help="Grid dimensions (e.g. '128 32'). Omit to infer from Block Num in CSV.",
    )

    args = parser.parse_args()

    if not args.real_csv or not args.model_trace:
        print("Error: --real-csv and --model-trace are required.", file=sys.stderr)
        print("Example:")
        print("  python perfbound/calibration/trace_calibrator.py \\")
        print("    --real-csv temp/real_op_summary.csv \\")
        print("    --model-trace temp/model_trace.json \\")
        print("    --calib-in perfbound/calibration/data/calib_910b3_v1.json \\")
        print("    --calib-out perfbound/calibration/data/calib_910b3_v2.json \\")
        print("    --report temp/calibration_report.json")
        sys.exit(1)

    run_calibration(
        real_csvs=args.real_csv,
        model_traces=args.model_trace,
        calib_in=args.calib_in,
        calib_out=args.calib_out,
        report_out=args.report,
        convergence_threshold=args.threshold,
        aic_cores=args.aic_cores,
        aiv_cores=args.aiv_cores,
        grid=args.grid,
    )


if __name__ == "__main__":
    _cli()
