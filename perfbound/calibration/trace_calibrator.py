#!/usr/bin/env python3
# trace_calibrator.py — Main entry point for workload-based calibration.
#
# Orchestrates the three-layer calibration pipeline:
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


DEFAULT_CALIB_IN = "perfbound/calibration/data/calib_910b3_v1.json"
DEFAULT_CALIB_OUT = "perfbound/calibration/data/calib_910b3_v2.json"


def run_calibration(
    real_csvs: List[str | Path],
    model_traces: List[str | Path],
    calib_in: str | Path = DEFAULT_CALIB_IN,
    calib_out: Optional[str | Path] = None,
    report_out: Optional[str | Path] = None,
    convergence_threshold: float = 0.05,
) -> CalibrationReport:
    """Run the full calibration pipeline.

    Args:
        real_csvs: List of msprof op_summary.csv files from real hardware.
        model_traces: List of Perfetto trace JSON files from tritonsim-opt.
        calib_in: Path to input calibration JSON (v1).
        calib_out: Path to output calibrated JSON (v2).
        report_out: Path for calibration report JSON.
        convergence_threshold: Max allowed efficiency deviation.

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

    per_core_results: List[PerCoreResult] = []
    pipeline_results: List[PipelineResult] = []
    kernel_names: List[str] = []

    for real_csv, model_trace in zip(real_csvs, model_traces):
        print(f"\n--- Calibrating kernel: {Path(real_csv).stem} ---")

        # Extract traces
        real = extract_real_trace(real_csv)
        model = extract_model_trace(model_trace)
        kernel_names.append(real.kernel_name)

        print(f"  Real:  AIC={real.aic_total_time_us:.1f}us  "
              f"AIV={real.aiv_total_time_us:.1f}us  "
              f"overlap={real.overlap_ratio:.3f}")
        print(f"  Model: AIC={model.aic_total_time_us:.1f}us  "
              f"AIV={model.aiv_total_time_us:.1f}us  "
              f"overlap={model.overlap_ratio:.3f}")

        # Layer 1: Per-core calibration
        per_core = calibrate_per_core(real, model, calib_db)
        per_core_results.append(per_core)
        print(f"  Per-core: AIC eff={per_core.aic_efficiency:.3f}  "
              f"AIV eff={per_core.aiv_efficiency:.3f}")

        # Layer 2: Pipeline calibration
        pipeline = calibrate_pipeline(real, model, calib_db)
        pipeline_results.append(pipeline)
        print(f"  Pipeline: eff={pipeline.pipeline_efficiency:.3f}")

    # Generate report
    report = generate_report(
        per_core_results, pipeline_results, kernel_names,
        description=f"Calibration vs {len(kernel_names)} kernel(s)",
        convergence_threshold=convergence_threshold,
    )

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
    )


if __name__ == "__main__":
    _cli()
