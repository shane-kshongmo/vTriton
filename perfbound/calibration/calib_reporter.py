#!/usr/bin/env python3
# calib_reporter.py — Calibration report generation.
#
# Produces human-readable and machine-parseable calibration reports
# comparing model predictions against real hardware measurements.

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .constants import CalibrationDB
from .per_core_calibrator import PerCoreResult
from .pipeline_calibrator import PipelineResult


@dataclass
class CalibrationReport:
    """Complete calibration report for one or more kernels."""
    version: str = "v1"
    description: str = ""
    kernels: List[str] = field(default_factory=list)

    # Aggregate metrics
    avg_aic_efficiency: float = 0.0
    avg_aiv_efficiency: float = 0.0
    avg_pipeline_efficiency: float = 0.0

    # Per-kernel results
    per_core_results: List[Dict[str, Any]] = field(default_factory=list)
    pipeline_results: List[Dict[str, Any]] = field(default_factory=list)

    # Recommended adjustments (merged across kernels)
    adjustments: Dict[str, float] = field(default_factory=dict)

    # Convergence status
    converged: bool = False
    convergence_notes: str = ""


def generate_report(
    per_core_results: List[PerCoreResult],
    pipeline_results: List[PipelineResult],
    kernel_names: Optional[List[str]] = None,
    description: str = "",
    convergence_threshold: float = 0.05,
) -> CalibrationReport:
    """Generate a calibration report from per-core and pipeline results.

    Args:
        per_core_results: List of per-core calibration results.
        pipeline_results: List of pipeline calibration results.
        kernel_names: Names of calibrated kernels.
        description: Human-readable description.
        convergence_threshold: Max allowed deviation from 1.0 for convergence.

    Returns:
        CalibrationReport with aggregated metrics and adjustments.
    """
    report = CalibrationReport(description=description)

    if kernel_names:
        report.kernels = kernel_names
    else:
        report.kernels = [r.kernel_name for r in per_core_results]

    # Aggregate per-core
    aic_effs = [r.aic_efficiency for r in per_core_results if r.aic_efficiency > 0]
    aiv_effs = [r.aiv_efficiency for r in per_core_results if r.aiv_efficiency > 0]

    if aic_effs:
        report.avg_aic_efficiency = sum(aic_effs) / len(aic_effs)
    if aiv_effs:
        report.avg_aiv_efficiency = sum(aiv_effs) / len(aiv_effs)

    # Aggregate pipeline
    pipe_effs = [r.pipeline_efficiency for r in pipeline_results
                 if r.pipeline_efficiency > 0]
    if pipe_effs:
        report.avg_pipeline_efficiency = sum(pipe_effs) / len(pipe_effs)

    # Per-kernel details
    for r in per_core_results:
        report.per_core_results.append({
            "kernel": r.kernel_name,
            "aic": {"model_us": round(r.aic_model_us, 2),
                    "real_us": round(r.aic_real_us, 2),
                    "efficiency": round(r.aic_efficiency, 4)},
            "aiv": {"model_us": round(r.aiv_model_us, 2),
                    "real_us": round(r.aiv_real_us, 2),
                    "efficiency": round(r.aiv_efficiency, 4)},
            "notes": r.notes,
        })

    for r in pipeline_results:
        report.pipeline_results.append({
            "kernel": r.kernel_name,
            "model_overlap": round(r.model_overlap_ratio, 4),
            "real_overlap": round(r.real_overlap_ratio, 4),
            "efficiency": round(r.pipeline_efficiency, 4),
            "notes": r.notes,
        })

    # Merge adjustments
    all_adjustments: Dict[str, List[float]] = {}
    for r in per_core_results:
        for k, v in r.recommended_adjustments.items():
            all_adjustments.setdefault(k, []).append(v)
    for r in pipeline_results:
        for k, v in r.recommendations.items():
            all_adjustments.setdefault(k, []).append(v)

    for key, vals in all_adjustments.items():
        report.adjustments[key] = sum(vals) / len(vals)

    # Check convergence
    max_dev = 0.0
    for eff in aic_effs + aiv_effs + pipe_effs:
        if eff > 0:
            max_dev = max(max_dev, abs(eff - 1.0))

    report.converged = max_dev <= convergence_threshold
    if report.converged:
        report.convergence_notes = (
            f"All efficiencies within ±{convergence_threshold:.0%} of 1.0. "
            f"Calibration converged."
        )
    else:
        report.convergence_notes = (
            f"Max efficiency deviation: {max_dev:.2%} > threshold {convergence_threshold:.0%}. "
            f"Additional kernel profiles or iteration needed."
        )

    return report


def save_report(report: CalibrationReport, output_path: str | Path) -> Path:
    """Save calibration report to JSON."""
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        json.dump(asdict(report), f, indent=2)
    return output_path


def print_report(report: CalibrationReport) -> None:
    """Print a human-readable calibration report to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  CALIBRATION REPORT: {report.description or 'TritonSim Calibration'}")
    print(f"  Kernels: {', '.join(report.kernels)}")
    print(sep)

    print(f"\n  Per-Core Efficiencies:")
    for r in report.per_core_results:
        print(f"    {r['kernel']}:")
        print(f"      AIC: model={r['aic']['model_us']:.1f}us  real={r['aic']['real_us']:.1f}us  "
              f"eff={r['aic']['efficiency']:.3f}")
        print(f"      AIV: model={r['aiv']['model_us']:.1f}us  real={r['aiv']['real_us']:.1f}us  "
              f"eff={r['aiv']['efficiency']:.3f}")

    print(f"\n  Pipeline Overlap:")
    for r in report.pipeline_results:
        print(f"    {r['kernel']}: model={r['model_overlap']:.3f}  real={r['real_overlap']:.3f}  "
              f"eff={r['efficiency']:.3f}")

    print(f"\n  Averages:")
    print(f"    AIC efficiency:    {report.avg_aic_efficiency:.3f}")
    print(f"    AIV efficiency:    {report.avg_aiv_efficiency:.3f}")
    print(f"    Pipeline efficiency: {report.avg_pipeline_efficiency:.3f}")

    if report.adjustments:
        print(f"\n  Recommended Adjustments:")
        for k, v in sorted(report.adjustments.items()):
            print(f"    {k}: {v:.1f}")

    print(f"\n  Convergence: {'✓ CONVERGED' if report.converged else '✗ NOT CONVERGED'}")
    print(f"    {report.convergence_notes}")
    print(f"{sep}\n")
