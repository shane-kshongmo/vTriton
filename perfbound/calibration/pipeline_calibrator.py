#!/usr/bin/env python3
# pipeline_calibrator.py — Layer 2: Pipeline overlap calibration.
#
# Compares real pipeline overlap (AIC↔AIV concurrency) against the model's
# ideal (fully-parallel) assumption, producing an overlap factor that can
# be used to adjust pipe_barrier and mandatory_handoff parameters.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .constants import CalibrationDB, CalibrationConstant
from .msprof_trace_extractor import RealTrace
from .model_trace_extractor import ModelTrace


@dataclass
class PipelineResult:
    """Pipeline calibration result for one kernel."""
    kernel_name: str = ""

    model_overlap_ratio: float = 0.0
    real_overlap_ratio: float = 0.0
    pipeline_efficiency: float = 0.0

    recommendations: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


def calibrate_pipeline(
    real: RealTrace,
    model: ModelTrace,
    calib_db: Optional[CalibrationDB] = None,
) -> PipelineResult:
    """Compare real vs model pipeline overlap and propose adjustments.

    The pipeline_efficiency = real_overlap / model_overlap reflects
    how much of the ideal Cube∥Vector parallelism is actually achieved
    on hardware.  When < 1, mandatory_handoff_cycles or
    pipe_barrier_cycles_per_iter may need to increase to account for
    synchronization bubbles.

    Args:
        real: Real hardware trace from msprof.
        model: Model trace from Perfetto.
        calib_db: Current calibration DB for reference values.

    Returns:
        PipelineResult with overlap comparison and adjustment suggestions.
    """
    result = PipelineResult(
        kernel_name=real.kernel_name,
        model_overlap_ratio=model.overlap_ratio,
        real_overlap_ratio=real.overlap_ratio,
    )

    if model.overlap_ratio > 0 and real.overlap_ratio > 0:
        result.pipeline_efficiency = real.overlap_ratio / model.overlap_ratio
    elif real.overlap_ratio > 0:
        result.pipeline_efficiency = real.overlap_ratio
    else:
        result.pipeline_efficiency = 1.0

    # Get current values
    current_handoff = 7621.0
    current_barrier = 7500.0

    if calib_db is not None:
        current_handoff = calib_db.mandatory_handoff_cycles
        current_barrier = calib_db.pipe_barrier_cycles_per_iter

    # If overlap is worse than model, increase serial parameters
    if result.pipeline_efficiency < 0.98:
        gap = 1.0 - result.pipeline_efficiency
        # Distribute the gap between handoff and barrier proportionally
        result.recommendations["mandatory_handoff_cycles"] = (
            current_handoff * (1.0 + gap * 0.6)
        )
        result.recommendations["pipe_barrier_cycles_per_iter"] = (
            current_barrier * (1.0 + gap * 0.4)
        )

        result.notes = (
            f"Pipeline overlap gap: {gap:.1%}. "
            f"Suggested handoff={result.recommendations['mandatory_handoff_cycles']:.0f} "
            f"barrier={result.recommendations['pipe_barrier_cycles_per_iter']:.0f}"
        )
    else:
        result.notes = "Pipeline overlap within 2% — no adjustment needed."

    return result


def apply_pipeline_adjustments(
    calib_db: CalibrationDB,
    result: PipelineResult,
) -> CalibrationDB:
    """Apply pipeline calibration adjustments to a CalibrationDB."""
    import copy
    db = copy.deepcopy(calib_db)

    for key, new_val in result.recommendations.items():
        if key == "mandatory_handoff_cycles":
            db.mandatory_handoff_cycles = new_val
        elif key == "pipe_barrier_cycles_per_iter":
            db.pipe_barrier_cycles_per_iter = new_val

        if key in db.constants:
            db.constants[key].value = new_val
            db.constants[key].source = "pipeline_calibration"
        else:
            db.constants[key] = CalibrationConstant(
                name=key, value=new_val, unit="cycles",
                ci_95=0.0, source="pipeline_calibration", n_runs=1,
                notes=f"Calibrated from {result.kernel_name} pipeline overlap",
            )

    return db
