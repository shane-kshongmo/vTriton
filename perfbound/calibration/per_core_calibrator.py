#!/usr/bin/env python3
# per_core_calibrator.py — Layer 1: Per-core time calibration.
#
# Compares model AIC/AIV total time against real hardware measurements
# from msprof, producing efficiency ratios and recommended adjustments
# to startup_latency and scalar_overhead_factor.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .constants import CalibrationDB, CalibrationConstant
from .msprof_trace_extractor import RealTrace
from .model_trace_extractor import ModelTrace


@dataclass
class PerCoreResult:
    """Calibration result for one kernel."""
    kernel_name: str = ""

    aic_model_us: float = 0.0
    aic_real_us: float = 0.0
    aic_efficiency: float = 0.0

    aiv_model_us: float = 0.0
    aiv_real_us: float = 0.0
    aiv_efficiency: float = 0.0

    recommended_adjustments: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    @property
    def is_valid(self) -> bool:
        return self.aic_efficiency > 0 and self.aiv_efficiency > 0


def calibrate_per_core(
    real: RealTrace,
    model: ModelTrace,
    calib_db: Optional[CalibrationDB] = None,
) -> PerCoreResult:
    """Compare model vs real per-core times and propose adjustments.

    The AIC efficiency ratio reflects the cumulative error of Cube
    throughput + CubeMTE2 bandwidth + FixPipe bandwidth + Cube startup
    latency.  The AIV ratio reflects Vector + VecMTE2 + MTE3 + Vector
    startup.

    Recommended adjustments target startup_latency and scalar_overhead
    because these are the ONLY unmeasured seed values in the
    calibration database — M1 microbenchmarks already provide measured
    sustained rates for all compute and memory components.

    Adjustment logic: if model is FASTER than real (efficiency > 1.0),
    increase the startup latency proportionally.  If model is SLOWER
    (efficiency < 1.0), decrease — but never below 0.

    Args:
        real: Real hardware trace from msprof.
        model: Model trace from Perfetto.
        calib_db: Current calibration DB to extract current startup values.

    Returns:
        PerCoreResult with efficiency ratios and suggested adjustments.
    """
    result = PerCoreResult(
        kernel_name=real.kernel_name,
        aic_model_us=model.aic_total_time_us,
        aic_real_us=real.aic_total_time_us,
        aiv_model_us=model.aiv_total_time_us,
        aiv_real_us=real.aiv_total_time_us,
    )

    if model.aic_total_time_us > 0 and real.aic_total_time_us > 0:
        result.aic_efficiency = model.aic_total_time_us / real.aic_total_time_us
    if model.aiv_total_time_us > 0 and real.aiv_total_time_us > 0:
        result.aiv_efficiency = model.aiv_total_time_us / real.aiv_total_time_us

    adjustments = {}

    # Get current startup latencies from calibration DB or defaults
    current_startup_cube = 20.0
    current_startup_vector = 35.0
    current_startup_mte2 = 50.0
    current_startup_mte3 = 40.0
    current_scalar_overhead = 3.74

    if calib_db is not None:
        sl = calib_db.startup_latency
        current_startup_cube = float(sl.get("cube", 20))
        current_startup_vector = float(sl.get("vector", 35))
        current_startup_mte2 = float(sl.get("mte2", 50))
        current_startup_mte3 = float(sl.get("mte3", 40))
        current_scalar_overhead = calib_db.scalar_overhead_factor

    # AIC efficiency → adjust Cube-side startups
    if result.aic_efficiency > 0 and abs(result.aic_efficiency - 1.0) > 0.02:
        factor_aic = result.aic_efficiency
        if factor_aic > 1.0:
            adjustments["startup_latency.cube"] = current_startup_cube * factor_aic
            adjustments["startup_latency.mte2"] = current_startup_mte2 * factor_aic
        else:
            adjustments["startup_latency.cube"] = max(1.0, current_startup_cube * factor_aic)
            adjustments["startup_latency.mte2"] = max(1.0, current_startup_mte2 * factor_aic)

    # AIV efficiency → adjust Vector-side startups
    if result.aiv_efficiency > 0 and abs(result.aiv_efficiency - 1.0) > 0.02:
        factor_aiv = result.aiv_efficiency
        if factor_aiv > 1.0:
            adjustments["startup_latency.vector"] = current_startup_vector * factor_aiv
            adjustments["startup_latency.mte3"] = current_startup_mte3 * factor_aiv
            adjustments["scalar_overhead_factor"] = current_scalar_overhead * factor_aiv
        else:
            adjustments["startup_latency.vector"] = max(1.0, current_startup_vector * factor_aiv)
            adjustments["startup_latency.mte3"] = max(1.0, current_startup_mte3 * factor_aiv)
            adjustments["scalar_overhead_factor"] = max(1.0, current_scalar_overhead * factor_aiv)

    result.recommended_adjustments = adjustments

    if not adjustments:
        result.notes = "Both AIC and AIV efficiencies within 2% — no adjustment needed."
    else:
        parts = []
        if result.aic_efficiency > 0:
            parts.append(f"AIC efficiency: {result.aic_efficiency:.3f}")
        if result.aiv_efficiency > 0:
            parts.append(f"AIV efficiency: {result.aiv_efficiency:.3f}")
        result.notes = "; ".join(parts)

    return result


def apply_per_core_adjustments(
    calib_db: CalibrationDB,
    result: PerCoreResult,
) -> CalibrationDB:
    """Apply per-core calibration adjustments to a CalibrationDB.

    Returns a modified copy; does not mutate the original.
    """
    import copy
    db = copy.deepcopy(calib_db)

    for key, new_val in result.recommended_adjustments.items():
        parts = key.split(".")
        if len(parts) == 2 and parts[0] == "startup_latency":
            unit_name = parts[1]
            db.startup_latency[unit_name] = new_val
            name = f"startup_latency.{unit_name}"
            if name in db.constants:
                db.constants[name].value = new_val
                db.constants[name].source = "per_core_calibration"
                db.constants[name].notes = (
                    f"Calibrated from {result.kernel_name}; "
                    f"efficiency={result.aic_efficiency if 'aic' in unit_name else result.aiv_efficiency:.3f}"
                )
            else:
                db.constants[name] = CalibrationConstant(
                    name=name, value=new_val, unit="cycles",
                    ci_95=0.0, source="per_core_calibration", n_runs=1,
                    notes=f"Derived from {result.kernel_name}",
                )
        elif key == "scalar_overhead_factor":
            db.scalar_overhead_factor = new_val
        elif key == "pipe_barrier_cycles_per_iter":
            db.pipe_barrier_cycles_per_iter = new_val

    return db
