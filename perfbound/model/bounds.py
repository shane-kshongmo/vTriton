# M4 — Single entry point wiring both analytical models.
#
# compute_bounds picks i_binding and total_work consistently
# (memory-bound: BW_gm_ub + Σ MTE bytes; compute-bound: Cube FLOP/us + Σ flops)
# and calls compute_grid_floor, compute_component_floor, classify_handoffs.
#
# Wave scaling (A.5 Change #2b): the extract is ONE program; the busiest
# core runs waves = ceil(total_programs / n_cores) programs.
#   - Tier 2: T_core_floor scales by waves (busiest-core component work)
#   - Tier 1: total_work = total_programs × per_program_work, naturally
#     carrying the per-core waves factor through the grid floor.
#
# Source: .omc/plans/a4_two_analytical_models.md Change #3
#         .omc/plans/a5_bound_combiner.md Change #2b

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..extract.hivm_extractor import HIVMExtract
from ..extract.dsl_extractor import GridInfo
from ..calibration.constants import CoreConfig
from .grid_model import GridBound, compute_grid_floor
from .component_model import ComponentBound, compute_component_floor
from .serialization import SerializationSplit, classify_handoffs

if TYPE_CHECKING:
    from ..calibration.constants import CalibrationDB


@dataclass
class BoundPieces:
    """The three pieces of an analytical bound, pre-combination.

    A.5's combine() consumes this to produce T_bound with attribution.
    """
    grid: GridBound
    component: ComponentBound
    serial: SerializationSplit


def compute_bounds(
    grid_info: GridInfo,
    extract: HIVMExtract,
    calib_db: "CalibrationDB",
    is_cube_kernel: bool = True,
    n_cores: int | None = None,
    total_programs: int | None = None,
) -> BoundPieces:
    """Compute the three bound pieces from grid + extract + calibration.

    Picks i_binding and total_work to match the binding component's unit.
    Applies wave scaling: the busiest core runs ceil(total_programs/n_cores)
    programs, so both Tier 1 and Tier 2 floors scale accordingly.

    Args:
        grid_info: M2-extracted grid quantities.
        extract: M3 HIVM extraction result.
        calib_db: Calibration database with sustained rates.
        is_cube_kernel: True for Cube-bearing kernels (20 AIC).
        n_cores: Number of cores.  Defaults to core config value if None.
        total_programs: Total program instances.  Defaults to grid_info
                        total_programs if None.

    Returns:
        BoundPieces with grid, component, and serial pieces.
    """
    core = calib_db.core
    cube = calib_db.cube
    vector = calib_db.vector
    memory = calib_db.memory

    # Wave scaling: busiest core runs `waves` programs
    _n_cores = n_cores if n_cores is not None else (
        core.n_cores_cube if is_cube_kernel else core.n_cores_vector_only
    )
    _total_programs = total_programs if total_programs is not None else grid_info.total_programs
    waves = math.ceil(_total_programs / _n_cores) if _n_cores > 0 else 1

    # Active cores contending for HBM = programs running concurrently, capped by
    # the physical core count.  Drives the occupancy-aware MTE bandwidth cap so
    # both tiers (component + grid) agree on the achievable aggregate.
    active_cores = min(_total_programs, _n_cores) if _n_cores > 0 else _total_programs

    # Compute component floor first to discover which component binds
    comp = compute_component_floor(
        extract, cube, vector, memory, core, active_cores,
        pipe_barrier_cycles_per_iter=calib_db.pipe_barrier_cycles_per_iter,
    )

    # Apply wave scaling to component floor:
    # busiest core runs `waves` programs, so per-core time is waves × single-program time.
    # Scale t_core_floor_us and per_component_us by waves.
    # Do NOT scale total_ops/total_bytes — they stay per-program so that gap
    # helpers (_compute_gap1/4) produce correctly waves-scaled absolute gaps:
    #   gap = comp_time_w × (op_work_1prog / total_ops_1prog)
    # where comp_time_w already includes the waves factor.
    if waves > 1:
        comp = replace(
            comp,
            t_core_floor_us=comp.t_core_floor_us * waves,
            per_component_us={k: v * waves for k, v in comp.per_component_us.items()},
        )

    # Pick i_binding and total_work to match the binding component's unit
    binding = comp.binding_component
    from ..extract.op_classifier import Component

    if binding in (Component.MTE_GM, Component.MTE_L1, Component.MTE_UB):
        # Memory-bound: i_binding = per-core BW in B/us, total_work = bytes.
        # Derive the per-core rate from the component floor, which already
        # applies the occupancy-aware HBM cap (min(single_core, hbm_peak/active)).
        # The grid floor multiplies by n_cores·occupancy = active_cores, so the
        # effective aggregate becomes the measured HBM peak at full occupancy —
        # and both tiers use the SAME achievable rate (no desync, stays sound).
        # (Supersedes the flat BW_hbm_allcore_sustained constant, which assumed a
        # fixed all-core rate independent of how many programs actually run.)
        binding_time = comp.per_component_us.get(binding.value, 0.0)
        if waves > 1:
            binding_time /= waves
        binding_bytes = comp.total_bytes.get(binding.value, 0.0)
        i_binding = (
            binding_bytes / binding_time
            if binding_bytes > 0 and binding_time > 0
            else 1.0
        )
        per_program_work = comp.total_bytes.get(binding.value, 0.0)
        # Scale by total_programs for chip-level grid floor
        total_work = per_program_work * _total_programs
    else:
        # Compute-bound (Cube, Vector): derive i_binding from the component
        # floor's binding rate so the grid tier and component tier use the SAME
        # achievable rate.  Recomputing it from an aggregate (peak) constant
        # here desynchronised the two tiers and made the grid floor unsound
        # (it could exceed measured time) for near-peak compute kernels.
        binding_work_1prog = comp.total_ops.get(binding.value, 0.0)
        binding_time_1prog = comp.per_component_us.get(binding.value, 0.0)
        if waves > 1:
            binding_time_1prog /= waves
        if binding_work_1prog > 0 and binding_time_1prog > 0:
            i_binding = binding_work_1prog / binding_time_1prog  # work/us
        elif binding == Component.CUBE:
            from .component_model import _get_cube_throughput_ops_per_us, _prec_to_dtype
            from ..extract.op_classifier import Precision
            i_binding = _get_cube_throughput_ops_per_us(
                _prec_to_dtype(Precision.FP16), cube
            ) or 1.0
        else:
            i_binding = (vector.throughput_fp16_tflops * 1e6) or 1.0  # FLOP/us
        # Grid floor work is the binding component's work (matches i_binding's
        # unit), scaled across all programs.
        per_program_work = binding_work_1prog
        total_work = per_program_work * _total_programs

    total_work = max(total_work, 1.0)  # avoid division by zero

    grid = compute_grid_floor(grid_info, core, i_binding, total_work,
                              is_cube_kernel=is_cube_kernel)

    serial = classify_handoffs(
        extract.handoffs,
        mandatory_handoff_cycles=calib_db.mandatory_handoff_cycles,
        clock_ghz=core.clock_freq_ghz,
        memory=memory,
    )

    return BoundPieces(grid=grid, component=comp, serial=serial)
