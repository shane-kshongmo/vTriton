# M4 — Tier 1 Grid Analytical Model (pure function, no I/O)
#
# T_grid_floor = T_total_work / (n_cores · occupancy · load_balance · I_binding)
#
# This is the "perfect grid" lower bound: if all cores were fully occupied with
# perfectly balanced work, the per-core time would be this.
#
# bytes_in scaled by redundancy(grid) (=1 by default, conservative).
# A redundancy > 1 means GM reads are amplified (e.g., overlapping tiles).
#
# Source spec: .omc/specs/performance_bound_model.md §1.4, §A.4

from __future__ import annotations

from dataclasses import dataclass

from ..calibration.constants import CoreConfig
from ..extract.dsl_extractor import GridInfo


@dataclass
class GridBound:
    """Tier 1 bound output.

    All time values in microseconds.  I_binding units must match total_work
    (e.g., B/us for memory-bound kernels, FLOP/us for compute-bound).
    """
    t_grid_floor_us: float       # lower bound from grid occupancy

    # Decomposed terms (for diagnostics and attribution)
    total_work: float             # aggregate work across all programs
    n_cores: int                  # cores used
    occupancy: float              # min(G, n_cores) / n_cores  (≤1)
    load_balance: float           # mean(work) / max(work)  (≤1)
    redundancy: float             # GM read amplification (≥1, default 1)
    i_binding: float              # HW throughput at binding component

    busiest_core_id: int          # core with max work (for Tier 2 analysis)

    def __repr__(self) -> str:
        return (f"GridBound(T_grid_floor={self.t_grid_floor_us:.2f} us, "
                f"occupancy={self.occupancy:.3f}, "
                f"load_balance={self.load_balance:.3f}, "
                f"i_binding={self.i_binding:.1f})")


def compute_grid_floor(
    grid: GridInfo,
    core: CoreConfig,
    i_binding: float,
    total_work: float,
    is_cube_kernel: bool = True,
    n_cores_override: int | None = None,
) -> GridBound:
    """Compute T_grid_floor from Tier 1 grid information.

    The grid floor is the chip-level lower bound on time:

    T_grid_floor = total_work · redundancy / (n_cores · occupancy · load_balance · I_binding)

    total_work is caller-supplied in the SAME units as i_binding:
      - memory-bound: total_work = Σ bytes, i_binding = BW in B/us
      - compute-bound: total_work = Σ FLOPs, i_binding = throughput in FLOP/us

    GridInfo.work[p] (tile/element counts) is used for occupancy and
    load_balance RATIOS (units cancel) but NOT for the absolute numerator —
    deriving total_work from GridInfo.work would mix units.

    Args:
        grid: M2-extracted grid quantities (occupancy, load_balance, work).
        core: Core topology (AIC/AIV counts).
        i_binding: Hardware throughput at the binding component.
                   Units: B/us or FLOP/us — must match total_work units.
        total_work: Aggregate work (bytes or FLOPs) across all programs,
                    in the same unit as i_binding. Scaled by redundancy
                    internally.
        is_cube_kernel: If True, use Cube core count (20 AIC); if False,
                        use Vector-only count (40 AIV).
        n_cores_override: If given, use this core count directly (ignores
                        is_cube_kernel). Callers pass the SAME resolved count
                        they used for Tier-2 wave scaling so both tiers agree.

    Returns:
        GridBound with T_grid_floor and all decomposed terms.
    """
    n_cores = (
        n_cores_override if n_cores_override is not None
        else (core.n_cores_cube if is_cube_kernel else core.n_cores_vector_only)
    )

    # Use occupancy and load_balance from GridInfo if available,
    # otherwise compute from grid geometry
    occupancy = grid.occupancy if grid.occupancy > 0 else 1.0
    load_balance = grid.load_balance if grid.load_balance > 0 else 1.0
    redundancy = grid.redundancy if grid.redundancy > 0 else 1.0

    # total_work is caller-supplied (bytes or FLOPs), scaled by redundancy
    scaled_work = total_work * redundancy
    if scaled_work <= 0:
        scaled_work = 1.0

    # Effective parallel throughput
    effective_i = n_cores * occupancy * load_balance * i_binding

    if effective_i <= 0:
        t_grid_floor_us = float("inf")
    else:
        t_grid_floor_us = scaled_work / effective_i

    return GridBound(
        t_grid_floor_us=t_grid_floor_us,
        total_work=scaled_work,
        n_cores=n_cores,
        occupancy=occupancy,
        load_balance=load_balance,
        redundancy=redundancy,
        i_binding=i_binding,
        busiest_core_id=grid.busiest_core_id,
    )
