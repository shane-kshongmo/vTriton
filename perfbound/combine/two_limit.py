# M5 / A.7 — Two-Limit Computation
#
# T_bound_HIVM = bound with avoidable structural constraints analytically
#                relaxed, recomputed from an idealized extract (NOT by
#                subtracting gap values from T_bound_DSL).
# T_bound_DSL  = bound over HIVM bishengir actually emits (realized
#                structural constraints) — just the regular T_bound.
#
# compiler_headroom = T_bound_DSL − T_bound_HIVM (≥ 0 by construction
#                     since relaxation only lowers floors)
# author_headroom   = None until M6 supplies T_measured
#
# The idealized extract relaxes:
#   - Gap-1 mis-placed ops reassigned to their eligible unit
#   - Avoidable handoffs removed from the serialization set
# Gap-2 and Gap-4 efficiency remain in T_bound_HIVM (hardware limits).
#
# Source spec: .omc/specs/performance_bound_model.md §A.7

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from ..extract.hivm_extractor import HIVMExtract, OpRecord, HandoffRecord
from ..extract.op_classifier import Component
from ..extract.eligibility_oracle import get_eligibility

if TYPE_CHECKING:
    from ..calibration.constants import CalibrationDB
    from ..extract.dsl_extractor import GridInfo


@dataclass
class TwoLimitResult:
    """Two-limit gap analysis for a single kernel."""
    kernel_name: str

    t_bound_hivm_us: float       # analytically relaxed (hardware-legal)
    t_bound_dsl_us: float         # realized bishengir structure
    t_measured_us: Optional[float] = None  # from msprof (M6)

    @property
    def compiler_headroom_us(self) -> float:
        return self.t_bound_dsl_us - self.t_bound_hivm_us

    @property
    def author_headroom_us(self) -> Optional[float]:
        if self.t_measured_us is None:
            return None
        return self.t_measured_us - self.t_bound_dsl_us

    def __repr__(self) -> str:
        return (f"TwoLimit({self.kernel_name}: "
                f"HIVM={self.t_bound_hivm_us:.2f}, "
                f"DSL={self.t_bound_dsl_us:.2f}, "
                f"measured={self.t_measured_us})")


# ── Helpers ─────────────────────────────────────────────────────────────

_MATMUL_KEYWORDS = ("matmul", "mm", "bmm")
_REDUCTION_KEYWORDS = ("reduce", "sum", "max", "min", "arg")
_COMPARE_KEYWORDS = ("cmp", "compare")


def _op_category(op_name: str) -> str:
    """Map an op name to an eligibility-oracle category."""
    lower = op_name.lower()
    if any(k in lower for k in _MATMUL_KEYWORDS):
        return "matmul"
    if any(k in lower for k in _REDUCTION_KEYWORDS):
        return "reduction"
    if any(k in lower for k in _COMPARE_KEYWORDS):
        return "compare"
    return "elementwise"


def _build_idealized_extract(extract: HIVMExtract) -> HIVMExtract:
    """Build an idealized extract by relaxing avoidable structural constraints.

    Relaxations (Gap-1 placement only):
    - Re-assign ops whose realized component is NOT in their eligible set
      to the first eligible compute unit (recompute per-component O_c).
    - Remove avoidable handoffs (those between same-path components).
      Mandatory cross-path handoffs are kept.

    Gap-2 and Gap-4 efficiency are NOT relaxed (hardware limits remain).
    """
    # Deep copy operations to avoid mutating the original
    ideal_ops = []
    reassigned_ids: set[int] = set()

    for op in extract.operations:
        ideal_op = OpRecord(
            op_id=op.op_id,
            op_name=op.op_name,
            component=op.component,
            precision=op.precision,
            pipe=op.pipe,
            bytes_transferred=op.bytes_transferred,
            elements=op.elements,
            flops=op.flops,
            duration_cycles=op.duration_cycles,
            loop_multiplier=op.loop_multiplier,
            depends_on=list(op.depends_on),
            src_space=op.src_space,
            dst_space=op.dst_space,
            repeat=op.repeat,
            mask=op.mask,
            start_cycle=op.start_cycle,
            end_cycle=op.end_cycle,
            hw_unit=op.hw_unit,
        )

        # MTE has fixed assignment — skip
        if op.component in (Component.MTE_GM, Component.MTE_L1, Component.MTE_UB):
            ideal_ops.append(ideal_op)
            continue

        # Check if this op is mis-placed (Gap 1)
        category = _op_category(op.op_name)
        prec_str = op.precision.value if op.precision else None
        eligible = get_eligibility(category, prec_str)

        if op.component not in eligible and eligible:
            # Re-assign to the first eligible compute component
            # Priority: Cube > Vector > Scalar (prefer faster unit)
            for preferred in (Component.CUBE, Component.VECTOR, Component.SCALAR):
                if preferred in eligible:
                    ideal_op.component = preferred
                    reassigned_ids.add(op.op_id)
                    break

        ideal_ops.append(ideal_op)

    # Build idealized handoff list: keep only mandatory cross-path handoffs.
    # Avoidable handoffs (same-path) are removed — the idealized kernel
    # has perfect scheduling/ping-pong.
    ideal_handoffs = []
    for h in extract.handoffs:
        # Keep the handoff only if it's cross-path (mandatory)
        # Use the same logic as serialization._is_cross_component_mandatory
        from ..model.serialization import _same_path
        if not _same_path(h.producer_component, h.consumer_component):
            ideal_handoffs.append(h)

    return HIVMExtract(
        operations=ideal_ops,
        handoffs=ideal_handoffs,
        o_prec={},  # will be recomputed by compute_component_floor
        total_flops={},
        total_bytes={},
        transfer_sizes={},
        transfer_alignments={},
        unit_assignment={op.op_id: op.component.value for op in ideal_ops},
    )


def compute_two_limit(
    kernel_name: str,
    grid_info: "GridInfo",
    extract: HIVMExtract,
    calib_db: "CalibrationDB",
    t_bound_dsl_us: float,
    t_measured_us: Optional[float] = None,
    n_cores: int | None = None,
    total_programs: int | None = None,
) -> TwoLimitResult:
    """Compute T_bound_HIVM by recomputing the bound from an idealized extract.

    The idealized extract relaxes avoidable structural constraints:
    - Gap-1 mis-placed ops reassigned to their eligible unit
    - Avoidable handoffs removed

    T_bound_HIVM is computed by running the same compute_bounds/combine
    pipeline on this idealized extract.  compiler_headroom = T_bound_DSL −
    T_bound_HIVM ≥ 0 by construction.

    Args:
        kernel_name: Kernel identifier.
        grid_info: M2-extracted grid quantities.
        extract: Realized M3 HIVM extraction result.
        calib_db: Calibration database with sustained rates.
        t_bound_dsl_us: The realized DSL bound (from M5 combine).
        t_measured_us: Optional measured time for author-headroom gap (M6).
        n_cores: Number of cores (for wave scaling).
        total_programs: Total program instances (for wave scaling).

    Returns:
        TwoLimitResult with compiler + author headroom gaps.
    """
    from ..model.bounds import compute_bounds
    from .bound_combiner import combine

    # Build idealized extract and recompute bound
    ideal_extract = _build_idealized_extract(extract)
    ideal_pieces = compute_bounds(
        grid_info, ideal_extract, calib_db,
        n_cores=n_cores, total_programs=total_programs,
    )
    ideal_result = combine(
        ideal_pieces.grid, ideal_pieces.component, ideal_pieces.serial,
        kernel_name=kernel_name, extract=ideal_extract, calibration=calib_db,
    )

    t_bound_hivm_us = ideal_result.t_bound_us

    # Ensure T_bound_HIVM ≤ T_bound_DSL (relaxation can only lower floors)
    # If floating-point rounding causes a tiny inversion, clamp.
    t_bound_hivm_us = min(t_bound_hivm_us, t_bound_dsl_us)

    return TwoLimitResult(
        kernel_name=kernel_name,
        t_bound_hivm_us=t_bound_hivm_us,
        t_bound_dsl_us=t_bound_dsl_us,
        t_measured_us=t_measured_us,
    )
