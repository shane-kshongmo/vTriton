# A.5 — Two-limit tests.
#
# Covers:
#   - Change #7: two-limit by recomputation (not subtraction)
#   - Idealizing a Gap-1 kernel lowers T_bound_HIVM below T_bound_DSL
#   - A kernel already at the bound → compiler_headroom ≈ 0
#
# Source: .omc/plans/a5_bound_combiner.md Change #7

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from perfbound.combine.two_limit import compute_two_limit, _build_idealized_extract
from perfbound.extract.hivm_extractor import HIVMExtract, OpRecord, HandoffRecord
from perfbound.extract.op_classifier import Component, Precision
from perfbound.extract.dsl_extractor import GridInfo
from perfbound.calibration.calib_loader import load_default_calib_db
from perfbound.model.bounds import compute_bounds
from perfbound.combine.bound_combiner import combine


def _make_grid(n_cores=20, total_programs=20) -> GridInfo:
    return GridInfo(
        grid_dims=(total_programs,),
        total_programs=total_programs,
        tile_assignment={}, work={},
        occupancy=1.0, load_balance=1.0,
        redundancy=1.0, busiest_core_id=0,
    )


class TestIdealizedExtract:
    """_build_idealized_extract relaxes Gap-1 and avoidable handoffs."""

    def test_gap1_op_reassigned(self):
        """A FP16 compare on Scalar → reassigned to Vector in idealized."""
        ops = [
            OpRecord(op_id=1, op_name="cmp_fp16", component=Component.SCALAR,
                     precision=Precision.FP16, pipe="Scalar",
                     elements=1024, flops=0, loop_multiplier=1, depends_on=[]),
        ]
        extract = HIVMExtract(operations=ops, handoffs=[])
        ideal = _build_idealized_extract(extract)
        # FP16 compare is eligible for Vector → reassigned
        assert ideal.operations[0].component == Component.VECTOR

    def test_true_scalar_unchanged(self):
        """i32 compare stays on Scalar (eligible = {Scalar})."""
        ops = [
            OpRecord(op_id=1, op_name="cmp_i32", component=Component.SCALAR,
                     precision=Precision.INT32, pipe="Scalar",
                     elements=512, flops=0, loop_multiplier=1, depends_on=[]),
        ]
        extract = HIVMExtract(operations=ops, handoffs=[])
        ideal = _build_idealized_extract(extract)
        # i32 compare is Scalar-only → unchanged
        assert ideal.operations[0].component == Component.SCALAR

    def test_avoidable_handoff_removed(self):
        """Same-path handoff (Cube→Cube) removed in idealized."""
        ops = [
            OpRecord(op_id=1, op_name="matmul", component=Component.CUBE,
                     precision=Precision.FP16, pipe="Cube",
                     flops=1000, loop_multiplier=1, depends_on=[]),
        ]
        handoffs = [
            HandoffRecord(1, 1, Component.CUBE, Component.CUBE, 1024),
        ]
        extract = HIVMExtract(operations=ops, handoffs=handoffs)
        ideal = _build_idealized_extract(extract)
        # Same-path handoff removed
        assert len(ideal.handoffs) == 0

    def test_mandatory_handoff_kept(self):
        """Cross-path handoff (Cube→Vector) kept in idealized."""
        ops = [
            OpRecord(op_id=1, op_name="matmul", component=Component.CUBE,
                     precision=Precision.FP16, pipe="Cube",
                     flops=1000, loop_multiplier=1, depends_on=[]),
        ]
        handoffs = [
            HandoffRecord(1, 1, Component.CUBE, Component.VECTOR, 1024),
        ]
        extract = HIVMExtract(operations=ops, handoffs=handoffs)
        ideal = _build_idealized_extract(extract)
        # Cross-path handoff kept
        assert len(ideal.handoffs) == 1


class TestTwoLimitRecomputation:
    """compute_two_limit recomputes bound from idealized extract."""

    def test_gap1_kernel_hivm_lower_than_dsl(self):
        """Kernel with Gap-1 mis-placement → T_bound_HIVM < T_bound_DSL.

        This exercises the two-limit Gap-1 *mechanism* (idealizing a Scalar-
        placed op onto Vector lowers the bound), which requires a distinct,
        slower-than-vector scalar rate.  The default DB leaves scalar UNMEASURED
        (US-SB-007 open) and falls back to the Vector rate for soundness, so we
        opt this synthetic mechanism test into the derived/measured scalar rate
        explicitly.
        """
        db = load_default_calib_db()
        db.vector.scalar_throughput_measured = True  # exercise the slow scalar path

        # Create a kernel with a genuine Gap-1 mis-placement:
        # FP16 compare on Scalar (should be on Vector) plus Cube work
        # The Scalar fallback is the canonical Gap-1 case.
        ops = [
            OpRecord(op_id=1, op_name="cmp_fp16", component=Component.SCALAR,
                     precision=Precision.FP16, pipe="Scalar",
                     elements=1024 * 1024, flops=0, loop_multiplier=1, depends_on=[]),
            OpRecord(op_id=2, op_name="matmul", component=Component.CUBE,
                     precision=Precision.FP16, pipe="Cube",
                     bytes_transferred=0, elements=128 * 64,
                     flops=2 * 128 * 64 * 32, loop_multiplier=32, depends_on=[]),
        ]
        # The cmp_fp16 is mis-placed: it's on Scalar but eligible for Vector
        extract = HIVMExtract(operations=ops, handoffs=[],
                              unit_assignment={1: "scalar", 2: "cube"})

        grid_info = _make_grid()
        pieces = compute_bounds(grid_info, extract, db)
        result = combine(pieces.grid, pieces.component, pieces.serial,
                         kernel_name="gap1_test", extract=extract)

        two_limit = compute_two_limit(
            kernel_name="gap1_test",
            grid_info=grid_info,
            extract=extract,
            calib_db=db,
            t_bound_dsl_us=result.t_bound_us,
        )
        # compiler_headroom ≥ 0 by construction
        assert two_limit.compiler_headroom_us >= 0.0
        # T_bound_HIVM ≤ T_bound_DSL (relaxation can only lower)
        assert two_limit.t_bound_hivm_us <= two_limit.t_bound_dsl_us
        # STRICT inequality: idealizing Scalar→Vector should lower the bound
        # This is the canonical Gap-1 case: Scalar fallback is slower than Vector
        assert two_limit.t_bound_hivm_us < two_limit.t_bound_dsl_us
        # Verify headroom is meaningful (not just numerical noise)
        assert two_limit.compiler_headroom_us > 0.01  # >0.01 us, not ~0

    def test_already_at_bound_zero_headroom(self):
        """Kernel with no mis-placements or avoidable handoffs → headroom ≈ 0."""
        db = load_default_calib_db()

        # Clean kernel: all ops on correct units, no avoidable handoffs
        ops = [
            OpRecord(op_id=1, op_name="matmul", component=Component.CUBE,
                     precision=Precision.FP16, pipe="Cube",
                     bytes_transferred=0, elements=128 * 64,
                     flops=2 * 128 * 64 * 32, loop_multiplier=32, depends_on=[]),
        ]
        extract = HIVMExtract(operations=ops, handoffs=[],
                              unit_assignment={1: "cube"})

        grid_info = _make_grid()
        pieces = compute_bounds(grid_info, extract, db)
        result = combine(pieces.grid, pieces.component, pieces.serial,
                         kernel_name="at_bound_test", extract=extract)

        two_limit = compute_two_limit(
            kernel_name="at_bound_test",
            grid_info=grid_info,
            extract=extract,
            calib_db=db,
            t_bound_dsl_us=result.t_bound_us,
        )
        # No gaps → idealized = realized → headroom ≈ 0
        assert two_limit.compiler_headroom_us == pytest.approx(0.0, abs=1e-6)

    def test_author_headroom_none_without_measured(self):
        """Without T_measured, author_headroom is None."""
        db = load_default_calib_db()
        ops = [
            OpRecord(op_id=1, op_name="matmul", component=Component.CUBE,
                     precision=Precision.FP16, pipe="Cube",
                     flops=100000, loop_multiplier=1, depends_on=[]),
        ]
        extract = HIVMExtract(operations=ops, handoffs=[])
        grid_info = _make_grid()
        pieces = compute_bounds(grid_info, extract, db)
        result = combine(pieces.grid, pieces.component, pieces.serial,
                         kernel_name="no_measured", extract=extract)

        two_limit = compute_two_limit(
            kernel_name="no_measured",
            grid_info=grid_info,
            extract=extract,
            calib_db=db,
            t_bound_dsl_us=result.t_bound_us,
        )
        assert two_limit.author_headroom_us is None
