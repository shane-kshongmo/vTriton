# A.4 — compute_bounds driver tests.
#
# Verifies that compute_bounds picks (i_binding, total_work) correctly:
#   - memory-bound (MTE binds): i_binding = BW in B/us, total_work = bytes
#   - compute-bound (Cube binds): i_binding = FLOP/us, total_work = flops
#
# Source: .omc/plans/a4_two_analytical_models.md Change #3

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from perfbound.extract.hivm_extractor import HIVMExtract, OpRecord
from perfbound.extract.op_classifier import Component, Precision
from perfbound.extract.dsl_extractor import GridInfo
from perfbound.calibration.calib_loader import load_default_calib_db
from perfbound.model.bounds import compute_bounds


def _make_grid(n_cores=20, occupancy=1.0, load_balance=1.0, total_programs=1) -> GridInfo:
    return GridInfo(
        grid_dims=(n_cores,), total_programs=total_programs,
        tile_assignment={}, work={},
        occupancy=occupancy, load_balance=load_balance,
        redundancy=1.0, busiest_core_id=0,
    )


class TestComputeBoundsMemoryBound:
    """MTE-bound kernel: grid must use BW (B/us) and total bytes."""

    def test_memory_bound_grid_uses_bytes(self):
        """MTE-bound: i_binding is BW in B/us, total_work is bytes."""
        ops = [
            OpRecord(op_id=1, op_name="matmul", component=Component.CUBE,
                     precision=Precision.FP16, pipe="Cube",
                     bytes_transferred=0, elements=128 * 64,
                     flops=2 * 128 * 64 * 32, loop_multiplier=32, depends_on=[]),
            OpRecord(op_id=2, op_name="load", component=Component.MTE_GM,
                     precision=Precision.FP16, pipe="CubeMTE2",
                     bytes_transferred=393216, elements=0,
                     loop_multiplier=1, depends_on=[]),
        ]
        extract = HIVMExtract(operations=ops, handoffs=[],
                              unit_assignment={1: "cube", 2: "mte_gm"})
        db = load_default_calib_db()
        grid = _make_grid()
        result = compute_bounds(grid, extract, db,
                                n_cores=20, total_programs=1)

        # MTE_GM should bind at component level
        assert result.component.binding_component == Component.MTE_GM
        # Grid i_binding should be BW (B/us), not FLOP/us
        assert result.grid.i_binding < 1e6, \
            f"Expected BW in B/us (<1e6), got {result.grid.i_binding}"
        # Grid total_work should be bytes (393216), not flops
        assert result.grid.total_work == pytest.approx(393216.0, rel=1e-3)

    def test_allcore_hbm_rate_only_applies_to_mte_gm(self):
        ops = [
            OpRecord(
                op_id=1, op_name="store", component=Component.MTE_UB,
                precision=Precision.FP16, pipe="MTE3",
                bytes_transferred=180_000, src_space="ub", dst_space="gm",
            ),
        ]
        extract = HIVMExtract(
            operations=ops, handoffs=[], unit_assignment={1: "mte_ub"}
        )
        db = load_default_calib_db()
        db.constants["BW_hbm_allcore_sustained"].value = 1.0

        result = compute_bounds(
            _make_grid(), extract, db, n_cores=20, total_programs=1
        )

        assert result.component.binding_component == Component.MTE_UB
        assert result.grid.i_binding > 1_000.0


class TestComputeBoundsComputeBound:
    """Cube-bound kernel: grid must use Cube throughput (FLOP/us) and total flops."""

    def test_compute_bound_grid_uses_flops(self):
        """Cube-bound: i_binding is Cube FLOP/us, total_work is flops."""
        # Cube-only kernel with minimal MTE — Cube will bind
        ops = [
            OpRecord(op_id=1, op_name="matmul", component=Component.CUBE,
                     precision=Precision.FP16, pipe="Cube",
                     bytes_transferred=0, elements=128 * 64,
                     flops=2 * 128 * 64 * 32, loop_multiplier=32, depends_on=[]),
        ]
        extract = HIVMExtract(operations=ops, handoffs=[],
                              unit_assignment={1: "cube"})
        db = load_default_calib_db()
        grid = _make_grid()
        result = compute_bounds(grid, extract, db,
                                n_cores=20, total_programs=1)

        # Cube should bind at component level
        assert result.component.binding_component == Component.CUBE
        # Grid i_binding should be Cube FLOP/us (>1e6, not BW)
        assert result.grid.i_binding > 1e6, \
            f"Expected Cube FLOP/us (>1e6), got {result.grid.i_binding}"
        # Grid total_work should be flops (2*128*64*32*32 = 16,777,216), not bytes
        expected_flops = 2 * 128 * 64 * 32 * 32  # 16,777,216
        assert result.grid.total_work == pytest.approx(expected_flops, rel=1e-3)
        # Verify grid floor is in a sensible range for FLOP/us
        assert result.grid.t_grid_floor_us > 0
        assert result.grid.t_grid_floor_us < 1.0  # should be small (lots of FLOPs / high rate)


class TestComputeBoundsIntegration:
    """End-to-end: compute_bounds output feeds into combine."""

    def test_pieces_feed_combine(self):
        """compute_bounds output can be passed directly to combine."""
        from perfbound.combine.bound_combiner import combine

        ops = [
            OpRecord(op_id=1, op_name="matmul", component=Component.CUBE,
                     precision=Precision.FP16, pipe="Cube",
                     bytes_transferred=0, elements=128 * 64,
                     flops=2 * 128 * 64 * 32, loop_multiplier=32, depends_on=[]),
            OpRecord(op_id=2, op_name="load", component=Component.MTE_GM,
                     precision=Precision.FP16, pipe="CubeMTE2",
                     bytes_transferred=393216, elements=0,
                     loop_multiplier=1, depends_on=[]),
        ]
        extract = HIVMExtract(operations=ops, handoffs=[],
                              unit_assignment={1: "cube", 2: "mte_gm"})
        db = load_default_calib_db()
        grid = _make_grid()
        pieces = compute_bounds(grid, extract, db,
                                n_cores=20, total_programs=1)

        result = combine(pieces.grid, pieces.component, pieces.serial,
                         kernel_name="test_bounds_combine", extract=extract)
        assert result.t_bound_us > 0
        assert result.binding_component == Component.MTE_GM
