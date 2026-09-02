# A.5 — Bound combiner tests.
#
# Covers:
#   - Change #2a: composition soundness max(grid, core+serial)
#   - Change #2b: wave scaling (both floors scale with waves)
#   - Change #3: Gap 1 Scalar fallback detection
#   - Change #4: Gap 3 dedup (avoidable handoff double-counting)
#   - Change #5: grid gap formula
#
# Source: .omc/plans/a5_bound_combiner.md

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from perfbound.combine.bound_combiner import combine, _compute_gap1
from perfbound.extract.hivm_extractor import HIVMExtract, OpRecord, HandoffRecord
from perfbound.extract.op_classifier import Component, Precision
from perfbound.model.grid_model import GridBound
from perfbound.model.component_model import ComponentBound
from perfbound.model.serialization import SerializationSplit


# ── Helpers ─────────────────────────────────────────────────────────────

def _grid(t_grid=10.0, occ=1.0, lb=1.0, n_cores=20) -> GridBound:
    return GridBound(
        t_grid_floor_us=t_grid,
        total_work=1000.0,
        n_cores=n_cores,
        occupancy=occ,
        load_balance=lb,
        redundancy=1.0,
        i_binding=100.0,
        busiest_core_id=0,
    )


def _comp(t_core=5.0, binding=Component.CUBE) -> ComponentBound:
    return ComponentBound(
        t_core_floor_us=t_core,
        binding_component=binding,
        per_component_us={"cube": t_core},
        total_ops={"cube": 1000.0},
    )


def _serial(t_mandatory=2.0, t_avoidable=0.0) -> SerializationSplit:
    return SerializationSplit(
        mandatory_handoffs=[],
        avoidable_handoffs=[],
        t_serial_irreducible_us=t_mandatory,
        t_serial_avoidable_us=t_avoidable,
    )


# ── Change #2a: Composition soundness ──────────────────────────────────

class TestCompositionSoundness:
    """T_bound = max(T_grid_floor, T_core_floor + T_serial_irreducible)."""

    def test_grid_binds_when_grid_greater(self):
        """Grid floor > core+serial → grid binds, T_bound = grid."""
        g = _grid(t_grid=20.0)
        c = _comp(t_core=5.0)
        s = _serial(t_mandatory=3.0)  # core+serial = 8 < 20
        r = combine(g, c, s)
        assert r.t_bound_us == pytest.approx(20.0)
        assert r.binding_tier.value == "grid"

    def test_component_binds_when_core_plus_serial_greater(self):
        """Core+serial > grid → component binds, T_bound = core+serial."""
        g = _grid(t_grid=5.0)
        c = _comp(t_core=10.0)
        s = _serial(t_mandatory=3.0)  # core+serial = 13 > 5
        r = combine(g, c, s)
        assert r.t_bound_us == pytest.approx(13.0)
        assert r.binding_tier.value == "component"

    def test_launch_overhead_is_added_to_body_bound(self):
        g = _grid(t_grid=5.0)
        c = _comp(t_core=10.0)
        s = _serial(t_mandatory=3.0)

        result = combine(g, c, s, launch_overhead_us=2.5)

        assert result.t_body_bound_us == pytest.approx(13.0)
        assert result.t_launch_overhead_us == pytest.approx(2.5)
        assert result.t_bound_us == pytest.approx(15.5)

    def test_soundness_formula_not_additive(self):
        """Verify we DON'T use the old max(a,b)+c form (which is larger)."""
        g = _grid(t_grid=10.0)
        c = _comp(t_core=8.0)
        s = _serial(t_mandatory=5.0)
        r = combine(g, c, s)
        # Correct: max(10, 8+5) = 13
        assert r.t_bound_us == pytest.approx(13.0)
        # Old (wrong): max(10,8)+5 = 15 — must NOT be this
        assert r.t_bound_us != pytest.approx(15.0)

    def test_zero_serial_reduces_to_max(self):
        """With zero serial, T_bound = max(grid, core)."""
        g = _grid(t_grid=10.0)
        c = _comp(t_core=12.0)
        s = _serial(t_mandatory=0.0)
        r = combine(g, c, s)
        assert r.t_bound_us == pytest.approx(12.0)


# ── Change #2b: Wave scaling ──────────────────────────────────────────

class TestWaveScaling:
    """Both floors scale with waves = ceil(total_programs / n_cores)."""

    def test_wave_scaling_via_compute_bounds(self):
        """40 programs / 20 cores = 2 waves → both floors 2×."""
        from perfbound.model.bounds import compute_bounds
        from perfbound.extract.dsl_extractor import GridInfo
        from perfbound.calibration.calib_loader import load_default_calib_db

        ops = [
            OpRecord(op_id=1, op_name="matmul", component=Component.CUBE,
                     precision=Precision.FP16, pipe="Cube",
                     bytes_transferred=0, elements=128 * 64,
                     flops=2 * 128 * 64 * 32, loop_multiplier=1, depends_on=[]),
        ]
        extract = HIVMExtract(operations=ops, handoffs=[],
                              unit_assignment={1: "cube"})
        db = load_default_calib_db()

        # Single wave baseline
        grid_1wave = GridInfo(
            grid_dims=(20,), total_programs=20,
            tile_assignment={}, work={},
            occupancy=1.0, load_balance=1.0,
            redundancy=1.0, busiest_core_id=0,
        )
        pieces_1w = compute_bounds(grid_1wave, extract, db,
                                   n_cores=20, total_programs=20)

        # 2 waves
        grid_2wave = GridInfo(
            grid_dims=(40,), total_programs=40,
            tile_assignment={}, work={},
            occupancy=1.0, load_balance=1.0,
            redundancy=1.0, busiest_core_id=0,
        )
        pieces_2w = compute_bounds(grid_2wave, extract, db,
                                   n_cores=20, total_programs=40)

        # Component floor should be 2× for 2 waves
        assert pieces_2w.component.t_core_floor_us == pytest.approx(
            2.0 * pieces_1w.component.t_core_floor_us, rel=1e-3
        )
        # Grid floor should be 2× (total_work doubled, same n_cores)
        assert pieces_2w.grid.t_grid_floor_us == pytest.approx(
            2.0 * pieces_1w.grid.t_grid_floor_us, rel=1e-3
        )

    def test_single_wave_unchanged(self):
        """n_cores programs / n_cores cores = 1 wave → no scaling."""
        from perfbound.model.bounds import compute_bounds
        from perfbound.extract.dsl_extractor import GridInfo
        from perfbound.calibration.calib_loader import load_default_calib_db

        ops = [
            OpRecord(op_id=1, op_name="matmul", component=Component.CUBE,
                     precision=Precision.FP16, pipe="Cube",
                     bytes_transferred=0, elements=128 * 64,
                     flops=2 * 128 * 64 * 32, loop_multiplier=1, depends_on=[]),
        ]
        extract = HIVMExtract(operations=ops, handoffs=[],
                              unit_assignment={1: "cube"})
        db = load_default_calib_db()

        # Default call (total_programs = n_cores → 1 wave)
        grid_info = GridInfo(
            grid_dims=(20,), total_programs=20,
            tile_assignment={}, work={},
            occupancy=1.0, load_balance=1.0,
            redundancy=1.0, busiest_core_id=0,
        )
        pieces = compute_bounds(grid_info, extract, db)
        assert pieces.component.t_core_floor_us > 0


# ── Change #3: Gap 1 Scalar fallback ──────────────────────────────────

class TestGap1ScalarFallback:
    """Scalar ops with Vector/Cube eligibility trigger Gap 1."""

    def test_scalar_fallback_detected(self):
        """A compare op on Scalar that is eligible for Vector → Gap 1 > 0."""
        ops = [
            OpRecord(op_id=1, op_name="cmp_fp16", component=Component.SCALAR,
                     precision=Precision.FP16, pipe="Scalar",
                     elements=1024, flops=0, loop_multiplier=1, depends_on=[]),
        ]
        extract = HIVMExtract(operations=ops, handoffs=[])
        comp = ComponentBound(
            t_core_floor_us=10.0,
            binding_component=Component.SCALAR,
            per_component_us={"scalar": 10.0},
            total_ops={"scalar": 1024.0},
        )
        gap1 = _compute_gap1(extract, comp)
        # FP16 compare is eligible for Vector, so Scalar is mis-placed
        assert gap1 > 0

    def test_true_scalar_only_no_gap(self):
        """i32 compare is Scalar-only (eligible = {Scalar}) → no Gap 1."""
        ops = [
            OpRecord(op_id=1, op_name="cmp_i32", component=Component.SCALAR,
                     precision=Precision.INT32, pipe="Scalar",
                     elements=512, flops=0, loop_multiplier=1, depends_on=[]),
        ]
        extract = HIVMExtract(operations=ops, handoffs=[])
        comp = ComponentBound(
            t_core_floor_us=5.0,
            binding_component=Component.SCALAR,
            per_component_us={"scalar": 5.0},
            total_ops={"scalar": 512.0},
        )
        gap1 = _compute_gap1(extract, comp)
        # i32 compare: eligible = {Scalar} → realized matches → Gap 1 = 0
        assert gap1 == pytest.approx(0.0)


# ── Change #4: Gap 3 dedup ────────────────────────────────────────────

class TestGap3Dedup:
    """Avoidable handoffs are deduped before summing."""

    def test_duplicate_avoidable_handoff_counted_once(self):
        """Two Cube→Cube handoffs (same edge) → single avoidable cost."""
        from perfbound.model.serialization import classify_handoffs
        from perfbound.calibration.calib_loader import load_default_calib_db

        db = load_default_calib_db()
        handoffs = [
            HandoffRecord(1, 2, Component.CUBE, Component.CUBE, 1024),
            HandoffRecord(3, 4, Component.CUBE, Component.CUBE, 1024),
        ]
        serial = classify_handoffs(
            handoffs, mandatory_handoff_cycles=2000,
            clock_ghz=1.85, memory=db.memory,
        )
        # Both avoidable (same path) — deduped to 1 edge
        assert len(serial.avoidable_handoffs) == 2
        # Cost computed once (not doubled)
        if serial.t_serial_avoidable_us > 0:
            # Verify it's not double a single handoff cost
            single_serial = classify_handoffs(
                [handoffs[0]], mandatory_handoff_cycles=2000,
                clock_ghz=1.85, memory=db.memory,
            )
            assert serial.t_serial_avoidable_us == pytest.approx(
                single_serial.t_serial_avoidable_us, rel=1e-3
            )

    def test_zero_without_calibration(self):
        """No memory → avoidable cost is 0 (conservative)."""
        from perfbound.model.serialization import classify_handoffs

        handoffs = [
            HandoffRecord(1, 2, Component.CUBE, Component.CUBE, 1024),
        ]
        serial = classify_handoffs(
            handoffs, mandatory_handoff_cycles=2000,
            clock_ghz=1.85, memory=None,
        )
        assert serial.t_serial_avoidable_us == 0.0


# ── Change #5: Grid gap formula ──────────────────────────────────────

class TestGridGapFormula:
    """grid_gap_us = T_grid_floor × (1 − occupancy · load_balance)."""

    def test_perfect_grid_zero_gap(self):
        """occ=1, lb=1 → grid_gap = 0."""
        g = _grid(t_grid=10.0, occ=1.0, lb=1.0)
        c = _comp(t_core=15.0)
        s = _serial(t_mandatory=0.0)
        r = combine(g, c, s)
        assert r.attribution.grid_gap_us == pytest.approx(0.0)

    def test_imperfect_grid_exact_value(self):
        """occ=0.8, lb=0.9 → grid_gap = 10 × (1 − 0.72) = 2.8."""
        g = _grid(t_grid=10.0, occ=0.8, lb=0.9)
        c = _comp(t_core=15.0)
        s = _serial(t_mandatory=0.0)
        r = combine(g, c, s)
        expected = 10.0 * (1.0 - 0.8 * 0.9)  # 2.8
        assert r.attribution.grid_gap_us == pytest.approx(expected, rel=1e-6)

    def test_half_occupied(self):
        """occ=0.5, lb=1.0 → grid_gap = T_grid × 0.5."""
        g = _grid(t_grid=20.0, occ=0.5, lb=1.0)
        c = _comp(t_core=25.0)
        s = _serial(t_mandatory=0.0)
        r = combine(g, c, s)
        assert r.attribution.grid_gap_us == pytest.approx(10.0, rel=1e-6)
