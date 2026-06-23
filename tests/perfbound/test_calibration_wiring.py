# Tests for A.1 Step 5: CalibrationDB wired into M4 T_core_floor.
#
# Covers:
#   - load_default_calib_db() returns real P0 constants
#   - compute_component_floor_from_db uses real I_c (non-zero T_core_floor)
#   - bound_from_extract auto-loads calibration end-to-end
#   - graceful I_c = 0 fallback when CalibrationDB is absent
#   - memory.bw populated from CSV companion
#   - validation accepts the complete A.1 P0 measurement set

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from perfbound.extract.hivm_extractor import HIVMExtract, OpRecord, HandoffRecord
from perfbound.extract.op_classifier import Component, Precision
from perfbound.calibration.constants import CalibrationDB, CubeConfig, VectorConfig, MemHierarchy, CoreConfig, DType
from perfbound.model.component_model import compute_component_floor, compute_component_floor_from_db


# ── Helpers ────────────────────────────────────────────────────────────────

def _matmul_extract() -> HIVMExtract:
    """Minimal matmul extract (matches test_component_model.py fixture)."""
    ops = [
        OpRecord(op_id=1, op_name="matmul", component=Component.CUBE,
                 precision=Precision.FP16, pipe="Cube",
                 bytes_transferred=0, elements=2 * 128 * 64 * 32,
                 duration_cycles=100, loop_multiplier=32, depends_on=[]),
        OpRecord(op_id=2, op_name="cube_load", component=Component.MTE_GM,
                 precision=Precision.FP16, pipe="CubeMTE2",
                 bytes_transferred=128 * 32 * 2 + 32 * 64 * 2,
                 elements=0, duration_cycles=50, loop_multiplier=32,
                 depends_on=[]),
        OpRecord(op_id=3, op_name="cube_store", component=Component.MTE_UB,
                 precision=Precision.FP16, pipe="FixPipe",
                 bytes_transferred=128 * 64 * 2, elements=0,
                 duration_cycles=50, loop_multiplier=1, depends_on=[1]),
    ]
    return HIVMExtract(operations=ops, handoffs=[], unit_assignment={
        op.op_id: op.component.value for op in ops
    })


# ── CalibrationDB loading ──────────────────────────────────────────────────

class TestCalibrationLoad:
    """load_default_calib_db returns the promoted 910B3 DB."""

    def test_load_default_returns_db(self):
        from perfbound.calibration.calib_loader import load_default_calib_db
        db = load_default_calib_db()
        assert isinstance(db, CalibrationDB)

    def test_db_version_is_v1(self):
        from perfbound.calibration.calib_loader import load_default_calib_db
        db = load_default_calib_db()
        assert db.version == "v1"

    def test_cube_fp16_throughput_populated(self):
        from perfbound.calibration.calib_loader import load_default_calib_db
        db = load_default_calib_db()
        # Real measured value, not datasheet (task spec: Cube FP16 = 5.159 TFLOPS/core)
        assert abs(db.cube.throughput[DType.FP16] - 5.159) < 0.01, \
            f"Cube FP16 throughput {db.cube.throughput[DType.FP16]:.3f} not ~5.159 TFLOPS"

    def test_bw_gm_ub_populated(self):
        from perfbound.calibration.calib_loader import load_default_calib_db
        db = load_default_calib_db()
        bw, _ = db.memory.lookup_bw("gm", "ub")
        # Chronological post-warmup sustained value.
        assert abs(bw - 86487.9) < 10, f"gm->ub BW {bw:.0f} B/us not ~86488"

    def test_all_bw_paths_populated(self):
        from perfbound.calibration.calib_loader import load_default_calib_db
        db = load_default_calib_db()
        for src, dst in [("gm", "ub"), ("ub", "gm"), ("gm", "l1"), ("l1", "l0a")]:
            bw, _ = db.memory.lookup_bw(src, dst)
            assert bw > 0, f"{src}->{dst} bandwidth is 0"

    def test_p0_constants_present(self):
        from perfbound.calibration.calib_loader import load_default_calib_db
        db = load_default_calib_db()
        required = [
            "P_cube_fp16_sustained", "BW_gm_to_ub_sustained",
            "BW_ub_to_gm_sustained", "mandatory_handoff_cost_l0c_to_gm_to_ub",
        ]
        for name in required:
            assert name in db.constants, f"Missing constant: {name}"

    def test_all_p0_measurements_validate(self):
        from perfbound.calibration.calib_loader import (
            load_default_calib_db,
            validate_calibration,
        )
        db = load_default_calib_db()

        warnings = validate_calibration(db)

        assert db.validate_p0_constants() == []
        assert not any("BW_l0c_to_gm_sustained" in w for w in warnings)
        assert not any("BW_hbm_allcore_sustained" in w for w in warnings)

    def test_mandatory_handoff_cycles_nonzero(self):
        from perfbound.calibration.calib_loader import load_default_calib_db
        db = load_default_calib_db()
        assert db.mandatory_handoff_cycles > 0, \
            f"mandatory_handoff_cycles should be > 0, got {db.mandatory_handoff_cycles}"

    def test_load_from_explicit_path(self):
        from perfbound.calibration.calib_loader import load_calibration, DEFAULT_CALIB_PATH
        db = load_calibration(DEFAULT_CALIB_PATH)
        assert isinstance(db, CalibrationDB)
        assert db.version == "v1"


# ── compute_component_floor_from_db ───────────────────────────────────────

class TestComputeFromDB:
    """compute_component_floor_from_db gives non-zero real-rate T_core_floor."""

    def test_t_core_floor_nonzero_matmul(self):
        from perfbound.calibration.calib_loader import load_default_calib_db
        db = load_default_calib_db()
        extract = _matmul_extract()
        result = compute_component_floor_from_db(extract, db)
        assert result.t_core_floor_us > 0, \
            f"T_core_floor should be > 0 with real calibration, got {result.t_core_floor_us}"

    def test_t_core_floor_mte_binds(self):
        """MTE_GM binds over Cube (5.159 TFLOPS); T=5.265 us post-recalibration."""
        from perfbound.calibration.calib_loader import load_default_calib_db
        db = load_default_calib_db()
        extract = _matmul_extract()
        result = compute_component_floor_from_db(extract, db)
        # Nominal T_mte_gm = 393216 / 86487.9 = 4.546 us; the 2026-06-23
        # bandwidth-bound packet-efficiency recalibration (η normalised to the
        # true HBM peak 1167 GB/s) penalises the 4–8 KB packets → 5.265 us.
        assert abs(result.t_core_floor_us - 5.265) < 0.02, \
            f"T_core_floor {result.t_core_floor_us:.3f} not ~5.265 us"
        assert result.binding_component == Component.MTE_GM

    def test_matches_explicit_call(self):
        """compute_component_floor_from_db must equal explicit compute_component_floor."""
        from perfbound.calibration.calib_loader import load_default_calib_db
        db = load_default_calib_db()
        extract = _matmul_extract()

        result_db = compute_component_floor_from_db(extract, db)
        result_explicit = compute_component_floor(
            extract, db.cube, db.vector, db.memory, db.core
        )
        assert abs(result_db.t_core_floor_us - result_explicit.t_core_floor_us) < 1e-9


# ── bound_from_extract ─────────────────────────────────────────────────────

class TestBoundFromExtract:
    """bound_from_extract auto-loads calibration and returns real bounds."""

    def test_auto_load_gives_nonzero_t_core_floor(self):
        from perfbound.combine.bound_combiner import bound_from_extract
        extract = _matmul_extract()
        result = bound_from_extract(extract, kernel_name="matmul_autoload")
        assert result.t_core_floor_us > 0, \
            f"T_core_floor should be > 0 with auto-loaded calibration"

    def test_auto_load_golden_t_bound(self):
        """T_bound with auto-load = T_mte_gm = 5.265 us (no serial, MTE binds).

        Was 4.522 us; the 2026-06-23 true-peak packet-efficiency recalibration
        raises the mid-size-packet MTE floor (see test_t_core_floor_mte_binds).
        """
        from perfbound.combine.bound_combiner import bound_from_extract
        extract = _matmul_extract()
        result = bound_from_extract(extract, kernel_name="matmul_autoload")
        assert abs(result.t_bound_us - 5.265) < 0.05, \
            f"T_bound {result.t_bound_us:.3f} not ~5.265 us"

    def test_explicit_db_used_over_autoload(self):
        """Passing explicit calib_db overrides auto-load path."""
        from perfbound.combine.bound_combiner import bound_from_extract
        from perfbound.calibration.calib_loader import load_default_calib_db
        db = load_default_calib_db()
        extract = _matmul_extract()
        result = bound_from_extract(extract, calib_db=db, kernel_name="explicit_db")
        assert result.t_core_floor_us > 0

    def test_binding_component_mte_with_calibration(self):
        from perfbound.combine.bound_combiner import bound_from_extract
        extract = _matmul_extract()
        result = bound_from_extract(extract)
        assert result.binding_component == Component.MTE_GM
        assert result.binding_tier.value == "component"
