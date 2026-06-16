# Tests for calibration database loading and validation
#
# Validates that calib_910b3_v1.json loads correctly and contains all required P0 constants.

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from perfbound.calibration.constants import CalibrationDB, DType


def _get_calib_path() -> Path:
    """Get the path to calib_910b3_v1.json."""
    # File is at: vTriton/perfbound/calibration/data/calib_910b3_v1.json
    # Test is at: vTriton/tests/perfbound/test_calibration_load.py
    # So go up to parents[2] (vTriton), then into perfbound/calibration/data/
    project_root = Path(__file__).parents[2]
    return project_root / "perfbound" / "calibration" / "data" / "calib_910b3_v1.json"


def test_calib_db_loads_without_error():
    """Calibration JSON loads without error."""
    calib_path = _get_calib_path()
    assert calib_path.exists(), f"Calibration file not found: {calib_path}"

    db = CalibrationDB.load(str(calib_path))
    assert db is not None
    assert db.hardware_name == "Ascend 910B3"


def test_calib_db_has_cube_constants():
    """Cube throughput constants are present and positive."""
    calib_path = _get_calib_path()
    db = CalibrationDB.load(str(calib_path))

    # Check FP16
    fp16_tflops = db.cube.throughput.get(DType.FP16, 0.0)
    assert fp16_tflops > 0, "Cube FP16 throughput must be > 0"
    assert fp16_tflops > 1, f"Cube FP16 {fp16_tflops} TFLOPS seems too low"

    # Check INT8
    int8_tflops = db.cube.throughput.get(DType.INT8, 0.0)
    assert int8_tflops > 0, "Cube INT8 throughput must be > 0"
    assert int8_tflops > 1, f"Cube INT8 {int8_tflops} TFLOPS seems too low"


def test_calib_db_has_core_config():
    """Core configuration matches 910B3 specs."""
    calib_path = _get_calib_path()
    db = CalibrationDB.load(str(calib_path))

    assert db.core.aic_core_num == 20
    assert db.core.aiv_core_num == 40
    assert db.core.clock_freq_ghz == 1.85
    assert db.core.cv_bind is True


def test_calib_db_has_memory_hierarchy():
    """Memory hierarchy sizes match 910B3 specs."""
    calib_path = _get_calib_path()
    db = CalibrationDB.load(str(calib_path))

    assert db.memory.gm_size_gb == 32.0
    assert db.memory.l2_size_mb == 192.0
    assert db.memory.l1_size_kb == 1024.0
    assert db.memory.ub_size_kb == 256.0


def test_calib_db_has_p0_constant_provenance():
    """P0 constants carry provenance metadata (value, ci_95, source, n_runs)."""
    calib_path = _get_calib_path()
    db = CalibrationDB.load(str(calib_path))

    # Check that at least one constant has full provenance
    const_name = "P_cube_fp16_sustained"
    assert const_name in db.constants, f"Missing constant: {const_name}"

    const = db.constants[const_name]
    assert const.value > 0
    assert const.ci_95 >= 0
    assert const.ci_95 / const.value < 0.025
    assert const.source == "cce_microbench"
    assert const.n_runs >= 30  # P0 requires ≥30 runs


def test_calib_db_mandatory_handoff_cost():
    """Mandatory handoff cost is present and positive."""
    calib_path = _get_calib_path()
    db = CalibrationDB.load(str(calib_path))

    assert db.mandatory_handoff_cycles > 0, "Mandatory handoff cost must be > 0"
    assert db.mandatory_handoff_cycles > 1000, "Mandatory handoff cost seems too low (< 1000 cycles)"


def test_calib_db_loads_without_zero_p0():
    """All P0 constants have value > 0 (no unmeasured stubs)."""
    calib_path = _get_calib_path()
    db = CalibrationDB.load(str(calib_path))

    assert db.validate_p0_constants() == []
    for name in ("BW_l0c_to_gm_sustained", "BW_hbm_allcore_sustained"):
        constant = db.constants[name]
        assert constant.value > 0
        assert constant.source == "cce_microbench"
        assert constant.n_runs >= 30
        assert constant.ci_95 / constant.value < 0.025


def test_calib_db_uses_measured_cce_provenance():
    """Calibration is promoted from measured CCE microbenchmarks, not stubs."""
    calib_path = _get_calib_path()
    db = CalibrationDB.load(str(calib_path))

    assert db.version == "v1"
    assert db.constants
    assert all(v.source != "pending_measurement" for v in db.constants.values())
    measured = db.constants
    assert all(const.source == "cce_microbench" for const in measured.values())
    assert all(const.ci_95 / const.value < 0.025 for const in measured.values() if const.value > 0)
    assert db.vector.scalar_throughput_measured is True
    assert db.constants["P_scalar_add_sustained"].value == pytest.approx(0.5998078861614121)
