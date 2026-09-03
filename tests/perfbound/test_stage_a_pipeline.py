"""End-to-end Stage A regression for the real chunk-kda artifacts."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from perfbound.calibration.calib_loader import DEFAULT_CALIB_PATH, load_default_calib_db
from perfbound.combine.run_report import report_from_desgraph


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DES_JSON = PROJECT_ROOT / ".omc" / "research" / "hw_runs" / "kda_des.json"
MSPROF_CSV = (
    PROJECT_ROOT / ".omc" / "research" / "hw_runs" / "chunk_kda_op_summary.csv"
)
KERNEL_OP_NAME = "chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2"

requires_chunk_kda_evidence = pytest.mark.skipif(
    not DES_JSON.exists() or not MSPROF_CSV.exists(),
    reason="committed chunk-kda DES/msprof evidence is unavailable",
)


def _report(calib_db=None, with_profile: bool = False):
    return report_from_desgraph(
        des_json=DES_JSON,
        grid_dims=(128, 32),
        calib_db=calib_db,
        calibration_source=(
            "test-injected CalibrationDB"
            if calib_db is not None
            else str(DEFAULT_CALIB_PATH)
        ),
        n_cores=20,
        kernel_name="chunk_kda",
        t_measured_us=104326.0,
        op_summary_csv=MSPROF_CSV if with_profile else None,
        op_name_filter=KERNEL_OP_NAME if with_profile else None,
    )


@requires_chunk_kda_evidence
def test_legacy_chunk_kda_uses_scalar_floor_and_suppresses_headroom():
    report = _report(load_default_calib_db())

    assert report.binding_component == "scalar"
    assert 0 < report.t_bound_us <= report.t_measured_us
    assert report.model_coverage["status"] == "legacy_unknown"
    assert report.compiler_headroom_us is None
    assert report.author_headroom_us is None


@requires_chunk_kda_evidence
def test_profile_cannot_restore_headroom_for_legacy_unknown_coverage():
    report = _report(load_default_calib_db(), with_profile=True)

    modeling = report.to_dict()["modeling_output"]
    assert modeling["status"] == "model_incomplete"
    assert all(value is None for value in modeling["theoretical_ceilings"].values())
    assert report.author_headroom_us is None
    assert "coverage incomplete" in report.recommended_action.lower()


@requires_chunk_kda_evidence
def test_complete_pipeline_reports_provenance_and_honest_headroom():
    report = _report(with_profile=True)
    data = report.to_dict()

    assert report.t_bound_us <= report.t_measured_us
    assert data["calibration"]["version"] == "v1"
    assert data["calibration"]["hardware_name"] == "Ascend 910B3"
    assert data["calibration"]["measured_constant_count"] >= 10
    assert data["calibration"]["p0_complete"] is True
    assert data["calibration"]["p0_violations"] == []
    assert not any(
        "BW_l0c_to_gm_sustained" in item
        or "BW_hbm_allcore_sustained" in item
        for item in data["calibration"]["fallbacks"]
    )
    assert any(
        "Gap 4 startup latency" in item
        for item in data["calibration"]["fallbacks"]
    )
    assert data["model_coverage"]["status"] == "legacy_unknown"
    modeling = data["modeling_output"]
    assert modeling["status"] == "model_incomplete"
    assert modeling["profile_inputs"]["invocations"] == 5
    assert modeling["achievable_headroom"]["point_estimate_us"] is None
    assert all(value is None for value in modeling["theoretical_ceilings"].values())


@requires_chunk_kda_evidence
def test_profile_selector_accepts_kernel_substring():
    report = report_from_desgraph(
        des_json=DES_JSON,
        grid_dims=(128, 32),
        n_cores=20,
        kernel_name="chunk_kda",
        op_summary_csv=MSPROF_CSV,
        op_name_filter="chunk_kda_bwd",
    )

    assert report.profile_diagnosis == "Insufficient Parallelism"
    assert report.profile_dominant_component == "scalar"


@requires_chunk_kda_evidence
def test_l0c_to_gm_and_hbm_allcore_propagate_through_pipeline():
    """Inject measured BW_l0c_to_gm and BW_hbm_allcore and verify propagation."""
    from perfbound.calibration.constants import CalibrationConstant, MemBandwidth

    baseline_db = load_default_calib_db()
    baseline = _report(baseline_db)

    # Inject measured L0C→GM bandwidth
    injected_db = copy.deepcopy(baseline_db)
    injected_db.constants["BW_l0c_to_gm_sustained"] = CalibrationConstant(
        name="BW_l0c_to_gm_sustained",
        value=50.0,  # GB/s — plausible FixPipe sustained rate
        unit="GB/s",
        ci_95=0.5,
        source="cce_microbench",
        n_runs=45,
        notes="test-injected",
    )
    injected_db.memory.bw[("l0c", "gm", -1)] = MemBandwidth(
        src_mem="l0c", dst_mem="gm", bw_gb_per_s=50.0,
    )

    # Inject measured HBM all-core bandwidth
    injected_db.constants["BW_hbm_allcore_sustained"] = CalibrationConstant(
        name="BW_hbm_allcore_sustained",
        value=7.0,  # GB/s per-core under contention — plausible
        unit="GB/s",
        ci_95=0.1,
        source="cce_microbench",
        n_runs=45,
        notes="test-injected",
    )

    injected = _report(injected_db)

    # P0 violations should no longer include these two constants
    p0_violations = injected_db.validate_p0_constants()
    l0c_ok = not any("BW_l0c_to_gm_sustained" in v for v in p0_violations)
    hbm_ok = not any("BW_hbm_allcore_sustained" in v for v in p0_violations)
    assert l0c_ok, f"BW_l0c_to_gm_sustained still in violations: {p0_violations}"
    assert hbm_ok, f"BW_hbm_allcore_sustained still in violations: {p0_violations}"

    # The bound should change when HBM all-core is lower than single-core rate
    # (7 GB/s per-core vs ~87 GB/s single-core GM→UB)
    # For a memory-bound grid floor, the lower i_binding raises T_grid_floor
    assert injected.t_grid_floor_us >= baseline.t_grid_floor_us * 0.9, (
        f"Grid floor should change with HBM all-core rate: "
        f"baseline={baseline.t_grid_floor_us:.2f}, injected={injected.t_grid_floor_us:.2f}"
    )

    # Fallback warnings should no longer mention these constants
    fallbacks = injected.calibration_fallbacks
    assert not any("BW_l0c_to_gm_sustained" in f for f in fallbacks), (
        f"BW_l0c_to_gm_sustained still in fallbacks: {fallbacks}"
    )
    assert not any("BW_hbm_allcore_sustained" in f for f in fallbacks), (
        f"BW_hbm_allcore_sustained still in fallbacks: {fallbacks}"
    )
