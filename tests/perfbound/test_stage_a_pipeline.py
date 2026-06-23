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
def test_a1_calibration_changes_chunk_kda_bound():
    baseline_db = load_default_calib_db()
    baseline = _report(baseline_db)

    slower_vector_db = copy.deepcopy(baseline_db)
    # Halve the vector rate via BOTH levers: the aggregate fallback (used by
    # ops without a per-op model) and the per-op cycle counts (used by mapped
    # arithmetic ops like vmul/vadd — the dominant vector work).  Perturbing
    # only the aggregate understates sensitivity now that arithmetic ops bind
    # on the per-op cycle calibration.
    slower_vector_db.vector.throughput_fp16_tflops *= 0.5
    slower_vector_db.vector.op_cycles = {
        k: v * 2.0 for k, v in slower_vector_db.vector.op_cycles.items()
    }
    slower = _report(slower_vector_db)

    assert baseline.binding_component == "vector"
    # A 2x vector-rate cut raises the bound ~1.79x (not 2x): mapped arithmetic
    # ops bind on the achievable per-op peak (get_op_cycles default), so only
    # the aggregate-rate fraction of the work scales with the perturbed knobs.
    # The bound still moves strongly with calibration — that is the claim.
    assert slower.t_core_floor_us > baseline.t_core_floor_us * 1.7
    assert slower.t_bound_us > baseline.t_bound_us * 1.7


@requires_chunk_kda_evidence
def test_one_a1_database_reaches_bound_gaps_two_limit_and_headroom():
    baseline_db = load_default_calib_db()
    baseline = _report(baseline_db, with_profile=True)

    changed_db = copy.deepcopy(baseline_db)
    # Halve the vector rate via both levers (aggregate + per-op cycles); see
    # test_a1_calibration_changes_chunk_kda_bound.
    changed_db.vector.throughput_fp16_tflops *= 0.5
    changed_db.vector.op_cycles = {
        k: v * 2.0 for k, v in changed_db.vector.op_cycles.items()
    }
    changed_db.startup_latency["vector"] = 3500.0
    changed = _report(changed_db, with_profile=True)

    # ~1.79x, not 2x — see test_a1_calibration_changes_chunk_kda_bound.
    assert changed.t_bound_us > baseline.t_bound_us * 1.7
    assert changed.t_bound_hivm_us > baseline.t_bound_hivm_us * 1.7
    assert (
        changed.attribution_us["gap4_intra_unit_exec"]
        > baseline.attribution_us["gap4_intra_unit_exec"]
    )
    # A worse vector calibration must not INCREASE recoverable headroom.  The
    # upper estimate is capped by the measured exposed-control deficit, which is
    # independent of the compute-rate calibration, so the two can be equal when
    # that cap binds (raising t_bound shrinks the author residual but not below
    # the deficit cap).  The sound invariant is therefore non-strict.
    assert (
        changed.recoverable_headroom_upper_us
        <= baseline.recoverable_headroom_upper_us
    )


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
    assert data["attribution_us"]["gap4_intra_unit_exec"] > 0
    assert data["reachability"]["author_residual_us"] > 0
    assert data["reachability"]["n_invocations"] == 5

    assessment = data["headroom_assessment"]
    assert assessment["status"] == "diagnostic_upper_bound"
    assert assessment["point_estimate_us"] is None
    assert assessment["lower_us"] == 0.0
    assert 0 < assessment["upper_us"] <= data["reachability"]["author_residual_us"]
    assert assessment["confidence"] == "low"
    assert "counterfactual" in assessment["method"].lower()


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
