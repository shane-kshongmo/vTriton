# Tests for report.py three-level rendering (A.6.1)
#
# Validates KernelReport text rendering with Reachability Hierarchy,
# bound violation labels, component match, and to_dict reachability block.
#
# Source spec: .omc/plans/a6_validation_harness.md §7

import pytest

from perfbound.combine.report import KernelReport
from perfbound.combine.bound_combiner import BoundResult, BindingTier, Attribution
from perfbound.combine.two_limit import TwoLimitResult
from perfbound.extract.op_classifier import Component


def _make_bound_result(
    kernel_name: str = "test_kernel",
    t_bound_us: float = 1000.0,
    binding_component: Component = Component.CUBE,
) -> BoundResult:
    return BoundResult(
        kernel_name=kernel_name,
        t_bound_us=t_bound_us,
        t_grid_floor_us=800.0,
        t_core_floor_us=900.0,
        t_serial_irreducible_us=100.0,
        binding_tier=BindingTier.COMPONENT,
        binding_component=binding_component,
        attribution=Attribution(),
    )


def test_author_headroom_flows_through():
    """t_measured_us=5000.0 → KernelReport.author_headroom_us correct."""
    br = _make_bound_result(t_bound_us=1000.0)
    two_limit = TwoLimitResult(
        kernel_name="test_kernel",
        t_bound_hivm_us=800.0,
        t_bound_dsl_us=1000.0,
        t_measured_us=5000.0,
    )
    report = KernelReport.from_bound(br, two_limit=two_limit)
    # author_headroom = t_measured - t_bound_dsl = 5000 - 1000 = 4000
    assert report.author_headroom_us == 4000.0


def test_event_elapsed_is_not_used_as_author_headroom():
    br = _make_bound_result(t_bound_us=1000.0)
    report = KernelReport.from_bound(
        br,
        two_limit=TwoLimitResult(
            kernel_name="test_kernel",
            t_bound_hivm_us=800.0,
            t_bound_dsl_us=1000.0,
        ),
    )

    report.merge_event_elapsed(5000.0, source="triton.testing.do_bench")

    assert report.event_elapsed_us == 5000.0
    assert report.measurement_metric == "event_elapsed"
    assert report.t_measured_us is None
    assert report.author_headroom_us is None


def test_incomplete_model_coverage_suppresses_headroom_and_recommendation():
    br = _make_bound_result(t_bound_us=1000.0)
    report = KernelReport.from_bound(
        br,
        two_limit=TwoLimitResult(
            kernel_name="test_kernel",
            t_bound_hivm_us=800.0,
            t_bound_dsl_us=1000.0,
            t_measured_us=1200.0,
        ),
    )

    report.merge_model_coverage({
        "status": "conservative_partial",
        "outlined_calls": 2,
        "summarized_outlined_calls": 2,
        "zero_byte_transfers": 0,
    })

    assert report.model_coverage["status"] == "conservative_partial"
    assert report.compiler_headroom_us is None
    assert report.author_headroom_us is None
    assert report.headroom_status == "model_incomplete"
    assert "incomplete" in report.recommended_action.lower()


def test_modeling_output_ranks_opportunities_and_reports_sound_ceilings():
    bound = _make_bound_result(t_bound_us=1000.0)
    bound.attribution = Attribution(
        gap1_wrong_unit_us=20.0,
        gap1_frac=0.02,
        gap2_coalescing_us=40.0,
        gap2_frac=0.04,
    )
    report = KernelReport.from_bound(
        bound,
        two_limit=TwoLimitResult(
            kernel_name="test_kernel",
            t_bound_hivm_us=800.0,
            t_bound_dsl_us=1000.0,
            t_measured_us=5000.0,
        ),
    )
    report.merge_model_coverage({
        "status": "complete",
        "trace_timing_status": "partial",
        "semantic_overlay": {"applied": True, "complete": True},
    })
    report.calibration_p0_complete = True

    output = report.to_dict()["modeling_output"]

    assert output["status"] == "sound_theoretical_ceiling"
    assert output["basis"] == [
        "semantic_ir", "hivm_ir", "hardware_profile_values",
        "perf_bound_theory"
    ]
    assert output["profile_inputs"]["task_duration_us"] == 5000.0
    assert output["profile_inputs"]["measurement_metric"] == (
        "msprof_task_duration"
    )
    assert output["simulator_trace"] == {
        "role": "calibration_and_validation_only",
        "used_as_kernel_model_input": False,
    }
    assert output["compiler_floor_shift_us"] == pytest.approx(200.0)
    assert output["theoretical_ceilings"] == {
        "to_realized_dsl_floor_us": 4000.0,
        "to_idealized_hivm_floor_us": 4200.0,
        "speedup_to_realized_dsl_floor_upper": 5.0,
        "speedup_to_idealized_hivm_floor_upper": 6.25,
    }
    assert [item["name"] for item in output["opportunity_ranking"]] == [
        "gap2_coalescing", "gap1_wrong_unit"
    ]
    assert "not additive" in output["opportunity_ranking_semantics"]
    assert output["achievable_headroom"]["status"] == "not_established"
    assert output["achievable_headroom"]["point_estimate_us"] is None
    assert output["validity_gates"]["modeled_trace_timing_status"] == "partial"


def test_modeling_output_suppresses_ranking_and_ceilings_when_incomplete():
    report = KernelReport.from_bound(
        _make_bound_result(t_bound_us=1000.0),
        two_limit=TwoLimitResult(
            kernel_name="test_kernel",
            t_bound_hivm_us=800.0,
            t_bound_dsl_us=1000.0,
            t_measured_us=1200.0,
        ),
    )
    report.merge_model_coverage({
        "status": "conservative_partial",
        "trace_timing_status": "partial",
        "semantic_overlay": {"applied": True, "complete": False},
    })

    output = report.to_dict()["modeling_output"]

    assert output["status"] == "model_incomplete"
    assert output["opportunity_ranking"] == []
    assert all(
        value is None for value in output["theoretical_ceilings"].values()
    )


def test_modeling_output_requires_task_measurement_for_ceiling():
    report = KernelReport.from_bound(
        _make_bound_result(t_bound_us=1000.0),
        two_limit=TwoLimitResult(
            kernel_name="test_kernel",
            t_bound_hivm_us=800.0,
            t_bound_dsl_us=1000.0,
        ),
    )
    report.merge_model_coverage({
        "status": "complete",
        "trace_timing_status": "complete",
        "semantic_overlay": {"applied": True, "complete": True},
    })
    report.calibration_p0_complete = True

    output = report.to_dict()["modeling_output"]

    assert output["status"] == "measurement_required"
    assert "hardware_profile_values" not in output["basis"]
    assert output["compiler_floor_shift_us"] == pytest.approx(200.0)
    assert all(
        value is None for value in output["theoretical_ceilings"].values()
    )


def test_modeling_output_suppresses_ceiling_on_bound_violation():
    report = KernelReport.from_bound(
        _make_bound_result(t_bound_us=1000.0),
        two_limit=TwoLimitResult(
            kernel_name="test_kernel",
            t_bound_hivm_us=800.0,
            t_bound_dsl_us=1000.0,
            t_measured_us=900.0,
        ),
    )
    report.merge_model_coverage({
        "status": "complete",
        "trace_timing_status": "complete",
        "semantic_overlay": {"applied": True, "complete": True},
    })
    report.calibration_p0_complete = True

    output = report.to_dict()["modeling_output"]

    assert output["status"] == "bound_violation"
    assert output["validity_gates"]["bound_order_valid"] is False
    assert output["opportunity_ranking"] == []
    assert all(
        value is None for value in output["theoretical_ceilings"].values()
    )


def test_modeling_output_is_prominent_in_text_report():
    report = KernelReport.from_bound(
        _make_bound_result(t_bound_us=1000.0),
        two_limit=TwoLimitResult(
            kernel_name="test_kernel",
            t_bound_hivm_us=800.0,
            t_bound_dsl_us=1000.0,
            t_measured_us=1200.0,
        ),
    )
    report.merge_model_coverage({
        "status": "complete",
        "trace_timing_status": "complete",
        "semantic_overlay": {"applied": True, "complete": True},
    })
    report.calibration_p0_complete = True

    text = report.to_text()

    assert "Modeling Output" in text
    assert "rank optimization opportunities" in text
    assert "sound theoretical ceilings" in text
    assert "hardware_profile_values" in text
    assert "not a kernel model input" in text
    assert "achievable point estimate: not established" in text


def test_modeling_output_requires_known_complete_calibration():
    report = KernelReport.from_bound(
        _make_bound_result(t_bound_us=1000.0),
        two_limit=TwoLimitResult(
            kernel_name="test_kernel",
            t_bound_hivm_us=800.0,
            t_bound_dsl_us=1000.0,
            t_measured_us=1200.0,
        ),
    )
    report.merge_model_coverage({
        "status": "complete",
        "trace_timing_status": "complete",
        "semantic_overlay": {"applied": True, "complete": True},
    })

    output = report.to_dict()["modeling_output"]

    assert output["status"] == "calibration_unknown"
    assert all(
        value is None for value in output["theoretical_ceilings"].values()
    )


def test_modeling_output_requires_explicit_task_measurement_provenance():
    report = KernelReport.from_bound(
        _make_bound_result(t_bound_us=1000.0),
        two_limit=TwoLimitResult(
            kernel_name="test_kernel",
            t_bound_hivm_us=800.0,
            t_bound_dsl_us=1000.0,
            t_measured_us=1200.0,
        ),
    )
    report.merge_model_coverage({
        "status": "complete",
        "trace_timing_status": "complete",
        "semantic_overlay": {"applied": True, "complete": True},
    })
    report.calibration_p0_complete = True
    report.measurement_metric = None

    output = report.to_dict()["modeling_output"]

    assert output["status"] == "task_measurement_required"
    assert all(
        value is None for value in output["theoretical_ceilings"].values()
    )


def test_to_text_three_levels():
    """Reachability Hierarchy section present in text output."""
    br = _make_bound_result()
    two_limit = TwoLimitResult(
        kernel_name="test_kernel",
        t_bound_hivm_us=800.0,
        t_bound_dsl_us=1000.0,
    )
    report = KernelReport.from_bound(br, two_limit=two_limit)
    text = report.to_text()
    assert "Reachability Hierarchy" in text


def test_to_text_labels_author_value_as_residual_not_attainable_headroom():
    br = _make_bound_result(t_bound_us=1000.0)
    two_limit = TwoLimitResult(
        kernel_name="test_kernel",
        t_bound_hivm_us=800.0,
        t_bound_dsl_us=1000.0,
        t_measured_us=5000.0,
    )
    report = KernelReport.from_bound(br, two_limit=two_limit)
    text = report.to_text()
    assert "author residual" in text.lower()
    assert "not proven attainable" in text.lower()


def test_to_text_not_measured():
    """not yet measured when t_measured_us=None."""
    br = _make_bound_result()
    two_limit = TwoLimitResult(
        kernel_name="test_kernel",
        t_bound_hivm_us=800.0,
        t_bound_dsl_us=1000.0,
    )
    report = KernelReport.from_bound(br, two_limit=two_limit)
    text = report.to_text()
    assert "not yet measured" in text


def test_to_text_bound_violation():
    """BOUND VIOLATION when T_bound > T_measured."""
    br = _make_bound_result(t_bound_us=1500.0)
    two_limit = TwoLimitResult(
        kernel_name="test_kernel",
        t_bound_hivm_us=1200.0,
        t_bound_dsl_us=1500.0,
        t_measured_us=1100.0,  # T_bound > T_measured
    )
    report = KernelReport.from_bound(br, two_limit=two_limit)
    text = report.to_text()
    assert "BOUND VIOLATION" in text


def test_to_text_shows_source_and_n_invocations():
    """Source path + n=N invocations shown when measured."""
    br = _make_bound_result()
    two_limit = TwoLimitResult(
        kernel_name="test_kernel",
        t_bound_hivm_us=800.0,
        t_bound_dsl_us=1000.0,
        t_measured_us=1200.0,
    )
    report = KernelReport.from_bound(br, two_limit=two_limit)
    report.msprof_source = "/tmp/op_summary.csv"
    report.n_invocations = 12
    text = report.to_text()
    assert "source: /tmp/op_summary.csv" in text
    assert "n=12 invocations" in text


def test_to_text_shows_component_match():
    """match=✓ / match=✗ rendered when component_match is set."""
    br = _make_bound_result(binding_component=Component.CUBE)
    two_limit = TwoLimitResult(
        kernel_name="test_kernel",
        t_bound_hivm_us=800.0,
        t_bound_dsl_us=1000.0,
        t_measured_us=1200.0,
    )
    report = KernelReport.from_bound(br, two_limit=two_limit)
    report.component_match = True
    text = report.to_text()
    assert "match=✓" in text

    report.component_match = False
    text = report.to_text()
    assert "match=✗" in text


def test_to_dict_reachability_key():
    """to_dict()[reachability][t_bound_dsl_us] present."""
    br = _make_bound_result()
    two_limit = TwoLimitResult(
        kernel_name="test_kernel",
        t_bound_hivm_us=800.0,
        t_bound_dsl_us=1000.0,
    )
    report = KernelReport.from_bound(br, two_limit=two_limit)
    d = report.to_dict()
    assert "reachability" in d
    assert d["reachability"]["t_bound_dsl_us"] == 1000.0
    assert "author_residual_us" in d["reachability"]


def test_to_dict_is_violation_flag():
    """is_violation=True when T_bound > T_measured."""
    br = _make_bound_result(t_bound_us=1500.0)
    two_limit = TwoLimitResult(
        kernel_name="test_kernel",
        t_bound_hivm_us=1200.0,
        t_bound_dsl_us=1500.0,
        t_measured_us=1100.0,
    )
    report = KernelReport.from_bound(br, two_limit=two_limit)
    d = report.to_dict()
    assert d["reachability"]["is_violation"] is True


def test_to_dict_msprof_source_and_n_invocations():
    """reachability[msprof_source] and reachability[n_invocations] present."""
    br = _make_bound_result()
    two_limit = TwoLimitResult(
        kernel_name="test_kernel",
        t_bound_hivm_us=800.0,
        t_bound_dsl_us=1000.0,
        t_measured_us=1200.0,
    )
    report = KernelReport.from_bound(br, two_limit=two_limit)
    report.msprof_source = "/tmp/op_summary.csv"
    report.n_invocations = 12
    d = report.to_dict()
    assert d["reachability"]["msprof_source"] == "/tmp/op_summary.csv"
    assert d["reachability"]["n_invocations"] == 12


# ── merge_validation bridge tests ──────────────────────────────────────


def test_merge_validation_sets_provenance_fields():
    """merge_validation copies t_measured, msprof_source, n_invocations, component_match."""
    br = _make_bound_result()
    two_limit = TwoLimitResult(
        kernel_name="test_kernel",
        t_bound_hivm_us=800.0,
        t_bound_dsl_us=1000.0,
    )
    report = KernelReport.from_bound(br, two_limit=two_limit)
    assert report.msprof_source is None
    assert report.n_invocations is None
    assert report.component_match is None

    report.merge_validation(
        t_measured_us=1500.0,
        msprof_source="/tmp/op_summary.csv",
        n_invocations=5,
        component_match=True,
    )

    assert report.t_measured_us == 1500.0
    assert report.msprof_source == "/tmp/op_summary.csv"
    assert report.n_invocations == 5
    assert report.component_match is True
    # author_headroom = t_measured - t_bound_dsl = 1500 - 1000 = 500
    assert report.author_headroom_us == 500.0


# ── merge_profile tests ────────────────────────────────────────────────


def _make_mock_profile(
    diagnosis: str = "Insufficient Parallelism",
    n_sync_ops: int = 402,
    exposed_control_deficit_pts: float = 0.727,
    exposed_control_deficit_us: float = 58216.0,
    exposed_control_frac_measured: float = 0.846,
    exposed_control_frac_model: float = 0.119,
):
    from perfbound.extract.op_classifier import Component
    from types import SimpleNamespace
    return SimpleNamespace(
        diagnosis=diagnosis,
        dominant_component=Component.SCALAR,
        n_sync_ops=n_sync_ops,
        exposed_control_deficit_pts=exposed_control_deficit_pts,
        exposed_control_deficit_us=exposed_control_deficit_us,
        exposed_control_frac_measured=exposed_control_frac_measured,
        exposed_control_frac_model=exposed_control_frac_model,
    )


def test_merge_profile_overrides_recommendation_when_headroom_large():
    """merge_profile overrides recommended_action when author headroom >15% of T_measured."""
    br = _make_bound_result(t_bound_us=46110.0)
    two_limit = TwoLimitResult(
        kernel_name="chunk_kda",
        t_bound_hivm_us=40000.0,
        t_bound_dsl_us=46110.0,
        t_measured_us=80000.0,
    )
    report = KernelReport.from_bound(br, two_limit=two_limit)
    report.merge_profile(_make_mock_profile())
    assert "402" in report.recommended_action
    assert "barrier" in report.recommended_action.lower() or "sync" in report.recommended_action.lower()
    assert "tile" in report.recommended_action.lower()
    assert "scalar-throughput" in report.recommended_action.lower()


def test_merge_profile_no_override_when_headroom_small():
    """merge_profile does NOT override recommended_action when headroom ≤15%."""
    br = _make_bound_result(t_bound_us=1000.0)
    two_limit = TwoLimitResult(
        kernel_name="test_kernel",
        t_bound_hivm_us=850.0,
        t_bound_dsl_us=1000.0,
        t_measured_us=1050.0,  # headroom = 50 = 4.8% of T_measured
    )
    report = KernelReport.from_bound(br, two_limit=two_limit)
    original_action = report.recommended_action
    report.merge_profile(_make_mock_profile())
    assert report.recommended_action == original_action


def test_merge_profile_populates_fields():
    """merge_profile sets all profile fields regardless of headroom threshold."""
    br = _make_bound_result()
    two_limit = TwoLimitResult(
        kernel_name="test_kernel",
        t_bound_hivm_us=800.0,
        t_bound_dsl_us=1000.0,
        t_measured_us=5000.0,
    )
    report = KernelReport.from_bound(br, two_limit=two_limit)
    report.merge_profile(_make_mock_profile())
    assert report.profile_diagnosis == "Insufficient Parallelism"
    assert report.profile_dominant_component == "scalar"
    assert report.n_sync_ops == 402
    assert report.exposed_control_deficit_pts == pytest.approx(0.727)
    assert report.headroom_status == "diagnostic_upper_bound"
    assert report.recoverable_headroom_lower_us == 0.0
    assert report.recoverable_headroom_upper_us == pytest.approx(4000.0)
    assert report.recoverable_headroom_estimate_us is None
    assert report.headroom_confidence == "low"


def test_to_text_shows_profile_section():
    """to_text includes Profile Diagnosis section after merge_profile."""
    br = _make_bound_result()
    two_limit = TwoLimitResult(
        kernel_name="test_kernel",
        t_bound_hivm_us=800.0,
        t_bound_dsl_us=1000.0,
        t_measured_us=5000.0,
    )
    report = KernelReport.from_bound(br, two_limit=two_limit)
    report.merge_profile(_make_mock_profile())
    text = report.to_text()
    assert "Profile Diagnosis" in text
    assert "Insufficient Parallelism" in text


def test_to_dict_includes_profile_block():
    """to_dict includes profile key after merge_profile, None before."""
    br = _make_bound_result()
    two_limit = TwoLimitResult(
        kernel_name="test_kernel",
        t_bound_hivm_us=800.0,
        t_bound_dsl_us=1000.0,
    )
    report = KernelReport.from_bound(br, two_limit=two_limit)
    assert report.to_dict().get("profile") is None
    report.merge_profile(_make_mock_profile())
    d = report.to_dict()
    assert d["profile"]["diagnosis"] == "Insufficient Parallelism"
    assert d["profile"]["n_sync_ops"] == 402
    assert d["headroom_assessment"]["status"] == "unavailable"
    assert d["headroom_assessment"]["point_estimate_us"] is None


def test_merge_calibration_populates_provenance():
    from perfbound.calibration.calib_loader import load_default_calib_db

    report = KernelReport.from_bound(_make_bound_result())
    report.merge_calibration(
        load_default_calib_db(),
        "perfbound/calibration/data/calib_910b3_v1.json",
    )
    data = report.to_dict()["calibration"]

    assert data["version"] == "v1"
    assert data["hardware_name"] == "Ascend 910B3"
    assert data["measured_constant_count"] >= 10
    assert data["max_measured_ci_rel"] < 0.05
    assert any("scalar_overhead_factor" in item for item in data["warnings"])
    assert any("Gap 4 startup latency" in item for item in data["fallbacks"])
