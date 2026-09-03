import copy
import json

import pytest

from perfbound.combine.report_compat import (
    with_legacy_report_aliases,
    write_report_json,
)


def _canonical_report():
    return {
        "schema_version": "perfbound_report_v2",
        "kernel_name": "kernel",
        "calibration": {"measured_constant_count": 12},
        "modeling_output": {
            "status": "sound_theoretical_ceiling",
            "bounds": {
                "hivm_floor_us": 80.0,
                "dsl_floor_us": 100.0,
            },
            "profile_inputs": {
                "task_duration_us": 150.0,
                "measurement_metric": "msprof_task_duration",
                "msprof_source": "op_summary.csv",
                "invocations": 9,
                "task_wait_us": 12.0,
                "component_match": True,
                "diagnosis": "Compute Bound",
                "dominant_component": "vector",
                "exposed_control_fraction_measured": 0.4,
                "exposed_control_fraction_model": 0.2,
                "exposed_control_deficit_points": 0.2,
                "exposed_control_deficit_us": 30.0,
                "sync_operations": 4,
            },
            "screening_metrics": {"event_elapsed_us": 170.0},
            "compiler_floor_shift_us": 20.0,
            "theoretical_ceilings": {
                "to_realized_dsl_floor_us": 50.0,
                "to_idealized_hivm_floor_us": 70.0,
                "speedup_to_realized_dsl_floor_upper": 1.5,
                "speedup_to_idealized_hivm_floor_upper": 1.875,
            },
            "achievable_headroom": {
                "status": "not_established",
                "point_estimate_us": None,
                "evidence_required": "counterfactual required",
            },
            "validity_gates": {"model_coverage_status": "complete"},
        },
    }


def test_legacy_alias_adapter_is_non_mutating_and_complete():
    canonical = _canonical_report()
    before = copy.deepcopy(canonical)

    legacy = with_legacy_report_aliases(canonical)

    assert canonical == before
    assert legacy["compatibility"] == {
        "legacy_report_aliases": "v1",
        "deprecated": True,
    }
    assert legacy["reachability"] == {
        "t_bound_hivm_us": 80.0,
        "t_bound_dsl_us": 100.0,
        "t_measured_us": 150.0,
        "compiler_headroom_us": 20.0,
        "author_headroom_us": 50.0,
        "author_residual_us": 50.0,
        "is_violation": False,
        "msprof_source": "op_summary.csv",
        "n_invocations": 9,
        "component_match": True,
    }
    assert legacy["profile"]["diagnosis"] == "Compute Bound"
    assert legacy["profile"]["n_sync_ops"] == 4
    assert legacy["headroom_assessment"] == {
        "status": "unavailable",
        "lower_us": None,
        "upper_us": None,
        "point_estimate_us": None,
        "confidence": "none",
        "method": "counterfactual required",
        "potential_speedup_upper": None,
    }
    assert legacy["t_bound_hivm_us"] == 80.0
    assert legacy["t_measured_us"] == 150.0
    assert legacy["measurement_metric"] == "msprof_task_duration"
    assert legacy["event_elapsed_us"] == 170.0
    assert legacy["event_elapsed_source"] is None
    assert legacy["task_wait_us"] == 12.0
    assert legacy["calibration"]["derived_constant_count"] is None


def test_legacy_alias_adapter_maps_invalid_status_without_ceiling():
    canonical = _canonical_report()
    modeling = canonical["modeling_output"]
    modeling["status"] = "bound_violation"
    modeling["profile_inputs"]["task_duration_us"] = 90.0
    modeling["theoretical_ceilings"] = {
        key: None for key in modeling["theoretical_ceilings"]
    }

    legacy = with_legacy_report_aliases(canonical)

    assert legacy["reachability"]["is_violation"] is True
    assert legacy["reachability"]["author_residual_us"] == -10.0
    assert legacy["headroom_assessment"]["status"] == "unavailable"
    assert legacy["headroom_assessment"]["upper_us"] is None
    assert legacy["headroom_assessment"]["lower_us"] is None


def test_legacy_alias_adapter_suppresses_partial_coverage_residuals():
    canonical = _canonical_report()
    modeling = canonical["modeling_output"]
    modeling["status"] = "model_incomplete"
    modeling["validity_gates"]["model_coverage_status"] = "conservative_partial"

    legacy = with_legacy_report_aliases(canonical)

    assert legacy["reachability"]["compiler_headroom_us"] is None
    assert legacy["reachability"]["author_residual_us"] is None
    assert legacy["headroom_assessment"]["status"] == "model_incomplete"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"schema_version": "perfbound_report_v1", "modeling_output": {}},
        {"schema_version": "perfbound_report_v2"},
    ],
)
def test_legacy_alias_adapter_rejects_noncanonical_input(payload):
    with pytest.raises(ValueError):
        with_legacy_report_aliases(payload)


def test_legacy_alias_adapter_output_is_json_serializable():
    json.dumps(with_legacy_report_aliases(_canonical_report()))


def test_legacy_alias_adapter_preserves_missing_profile_as_none():
    canonical = _canonical_report()
    profile = canonical["modeling_output"]["profile_inputs"]
    profile["diagnosis"] = None

    assert with_legacy_report_aliases(canonical)["profile"] is None


def test_shared_writer_emits_canonical_or_legacy_json(tmp_path):
    canonical_path = tmp_path / "canonical.json"
    legacy_path = tmp_path / "legacy.json"

    write_report_json(_canonical_report(), canonical_path)
    write_report_json(_canonical_report(), legacy_path, legacy_aliases=True)

    assert "reachability" not in json.loads(canonical_path.read_text())
    assert "reachability" in json.loads(legacy_path.read_text())
