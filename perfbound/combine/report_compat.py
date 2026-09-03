"""Opt-in compatibility aliases for pre-v2 report consumers."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .report import REPORT_SCHEMA_VERSION

LEGACY_ALIAS_VERSION = "v1"


def _require_mapping(parent: dict, key: str) -> dict:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"canonical report is missing object: {key}")
    return value


def with_legacy_report_aliases(report: dict) -> dict:
    """Return a copy with deprecated v1 aliases for incremental migration."""
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"expected schema_version={REPORT_SCHEMA_VERSION}, "
            f"got {report.get('schema_version')!r}"
        )

    modeling = _require_mapping(report, "modeling_output")
    bounds = _require_mapping(modeling, "bounds")
    profile = _require_mapping(modeling, "profile_inputs")
    achievable = _require_mapping(modeling, "achievable_headroom")
    screening = _require_mapping(modeling, "screening_metrics")
    validity = _require_mapping(modeling, "validity_gates")

    output = copy.deepcopy(report)
    hivm_floor_us = bounds.get("hivm_floor_us")
    dsl_floor_us = bounds.get("dsl_floor_us")
    task_duration_us = profile.get("task_duration_us")
    coverage_complete = validity.get("model_coverage_status") == "complete"
    compiler_headroom_us = (
        dsl_floor_us - hivm_floor_us
        if coverage_complete
        and dsl_floor_us is not None
        and hivm_floor_us is not None
        else None
    )
    author_headroom_us = (
        task_duration_us - dsl_floor_us
        if coverage_complete
        and task_duration_us is not None
        and dsl_floor_us is not None
        else None
    )
    is_violation = (
        task_duration_us is not None
        and dsl_floor_us is not None
        and dsl_floor_us > task_duration_us
    )
    output.update({
        "t_bound_hivm_us": hivm_floor_us,
        "t_measured_us": task_duration_us,
        "compiler_headroom_us": compiler_headroom_us,
        "author_headroom_us": author_headroom_us,
        "measurement_metric": profile.get("measurement_metric"),
        "event_elapsed_us": screening.get("event_elapsed_us"),
        "event_elapsed_source": None,
        "task_wait_us": profile.get("task_wait_us"),
    })
    calibration = _require_mapping(output, "calibration")
    calibration["derived_constant_count"] = None
    output["reachability"] = {
        "t_bound_hivm_us": hivm_floor_us,
        "t_bound_dsl_us": dsl_floor_us,
        "t_measured_us": task_duration_us,
        "compiler_headroom_us": compiler_headroom_us,
        "author_headroom_us": author_headroom_us,
        "author_residual_us": author_headroom_us,
        "is_violation": is_violation,
        "msprof_source": profile.get("msprof_source"),
        "n_invocations": profile.get("invocations"),
        "component_match": profile.get("component_match"),
    }
    output["profile"] = (
        {
            "diagnosis": profile.get("diagnosis"),
            "dominant_component": profile.get("dominant_component"),
            "exposed_control_frac_measured": profile.get(
                "exposed_control_fraction_measured"
            ),
            "exposed_control_frac_model": profile.get(
                "exposed_control_fraction_model"
            ),
            "exposed_control_deficit_pts": profile.get(
                "exposed_control_deficit_points"
            ),
            "exposed_control_deficit_us": profile.get(
                "exposed_control_deficit_us"
            ),
            "n_sync_ops": profile.get("sync_operations"),
        }
        if profile.get("diagnosis") is not None
        else None
    )

    output["headroom_assessment"] = {
        "status": "unavailable" if coverage_complete else "model_incomplete",
        "lower_us": None,
        "upper_us": None,
        "point_estimate_us": None,
        "confidence": "none",
        "method": achievable.get("evidence_required"),
        "potential_speedup_upper": None,
    }
    output["compatibility"] = {
        "legacy_report_aliases": LEGACY_ALIAS_VERSION,
        "deprecated": True,
    }
    return output


def write_report_json(
    report: dict,
    path: str | Path,
    *,
    legacy_aliases: bool = False,
) -> None:
    """Write canonical JSON, optionally adding deprecated aliases."""
    payload = with_legacy_report_aliases(report) if legacy_aliases else report
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Add deprecated v1 aliases to a canonical perfbound report."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.input.read_text())
    write_report_json(report, args.output, legacy_aliases=True)


if __name__ == "__main__":
    _cli()
