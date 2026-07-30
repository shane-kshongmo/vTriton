# DES event-wait wiring tests.
#
# Covers the report-side wiring of des_event_wait_analyzer into
# run_report.report_from_desgraph: the DES critical-path event-wait
# attribution (Gap-3) must be surfaced into KernelReport.des_event_wait as a
# pure diagnostic, and it must NEVER change the primary t_bound_us (spec
# §2.2/§3: event waits explain elapsed time, they are not a bound term).

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from perfbound.calibration.calib_loader import load_default_calib_db
from perfbound.combine.run_report import report_from_desgraph


_CLOCK_GHZ = 1.85
_CYCLES_PER_US = _CLOCK_GHZ * 1000.0


def _ops():
    # One MTE load, one vector mul, and a wait_flag on the vector core that
    # sits on the critical path with a non-zero event wait.
    return [
        {"id": 0, "name": "load", "pipe": "PIPE_MTE2_V", "core_type": "VECTOR",
         "duration": 10, "issue_duration": 10, "event_wait_cycles": 0,
         "bytes": 4096, "elements": 0, "loop_multiplier": 1,
         "depends_on": [], "end_cycle": 10, "line": 10},
        {"id": 1, "name": "mul", "pipe": "PIPE_V", "core_type": "VECTOR",
         "duration": 20, "issue_duration": 20, "event_wait_cycles": 0,
         "bytes": 0, "elements": 2048, "flops": 2048, "loop_multiplier": 1,
         "depends_on": [0], "end_cycle": 30, "line": 45},
        {"id": 2, "name": "wait_flag", "pipe": "PIPE_MTE3", "core_type": "VECTOR",
         "duration": 10, "issue_duration": 10, "event_wait_cycles": 30,
         "bytes": 0, "elements": 0, "loop_multiplier": 1,
         "depends_on": [1], "end_cycle": 60, "line": 52},
    ]


def _write_des(path: Path, *, with_critical_path: bool):
    payload = {
        "schema_version": "a3_hivm_des_v1",
        "clock_ghz": _CLOCK_GHZ,
        "operations": _ops(),
    }
    if with_critical_path:
        payload["critical_path_summary"] = {
            "cycles": 60, "issue_cycles": 40,
            "event_wait_cycles": 30, "ops": [0, 1, 2],
        }
        payload["calibration_summary"] = {"sync_event_wait_cycles": 30}
    else:
        # Static-scheduled / legacy graph: critical path present but empty.
        payload["critical_path_summary"] = {
            "cycles": 0, "issue_cycles": 0, "event_wait_cycles": 0, "ops": []
        }
    path.write_text(json.dumps(payload))


def test_des_event_wait_surfaced_when_critical_path_populated(tmp_path):
    des = tmp_path / "with_cp.des.json"
    _write_des(des, with_critical_path=True)
    report = report_from_desgraph(
        des_json=des, grid_dims=(64,), calib_db=load_default_calib_db(),
        kernel_name="synthetic",
    )
    dew = report.des_event_wait
    assert dew is not None
    assert dew["populated"] is True
    assert dew["critical_path_event_wait_cycles"] == 30
    assert dew["critical_path_event_wait_us"] == 30 / _CYCLES_PER_US
    # Top wait group keys off name|pipe|core_type and only counts waits > 0.
    assert dew["top_wait_groups"], "expected a non-zero wait group"
    top = dew["top_wait_groups"][0]
    assert top["key"] == "wait_flag|PIPE_MTE3|VECTOR"
    assert top["wait_cycles"] == 30


def test_des_event_wait_never_changes_primary_bound(tmp_path):
    """The critical-path event wait is attribution, not a bound term: the same
    graph with vs without the populated critical path must give the same
    t_bound_us."""
    with_cp = tmp_path / "with_cp.des.json"
    without_cp = tmp_path / "no_cp.des.json"
    _write_des(with_cp, with_critical_path=True)
    _write_des(without_cp, with_critical_path=False)
    db = load_default_calib_db()
    r_with = report_from_desgraph(des_json=with_cp, grid_dims=(64,),
                                  calib_db=db, kernel_name="synthetic")
    r_without = report_from_desgraph(des_json=without_cp, grid_dims=(64,),
                                     calib_db=db, kernel_name="synthetic")
    assert r_with.t_bound_us == r_without.t_bound_us
    # Event-wait (30 cyc) exceeding the critical path is allowed — it is
    # overlapping attribution, and it must not have leaked into the bound.
    assert r_with.t_bound_us > 0


def test_empty_critical_path_reports_not_populated_with_warning(tmp_path):
    des = tmp_path / "static.des.json"
    _write_des(des, with_critical_path=False)
    report = report_from_desgraph(
        des_json=des, grid_dims=(64,), calib_db=load_default_calib_db(),
        kernel_name="synthetic",
    )
    assert report.des_event_wait is not None
    assert report.des_event_wait["populated"] is False
    assert any(
        "event-wait" in w and "--scheduler des" in w
        for w in report.calibration_warnings
    )
