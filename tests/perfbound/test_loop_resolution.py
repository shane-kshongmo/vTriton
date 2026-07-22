# Loop-resolution visibility tests.
#
# Covers the perfbound-side half of "make perf_bound see loop-driven gaps
# and optimization wins": the C++ `loop_diagnostics` block (added in commit
# aba4a2c, extended with upper_bound_trip_count_estimate/body_first_line/
# body_last_line this session) must be surfaced into KernelReport as a
# visible warning + a diagnostic (non-primary, non-sound) companion
# worst-case bound — and none of it may ever change the primary t_bound_us.
#
# Source: .claude/plans/tranquil-moseying-pixel.md ("Test strategy" #2)

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from perfbound.extract.hivm_extractor import (
    HIVMExtract,
    OpRecord,
    load_loop_diagnostics,
)
from perfbound.extract.op_classifier import Component, Precision
from perfbound.extract.dsl_extractor import GridInfo
from perfbound.calibration.calib_loader import load_default_calib_db
from perfbound.combine.bound_combiner import bound_from_extract, worst_case_bound_us
from perfbound.combine.run_report import report_from_desgraph


def _grid_info(total_programs=64, n_cores=20) -> GridInfo:
    return GridInfo(
        grid_dims=(n_cores,), total_programs=total_programs,
        tile_assignment={}, work={},
        occupancy=1.0, load_balance=1.0,
        redundancy=1.0, busiest_core_id=0,
    )


def _synthetic_ops() -> list[OpRecord]:
    """A couple ops outside any loop, several 'inside an unresolved loop'
    (line 40-60) elementwise vector ops that would benefit from a higher
    loop_multiplier — mirrors the real chunk_kda_fwd_kernel shape (most
    duration lives inside the unresolved loop's body)."""
    return [
        OpRecord(op_id=0, op_name="load", component=Component.MTE_GM,
                 precision=Precision.FP16, pipe="CubeMTE2",
                 bytes_transferred=4096, elements=0,
                 loop_multiplier=1, depends_on=[], line=10),
        OpRecord(op_id=1, op_name="mul", component=Component.VECTOR,
                 precision=Precision.FP16, pipe="Vector",
                 bytes_transferred=0, elements=2048, flops=2048,
                 loop_multiplier=1, depends_on=[0], line=45),
        OpRecord(op_id=2, op_name="reduce_sum", component=Component.VECTOR,
                 precision=Precision.FP16, pipe="Vector",
                 bytes_transferred=0, elements=2048, flops=2048,
                 loop_multiplier=1, depends_on=[1], line=52),
    ]


class TestLoadLoopDiagnostics:
    def test_returns_none_when_key_absent(self, tmp_path):
        """Backward compat: DES JSON from a binary predating the loop
        diagnostics feature must not crash — just report 'no data'."""
        des = tmp_path / "old_schema.des.json"
        des.write_text(json.dumps({
            "schema_version": "a3_hivm_des_v1",
            "clock_ghz": 1.85,
            "operations": [],
        }))
        assert load_loop_diagnostics(des) is None

    def test_parses_present_block(self, tmp_path):
        des = tmp_path / "with_loops.des.json"
        payload = {
            "total": 1, "resolved": 0, "unresolved": 1, "max_trip_count": 1,
            "loops": [{
                "line": 40, "lower": 0, "upper": 0, "step": 1,
                "trip_count": 1, "multiplier": 1, "resolved": False,
                "upper_bound_trip_count_estimate": 10,
                "body_first_line": 40, "body_last_line": 60,
            }],
        }
        des.write_text(json.dumps({
            "schema_version": "a3_hivm_des_v1",
            "clock_ghz": 1.85,
            "loop_diagnostics": payload,
            "operations": [],
        }))
        assert load_loop_diagnostics(des) == payload


class TestWorstCaseBoundUs:
    def test_none_without_loop_diagnostics(self):
        extract = HIVMExtract(operations=_synthetic_ops(), handoffs=[])
        db = load_default_calib_db()
        assert worst_case_bound_us(
            extract, None, _grid_info(), db,
        ) is None

    def test_none_when_no_unresolved_loop_has_estimate(self):
        extract = HIVMExtract(operations=_synthetic_ops(), handoffs=[])
        db = load_default_calib_db()
        # A loop entry present but fully resolved — no diagnostic needed.
        diag = {"total": 1, "resolved": 1, "unresolved": 0, "loops": [{
            "line": 40, "resolved": True,
            "upper_bound_trip_count_estimate": -1,
            "body_first_line": 0, "body_last_line": 0,
        }]}
        assert worst_case_bound_us(extract, diag, _grid_info(), db) is None

    def test_worst_case_exceeds_primary_and_never_mutates_primary(self):
        """The core soundness contract: worst_case_bound_us must be >= the
        primary bound_from_extract result on the SAME unmodified extract,
        and must not mutate the input extract's op list in place."""
        ops = _synthetic_ops()
        extract = HIVMExtract(operations=ops, handoffs=[])
        db = load_default_calib_db()

        primary = bound_from_extract(
            extract, db, kernel_name="synthetic", n_cores=20,
            total_programs=64,
        )

        diag = {"total": 1, "resolved": 0, "unresolved": 1, "loops": [{
            "line": 45, "resolved": False,
            "upper_bound_trip_count_estimate": 10,
            "body_first_line": 40, "body_last_line": 60,
        }]}
        worst_us = worst_case_bound_us(
            extract, diag, _grid_info(total_programs=64), db,
            kernel_name="synthetic", n_cores=20, total_programs=64,
        )

        assert worst_us is not None
        assert worst_us >= primary.t_bound_us
        # Input extract's ops must be untouched (worst-case path copies).
        assert all(op.loop_multiplier == 1 for op in extract.operations)


class TestReportFromDesgraphLoopResolution:
    """End-to-end through the actual run_report.py wiring (Phase 1 + 2)."""

    def _write_des(self, path: Path, with_loop_diagnostics: bool):
        ops_json = [
            {"id": 0, "name": "load", "pipe": "PIPE_MTE2_V", "duration": 10,
             "bytes": 4096, "elements": 0, "loop_multiplier": 1,
             "depends_on": [], "line": 10},
            {"id": 1, "name": "mul", "pipe": "PIPE_V", "duration": 20,
             "bytes": 0, "elements": 2048, "flops": 2048,
             "loop_multiplier": 1, "depends_on": [0], "line": 45},
            {"id": 2, "name": "reduce_sum", "pipe": "PIPE_V", "duration": 20,
             "bytes": 0, "elements": 2048, "flops": 2048,
             "loop_multiplier": 1, "depends_on": [1], "line": 52},
        ]
        payload = {
            "schema_version": "a3_hivm_des_v1",
            "clock_ghz": 1.85,
            "operations": ops_json,
        }
        if with_loop_diagnostics:
            payload["loop_diagnostics"] = {
                "total": 1, "resolved": 0, "unresolved": 1, "loops": [{
                    "line": 40, "resolved": False,
                    "upper_bound_trip_count_estimate": 10,
                    "body_first_line": 40, "body_last_line": 60,
                }],
            }
        path.write_text(json.dumps(payload))

    def test_backward_compat_no_loop_diagnostics(self, tmp_path):
        des = tmp_path / "no_loops.des.json"
        self._write_des(des, with_loop_diagnostics=False)
        db = load_default_calib_db()
        report = report_from_desgraph(
            des_json=des, grid_dims=(64,), calib_db=db,
            kernel_name="synthetic",
        )
        assert report.loop_resolution is None
        assert report.t_bound_us > 0  # primary bound still computed fine

    def test_loop_resolution_surfaced_and_warning_present(self, tmp_path):
        des = tmp_path / "with_loops.des.json"
        self._write_des(des, with_loop_diagnostics=True)
        db = load_default_calib_db()
        report = report_from_desgraph(
            des_json=des, grid_dims=(64,), calib_db=db,
            kernel_name="synthetic",
        )
        assert report.loop_resolution is not None
        assert report.loop_resolution["unresolved"] == 1
        assert report.loop_resolution["unresolved_lines"] == [40]
        worst = report.loop_resolution["t_bound_worst_case_us"]
        assert worst is not None
        assert worst >= report.t_bound_us

        # The warning must be appended, not swallowed by merge_calibration's
        # earlier overwrite of calibration_warnings.
        assert any(
            "data-dependent trip counts" in w for w in report.calibration_warnings
        )

        # Primary bound must be unaffected by the diagnostic path.
        d = report.to_dict()
        assert d["loop_resolution"]["t_bound_worst_case_us"] == pytest.approx(worst)
        assert d["t_bound_us"] == report.t_bound_us
