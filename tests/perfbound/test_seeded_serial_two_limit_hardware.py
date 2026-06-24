"""US-SB-008 seeded-serial two-limit hardware fixture."""

from __future__ import annotations

import csv
import json
import statistics as stats
from pathlib import Path

import numpy as np
import pytest

from perfbound.combine.run_report import report_from_desgraph


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / ".omc" / "research" / "hw_runs" / "seeded_serial"
RESULT = EVIDENCE / "seeded_serial_two_limit_result.json"


def _kernel_durations(csv_path: Path) -> list[float]:
    with csv_path.open() as f:
        return [
            float(row["Task Duration(us)"])
            for row in csv.DictReader(f)
            if row["Op Name"] == "seeded_serial_kernel"
        ]


def test_seeded_serial_two_limit_ordering_and_quantification():
    result = json.loads(RESULT.read_text())

    on = report_from_desgraph(
        EVIDENCE / "seeded_on_des.json",
        tuple(result["grid_dims"]),
        n_cores=result["n_cores"],
        kernel_name="seeded_serial_on",
    )
    off = report_from_desgraph(
        EVIDENCE / "seeded_off_des.json",
        tuple(result["grid_dims"]),
        n_cores=result["n_cores"],
        kernel_name="seeded_serial_off",
    )

    predicted = on.t_bound_us - off.t_bound_us
    measured = result["barrier_on"]["median_us"] - result["barrier_off_ttadapter"]["median_us"]

    assert off.t_bound_us == pytest.approx(result["bounds"]["t_bound_hivm_us"])
    assert on.t_bound_us == pytest.approx(result["bounds"]["t_bound_dsl_us"])
    assert off.t_bound_us <= on.t_bound_us <= result["hardware"]["t_measured_us"]
    assert predicted == pytest.approx(result["bounds"]["predicted_headroom_us"])
    assert measured == pytest.approx(result["hardware"]["measured_delta_us"])
    assert abs(predicted - measured) / measured < 0.20
    assert result["acceptance"]["compiler_ir_ttadapter_profiled"] is True
    assert result["acceptance"]["source_level_quantitative_validation"] is True
    assert result["acceptance"]["satisfies_us_sb_008"] is True


def test_seeded_serial_uses_heldout_barrier_calibration():
    result = json.loads(RESULT.read_text())
    calib = result["heldout_barrier_calibration"]

    on = calib["barrier_on"]
    off = calib["barrier_off_source"]
    on_durations = _kernel_durations(ROOT / on["op_summary_csv"])
    off_durations = _kernel_durations(ROOT / off["op_summary_csv"])
    on_measured = on_durations[result["warmup"]:]
    off_measured = off_durations[result["warmup"]:]

    assert len(on_measured) == len(off_measured) == result["iters"]
    assert stats.median(on_measured) == pytest.approx(on["median_us"])
    assert stats.median(off_measured) == pytest.approx(off["median_us"])
    assert on["median_us"] - off["median_us"] == pytest.approx(
        calib["measured_delta_us"]
    )
    assert calib["pipe_barrier_cycles_per_iter"] == pytest.approx(
        result["bounds"]["pipe_barrier_cycles_per_iter"]
    )


def test_seeded_serial_timing_fixture_is_warmed_and_stable():
    result = json.loads(RESULT.read_text())

    for key in ("barrier_on", "barrier_off_ttadapter", "barrier_off_source"):
        run = result[key]
        durations = _kernel_durations(ROOT / run["op_summary_csv"])
        measured = durations[result["warmup"]:]

        assert len(durations) == run["kernel_rows"]
        assert len(measured) == result["iters"] == run["measured_rows"]
        assert stats.median(measured) == pytest.approx(run["median_us"])
        assert stats.stdev(measured) / stats.mean(measured) < 0.05


def test_edited_npuir_is_not_claimed_as_profiled():
    result = json.loads(RESULT.read_text())
    edited = result["barrier_off_hivm"]

    assert edited["remote_bishengir_compile_accepted"] is True
    assert edited["remote_runnable_host_binary"] is False
    assert edited["cache_replacement_launch_tested"] is True
    assert edited["cache_replacement_launch_ok"] is False
    assert result["acceptance"]["edited_npuir_hardware_profiled"] is False


def test_ttadapter_edit_removes_only_the_compiler_barrier():
    result = json.loads(RESULT.read_text())
    on_ttadapter = ROOT / result["barrier_on"]["ttadapter"]
    off_ttadapter = ROOT / result["barrier_off_ttadapter"]["ttadapter"]
    on_text = on_ttadapter.read_text()
    off_text = off_ttadapter.read_text()

    assert on_text.count("gpu.barrier") == 1
    assert "gpu.barrier" not in off_text
    assert result["barrier_off_ttadapter"]["hand_edit"].startswith(
        "Removed exactly one gpu.barrier"
    )


def test_seeded_serial_outputs_are_bitwise_equal():
    # The 32MB stream-A (outa) arrays are excluded from git to keep history lean
    # (see the evidence dir's .gitignore); their bitwise equality is recorded in
    # result.json (outa_equal / max_abs_outa) and asserted below.  Pairs whose
    # artifacts are absent (CI / fresh clone) are skipped from the live check,
    # while the committed stream-B (outb) arrays always exercise a real
    # np.array_equal comparison.
    pairs = [
        ("seeded_on_outa.npy", "seeded_off_ttadapter_outa.npy"),
        ("seeded_on_outb.npy", "seeded_off_ttadapter_outb.npy"),
        ("seeded_on_outa.npy", "seeded_off_src_outa.npy"),
        ("seeded_on_outb.npy", "seeded_off_src_outb.npy"),
    ]
    checked = 0
    for lhs, rhs in pairs:
        lp, rp = EVIDENCE / lhs, EVIDENCE / rhs
        if not (lp.exists() and rp.exists()):
            continue
        assert np.array_equal(np.load(lp), np.load(rp))
        checked += 1
    assert checked >= 2  # the two committed outb pairs are always live-checked

    result = json.loads(RESULT.read_text())
    assert result["correctness"]["outa_equal"] is True
    assert result["correctness"]["outb_equal"] is True
