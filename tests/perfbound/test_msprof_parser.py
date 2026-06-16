# Tests for msprof_parser.py (A.6.1)
#
# Validates CSV parsing, AiCore filtering, median computation, warmup discard,
# and component duration parsing.
#
# Source spec: .omc/plans/a6_validation_harness.md §7

import pytest
import tempfile
from pathlib import Path

from perfbound.validate.msprof_parser import (
    parse_kernel_time_us,
    parse_component_durations,
    TimingResult,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE_CSV = FIXTURE_DIR / "op_summary_sample.csv"


def test_aicore_filter_excludes_ai_cpu():
    """AI_CPU rows are excluded from timing."""
    result = parse_kernel_time_us(SAMPLE_CSV, "target_kernel", n_warmup=0)
    # Only 3 AI_CORE rows for target_kernel (1000, 1050, 5000)
    assert result.n_invocations == 3


def test_median_vs_mean_differ():
    """Median (1050) ≠ mean (≈2350) for outlier set."""
    result = parse_kernel_time_us(SAMPLE_CSV, "target_kernel", n_warmup=0)
    # Values: 1000, 1050, 5000 → median=1050
    assert result.t_us == 1050.0
    # NOT the mean (2350)
    assert result.t_us != pytest.approx(2350.0, rel=0.1)


def test_timing_result_fields():
    """TimingResult has .t_us, .n_invocations, .n_warmup_discarded."""
    result = parse_kernel_time_us(SAMPLE_CSV, "target_kernel", n_warmup=0)
    assert hasattr(result, "t_us")
    assert hasattr(result, "n_invocations")
    assert hasattr(result, "n_warmup_discarded")
    assert isinstance(result, TimingResult)


def test_warmup_discarded():
    """n_warmup=1 removes first invocation; n_warmup_discarded=1."""
    result = parse_kernel_time_us(SAMPLE_CSV, "target_kernel", n_warmup=1)
    assert result.n_warmup_discarded == 1
    assert result.n_invocations == 2


def test_op_name_filter_exact():
    """other_kernel rows excluded when filter=target_kernel."""
    result = parse_kernel_time_us(SAMPLE_CSV, "target_kernel", n_warmup=0)
    # 3 AI_CORE rows for target_kernel
    assert result.n_invocations == 3

    result_other = parse_kernel_time_us(SAMPLE_CSV, "other_kernel", n_warmup=0)
    # 1 AI_CORE row for other_kernel
    assert result_other.n_invocations == 1
    assert result_other.t_us == 900.0


def test_overlapping_rows_are_one_invocation(tmp_path):
    csv_path = tmp_path / "overlap.csv"
    csv_path.write_text(
        "Op Name,Task Type,Task Start Time(us),Task Duration(us)\n"
        "kernel,MIX_AIC,1000,100\n"
        "kernel,MIX_AIV,1002,95\n"
        "kernel,MIX_AIC,1200,110\n"
    )

    result = parse_kernel_time_us(csv_path, "kernel", n_warmup=0)

    assert result.n_invocations == 2
    assert result.t_us == 105.0


def test_no_rows_raises_valueerror():
    """ValueError when no matching rows."""
    with pytest.raises(ValueError, match="No AiCore rows"):
        parse_kernel_time_us(SAMPLE_CSV, "nonexistent_kernel", n_warmup=0)


def test_malformed_row_skipped(capsys):
    """NaN duration row → warning printed, row skipped."""
    csv_content = """Op Name,Task Type,Task Start Time(us),Task Duration(us)
kernel,AI_CORE,0,NaN
kernel,AI_CORE,1000,500
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        csv_path = Path(f.name)

    try:
        result = parse_kernel_time_us(csv_path, "kernel", n_warmup=0)
        assert result.n_invocations == 1
        assert result.t_us == 500.0
        captured = capsys.readouterr()
        assert "Warning" in captured.err or "invalid duration" in captured.err.lower()
    finally:
        csv_path.unlink()


def test_zero_duration_excluded(capsys):
    """Zero-duration rows treated as malformed."""
    csv_content = """Op Name,Task Type,Task Start Time(us),Task Duration(us)
kernel,AI_CORE,0,0
kernel,AI_CORE,1000,500
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        csv_path = Path(f.name)

    try:
        result = parse_kernel_time_us(csv_path, "kernel", n_warmup=0)
        assert result.n_invocations == 1
        assert result.t_us == 500.0
    finally:
        csv_path.unlink()


def test_component_match_cube_bound():
    """AI_CORE rows dominate → component_match=True for CUBE-predicted."""
    from perfbound.extract.op_classifier import Component
    from perfbound.validate.harness import _check_component_match

    durations = {"aicore": 1000.0, "mte": 100.0, "aicpu": 50.0, "other": 0.0}
    match = _check_component_match(durations, Component.CUBE)
    assert match is True


def test_component_match_mismatch():
    """MTE rows dominate → component_match=False for CUBE-predicted."""
    from perfbound.extract.op_classifier import Component
    from perfbound.validate.harness import _check_component_match

    durations = {"aicore": 100.0, "mte": 1000.0, "aicpu": 50.0, "other": 0.0}
    match = _check_component_match(durations, Component.CUBE)
    assert match is False


def test_component_match_none_when_no_task_type():
    """Old CSV without Task Type column → component_match=None."""
    from perfbound.extract.op_classifier import Component
    from perfbound.validate.harness import _check_component_match

    durations = {"aicore": 0.0, "mte": 0.0, "aicpu": 0.0, "other": 0.0}
    match = _check_component_match(durations, Component.CUBE)
    assert match is None


def test_parse_component_durations():
    """parse_component_durations returns correct totals (no filter)."""
    durations = parse_component_durations(SAMPLE_CSV)
    # AI_CORE: 1000 + 1050 + 5000 + 900 = 7950 (all kernels)
    # AI_CPU: 800
    assert durations["aicore"] == 7950.0
    assert durations["aicpu"] == 800.0
    assert durations["mte"] == 0.0


def test_parse_component_durations_with_filter():
    """parse_component_durations filters to op_name when provided."""
    durations = parse_component_durations(SAMPLE_CSV, op_name_filter="target_kernel")
    # AI_CORE: 1000 + 1050 + 5000 = 7050 (target_kernel only)
    # AI_CPU: 800 (target_kernel's AI_CPU row)
    # other_kernel's 900 AI_CORE excluded
    assert durations["aicore"] == 7050.0
    assert durations["aicpu"] == 800.0

    durations_other = parse_component_durations(SAMPLE_CSV, op_name_filter="other_kernel")
    assert durations_other["aicore"] == 900.0
    assert durations_other["aicpu"] == 0.0
