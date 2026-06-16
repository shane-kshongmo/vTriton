# A.5 — Report tests.
#
# Covers:
#   - Change #6: at-bound state when all gaps ≈ 0
#   - Recommendation mapping for each dominant gap
#   - Report round-trip (to_dict, to_json, to_text)
#
# Source: .omc/plans/a5_bound_combiner.md Change #6

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from perfbound.combine.bound_combiner import BoundResult, Attribution, BindingTier
from perfbound.combine.report import KernelReport
from perfbound.extract.op_classifier import Component


def _make_result(
    t_bound=100.0,
    grid_frac=0.0,
    gap1_frac=0.0,
    gap2_frac=0.0,
    gap3_frac=0.0,
    gap4_frac=0.0,
    binding_tier=BindingTier.COMPONENT,
    binding_component=Component.CUBE,
) -> BoundResult:
    attr = Attribution(
        grid_gap_us=t_bound * grid_frac,
        gap1_wrong_unit_us=t_bound * gap1_frac,
        gap2_coalescing_us=t_bound * gap2_frac,
        gap3_avoidable_serial_us=t_bound * gap3_frac,
        gap4_intra_unit_exec_us=t_bound * gap4_frac,
        grid_gap_frac=grid_frac,
        gap1_frac=gap1_frac,
        gap2_frac=gap2_frac,
        gap3_frac=gap3_frac,
        gap4_frac=gap4_frac,
    )
    return BoundResult(
        kernel_name="test_kernel",
        t_bound_us=t_bound,
        t_grid_floor_us=t_bound * 0.5,
        t_core_floor_us=t_bound * 0.8,
        t_serial_irreducible_us=t_bound * 0.1,
        binding_tier=binding_tier,
        binding_component=binding_component,
        attribution=attr,
    )


class TestAtBoundState:
    """When all gap fractions ≈ 0, report says 'at bound'."""

    def test_zero_gaps_at_bound_message(self):
        """All gaps = 0 → recommendation mentions 'at bound'."""
        result = _make_result(t_bound=100.0)
        report = KernelReport.from_bound(result)
        assert "at" in report.recommended_action.lower() or "bound" in report.recommended_action.lower()
        assert "algorithmic redesign" in report.recommended_action.lower()

    def test_nonzero_gaps_gives_recommendation(self):
        """Dominant gap = gap2 → recommendation says 'merge transfers'."""
        result = _make_result(t_bound=100.0, gap2_frac=0.5)
        report = KernelReport.from_bound(result)
        assert "merge transfers" in report.recommended_action.lower() or "transfer" in report.recommended_action.lower()


class TestRecommendationMapping:
    """Each dominant gap maps to the correct recommendation."""

    def test_grid_dominant(self):
        result = _make_result(t_bound=100.0, grid_frac=0.8)
        report = KernelReport.from_bound(result)
        assert "grid" in report.recommended_action.lower() or "occupancy" in report.recommended_action.lower()

    def test_gap1_dominant(self):
        result = _make_result(t_bound=100.0, gap1_frac=0.6)
        report = KernelReport.from_bound(result)
        assert "dsl" in report.recommended_action.lower() or "eligible" in report.recommended_action.lower()

    def test_gap3_dominant(self):
        result = _make_result(t_bound=100.0, gap3_frac=0.7)
        report = KernelReport.from_bound(result)
        assert "ping-pong" in report.recommended_action.lower() or "overlap" in report.recommended_action.lower()

    def test_gap4_dominant(self):
        result = _make_result(t_bound=100.0, gap4_frac=0.5)
        report = KernelReport.from_bound(result)
        assert "simd" in report.recommended_action.lower() or "repeat" in report.recommended_action.lower() or "utilization" in report.recommended_action.lower()


class TestReportRoundTrip:
    """Report serialization round-trips."""

    def test_to_dict_has_all_fields(self):
        result = _make_result(t_bound=50.0, gap1_frac=0.1)
        report = KernelReport.from_bound(result)
        d = report.to_dict()
        assert d["kernel_name"] == "test_kernel"
        assert d["t_bound_us"] == pytest.approx(50.0)
        assert "attribution" in d
        assert "attribution_us" in d
        assert "recommended_action" in d

    def test_to_json_parseable(self):
        result = _make_result(t_bound=75.0)
        report = KernelReport.from_bound(result)
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["t_bound_us"] == pytest.approx(75.0)

    def test_to_text_contains_bound(self):
        result = _make_result(t_bound=42.0)
        report = KernelReport.from_bound(result)
        text = report.to_text()
        assert "42.00" in text
        assert "Performance Bound Report" in text

    def test_two_limit_in_report(self):
        """When two_limit is provided, report includes HIVM bound in Reachability Hierarchy."""
        from perfbound.combine.two_limit import TwoLimitResult
        result = _make_result(t_bound=100.0)
        tl = TwoLimitResult(
            kernel_name="test_kernel",
            t_bound_hivm_us=80.0,
            t_bound_dsl_us=100.0,
        )
        report = KernelReport.from_bound(result, two_limit=tl)
        assert report.t_bound_hivm_us == pytest.approx(80.0)
        assert report.compiler_headroom_us == pytest.approx(20.0)
        text = report.to_text()
        assert "Reachability Hierarchy" in text
        assert "Hardware floor" in text
