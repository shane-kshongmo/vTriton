# Multi-kernel soundness validation (US-SB-005).
#
# Drives the validation harness with REAL 910B3 msprof CSVs for several kernels
# and asserts T_bound <= T_measured (soundness) for each.  This is the n>=5
# spread the spec wants; any BOUND_VIOLATION is a model bug, not a test bug.
#
# Each kernel here is profiled live (scripts/remote_bench.py) and its fixture
# committed under tests/perfbound/fixtures/.  See
# .omc/research/hw_runs/multi_kernel_results.json for the aggregate table.

from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from perfbound.combine.bound_combiner import BoundResult, BindingTier
from perfbound.extract.op_classifier import Component
from perfbound.validate.harness import (
    ValidationCase,
    ValidationStatus,
    validate_from_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = PROJECT_ROOT / "tests" / "perfbound" / "fixtures"
HW_CONFIG = json.loads((PROJECT_ROOT / "configs" / "ascend_910b3_v4.json").read_text())
HBM_TBPS = HW_CONFIG["memory_spaces"]["hbm"]["bandwidth_tbps"]


def _hbm_floor_bound(kernel: str, bytes_total: int) -> BoundResult:
    t_mem_us = bytes_total / (HBM_TBPS * 1e12) * 1e6
    return BoundResult(
        kernel_name=kernel, t_bound_us=t_mem_us, t_grid_floor_us=t_mem_us,
        t_core_floor_us=0.0, t_serial_irreducible_us=0.0,
        binding_tier=BindingTier.GRID, binding_component=Component.MTE_GM,
    )


# (kernel, profiler_op_name, n_warmup, bound_factory) ----------------------
_VADD_N = 16 * 1024 * 1024


_CASES = [
    pytest.param(
        "vector_add", "vector_add_op_summary_910b3.csv", "add_kernel", 1,
        _hbm_floor_bound("vector_add", 3 * _VADD_N * 4),  # 2 loads + 1 store fp32
        id="vector_add",
    ),
]


@pytest.mark.parametrize("kernel,fixture,op_name,n_warmup,bound", _CASES)
def test_kernel_bound_is_sound(kernel, fixture, op_name, n_warmup, bound):
    """Real-hardware soundness: T_bound <= T_measured (no BOUND_VIOLATION)."""
    csv = FIXTURES / fixture
    if not csv.exists():
        pytest.skip(f"fixture {fixture} not present")
    case = ValidationCase(
        kernel_name=kernel, profiler_op_name=op_name,
        bound_result=bound, csv_path=csv, n_warmup=n_warmup,
    )
    result = validate_from_csv(case)
    assert result.status == ValidationStatus.PASS, (
        f"{kernel}: {result.status} (t_bound={result.t_bound_us:.1f} "
        f"t_measured={result.t_measured_us:.1f}): {result.notes}"
    )
    assert result.t_measured_us > result.t_bound_us > 0
    assert result.tightness > 1.0


def test_soundness_rate_is_one():
    """Aggregate soundness across the committed multi-kernel set must be 1.0."""
    table = json.loads(
        (PROJECT_ROOT / ".omc" / "research" / "hw_runs" / "multi_kernel_results.json").read_text()
    )
    statuses = [k["status"] for k in table["kernels"]]
    assert statuses, "no kernels in results table"
    rate = sum(s == "PASS" for s in statuses) / len(statuses)
    assert rate == 1.0, f"soundness_rate={rate}; statuses={statuses}"
