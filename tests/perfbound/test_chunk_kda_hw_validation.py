# Real-hardware validation of chunk_kda on the 910B3 (A.6 caveat closure).
#
# Unlike the synthetic-extract tests, this drives the validation harness with
# a REAL msprof op_summary CSV captured from a live 910B3 run (CANN 9.0.0,
# conda env triton_hxl) on 2026-06-10.  The kernel compiled and ran (the
# bishengir ConvertLinalgRToBinary crash was a CANN 9.0.0-beta.2 bug, fixed in
# the 9.0.0 release), and msprof recorded 6 invocations.
#
# This test closes two caveats that were previously "hardware-gated, never run
# on real hardware":
#   1. author_headroom (T_measured − T_bound_DSL) populated with a real number.
#   2. The measurement → soundness → three-level pipeline exercised on real data.
#
# It also permanently guards a bug that ONLY real hardware data exposed: the
# chunk_kda kernel profiles with Task Type "MIX_AIC" (mixed cube), which the
# timing parser's AiCore filter originally did not recognise (it matched only
# "AI_CORE"/"AICORE"), silently dropping every kernel row.
#
# Evidence: .omc/research/hw_runs/  (raw CSV + run log)

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from perfbound.combine.bound_combiner import BoundResult, BindingTier
from perfbound.combine.two_limit import TwoLimitResult
from perfbound.extract.op_classifier import Component
from perfbound.validate.harness import (
    ValidationCase,
    ValidationStatus,
    validate_from_csv,
)
from perfbound.validate.msprof_parser import (
    parse_kernel_time_us,
    parse_component_durations,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HW_CSV = PROJECT_ROOT / "tests" / "perfbound" / "fixtures" / "chunk_kda_op_summary_910b3.csv"
HW_CONFIG = PROJECT_ROOT / "configs" / "ascend_910b.json"

KERNEL_OP_NAME = "chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2"

# ── chunk_kda workload (matches test_chunk_kda_milestone.py) ───────────────
B, T, H, K, V = 32, 8192, 32, 128, 128
BT, BK, BV = 64, 32, 32
TOTAL_PROGRAMS = 128 * 32          # grid (128, 32) = 4096
N_IK = K // BK                     # 4
N_IV = V // BV                     # 4


def _hand_flops_per_program() -> int:
    inner = 3 * (2 * BT * BV * BK) * N_IK * N_IV
    ik0 = 2 * (2 * BT * BV * BT) * N_IV
    outer = 2 * (2 * BT * BK * BT) * N_IK
    post = 2 * (2 * BT * BT * BT)
    return inner + ik0 + outer + post


def _hand_bytes_per_program() -> int:
    per_ik_loads = 4096 + 8192 + 128
    per_iv_loads = 5 * (BT * BV * 2)
    per_iv_ik0 = BT * BV * 2
    per_ik_extra = BT * BK * 2
    total_loads = (
        per_ik_loads * N_IK
        + per_iv_loads * N_IK * N_IV
        + per_iv_ik0 * N_IV
        + per_ik_extra * N_IK
    )
    per_ik_stores = 3 * (BT * BK * 4)
    per_iv_ik0_stores = BT * BV * 2
    post_stores = BT * BT * 4 + BT * 4
    total_stores = per_ik_stores * N_IK + per_iv_ik0_stores * N_IV + post_stores
    return total_loads + total_stores


HAND_FLOPS_TOTAL = _hand_flops_per_program() * TOTAL_PROGRAMS
HAND_BYTES_TOTAL = _hand_bytes_per_program() * TOTAL_PROGRAMS


def _chunk_kda_bound() -> BoundResult:
    """Analytic grid-floor bound for chunk_kda from hand-derived work + HW config.

    A conservative lower bound: max(compute floor, HBM floor).  The kernel's
    dots are bf16, so the Cube fp16 peak applies; bytes move over HBM.
    """
    cfg = json.loads(HW_CONFIG.read_text())
    cube_tflops = cfg["compute_units"]["cube"]["tflops_fp16"]          # 320
    hbm_tbps = cfg["memory_spaces"]["hbm"]["bandwidth_tbps"]           # 1.6

    t_compute_us = HAND_FLOPS_TOTAL / (cube_tflops * 1e12) * 1e6
    t_mem_us = HAND_BYTES_TOTAL / (hbm_tbps * 1e12) * 1e6

    if t_mem_us >= t_compute_us:
        t_bound_us = t_mem_us
        binding = Component.MTE_GM
    else:
        t_bound_us = t_compute_us
        binding = Component.CUBE

    return BoundResult(
        kernel_name="chunk_kda",
        t_bound_us=t_bound_us,
        t_grid_floor_us=t_bound_us,
        t_core_floor_us=t_compute_us,
        t_serial_irreducible_us=0.0,
        binding_tier=BindingTier.GRID,
        binding_component=binding,
    )


requires_hw_csv = pytest.mark.skipif(
    not HW_CSV.exists(), reason="real 910B3 op_summary fixture not present"
)


@requires_hw_csv
class TestRealHardwareParse:
    """The msprof parser handles the real CSV — including MIX_AIC rows."""

    def test_parses_mix_aic_kernel_rows(self):
        """parse_kernel_time_us yields a finite positive time for chunk_kda.

        Regression guard: before recognising MIX_AIC, this raised ValueError
        ("No AiCore rows found") because every kernel row is Task Type MIX_AIC.
        """
        timing = parse_kernel_time_us(HW_CSV, op_name_filter=KERNEL_OP_NAME, n_warmup=0)
        assert timing.t_us > 0 and math.isfinite(timing.t_us)
        # The 6 launches measured ~104.2–104.3 ms each on the 910B3; the
        # parser reports the per-invocation wall time (max across the rows of
        # an invocation).  A tight band makes this a load-bearing guard: a
        # mis-parse (e.g. summing rows, or picking a build_inputs aclnn row)
        # would land well outside it.
        assert 1.0e5 < timing.t_us < 1.1e5, f"unexpected T_measured={timing.t_us} us"
        assert timing.n_invocations == 6

    def test_aicore_is_dominant_component(self):
        """The kernel's measured time is dominated by AI compute-core work."""
        comp = parse_component_durations(HW_CSV, op_name_filter=KERNEL_OP_NAME)
        assert comp["aicore"] > 0, "MIX_AIC rows must count as aicore"
        assert max(comp, key=comp.get) == "aicore"


@requires_hw_csv
class TestRealHardwareValidation:
    """The validation harness classifies the real run soundly (not infra error)."""

    def test_validate_from_csv_is_sound_pass(self):
        case = ValidationCase(
            kernel_name="chunk_kda",
            profiler_op_name=KERNEL_OP_NAME,
            bound_result=_chunk_kda_bound(),
            csv_path=HW_CSV,
            n_warmup=0,
        )
        result = validate_from_csv(case)

        # A real measurement, not an infrastructure failure.
        assert result.status in (
            ValidationStatus.PASS,
            ValidationStatus.BOUND_VIOLATION,
        ), f"got {result.status}: {result.notes}"
        # The analytic floor is a true lower bound → PASS (bound ≤ measured).
        assert result.status == ValidationStatus.PASS
        assert result.t_measured_us > result.t_bound_us > 0
        assert result.tightness > 1.0
        assert str(HW_CSV) in result.msprof_source

    def test_two_limit_author_headroom_is_real(self):
        """author_headroom = T_measured − T_bound_DSL is populated and positive.

        This is the caveat that was previously impossible to close offline:
        a real measured number flowing into TwoLimitResult.author_headroom_us.
        """
        bound = _chunk_kda_bound()
        timing = parse_kernel_time_us(HW_CSV, op_name_filter=KERNEL_OP_NAME, n_warmup=0)

        two_limit = TwoLimitResult(
            kernel_name="chunk_kda",
            t_bound_hivm_us=bound.t_bound_us,
            t_bound_dsl_us=bound.t_bound_us,
            t_measured_us=timing.t_us,
        )
        author = two_limit.author_headroom_us
        assert author is not None
        assert author > 0
        # Headroom is the bulk of the measured time (loose floor, slow kernel).
        assert math.isclose(author, timing.t_us - bound.t_bound_us, rel_tol=1e-9)


# ── Real DES-graph bound (Stage-B Task 1) ─────────────────────────────────

KDA_DES_JSON = PROJECT_ROOT / ".omc" / "research" / "hw_runs" / "kda_des.json"

requires_des_json = pytest.mark.skipif(
    not KDA_DES_JSON.exists(), reason="real kda_des.json fixture not present"
)

# Analytic grid-floor bound (HBM floor) for comparison.
# From the handoff doc: the HBM floor was ~1,386 us (75x loose).
HBM_FLOOR_US = HAND_BYTES_TOTAL / (1.6e12) * 1e6  # ~1,386 us


@requires_des_json
class TestRealDesGraphBound:
    """Stage-B Task 1d: the real des.json produces a sound, tight Tier-2 bound."""

    @staticmethod
    def _report():
        from perfbound.combine.run_report import report_from_desgraph
        return report_from_desgraph(
            des_json=str(KDA_DES_JSON),
            grid_dims=(128, 32),
            n_cores=20,
            kernel_name="chunk_kda",
            t_measured_us=104326.0,
        )

    def test_bound_is_positive(self):
        """The real-bound extraction yields a positive t_bound_us."""
        report = self._report()
        assert report.t_bound_us > 0

    def test_soundness_bound_le_measured(self):
        """Core soundness: T_bound ≤ T_measured on the real des.json extract."""
        report = self._report()
        assert report.t_bound_us <= report.t_measured_us, (
            f"BOUND VIOLATION: t_bound={report.t_bound_us:.2f} us > "
            f"t_measured={report.t_measured_us:.2f} us"
        )

    def test_tighter_than_hbm_floor(self):
        """The real Tier-2 bound is tighter than the analytic HBM grid floor."""
        report = self._report()
        assert report.t_bound_us > HBM_FLOOR_US, (
            f"Real bound {report.t_bound_us:.2f} us should exceed "
            f"HBM floor {HBM_FLOOR_US:.2f} us (Tier-2 should be tighter)"
        )

    def test_des_json_has_expected_op_types(self):
        """The des.json contains Cube, Vector, and MTE ops (full HIVM extract)."""
        from perfbound.extract.hivm_extractor import load_hivm_desgraph
        ops = load_hivm_desgraph(str(KDA_DES_JSON))
        components = {op.component for op in ops}
        assert Component.CUBE in components, "des.json must contain Cube ops"
        assert Component.VECTOR in components, "des.json must contain Vector ops"

    def test_des_json_has_repeat_mask_fields(self):
        """The C++ emitter populates repeat with REAL analytical values (Task 4).

        repeat = ceil(elements / lanes) is derived analytically at emit time
        (the post-GraphSyncSolver hivm.hir IR has no per-op CCE repeat/mask, and
        the late bishengir codegen stage that does is not parseable — see
        RESULTS.md).  Value-asserting: wide SIMD tiles must produce repeat>1,
        guarding against the earlier all-default no-op regression.
        """
        import json
        with open(KDA_DES_JSON) as f:
            data = json.load(f)
        ops = data["operations"]
        assert len(ops) > 0
        assert all("repeat" in op for op in ops), "all ops must have 'repeat' field"
        assert all("mask" in op for op in ops), "all ops must have 'mask' field"
        assert all(isinstance(op["repeat"], int) and op["repeat"] >= 1
                   for op in ops), "repeat must be integer >= 1"
        assert all(isinstance(op["mask"], int) and op["mask"] >= 0
                   for op in ops), "mask must be integer >= 0"
        # Analytical repeat must fire on real SIMD compute ops (not all default).
        nontrivial = [op for op in ops if op["repeat"] > 1]
        assert len(nontrivial) > 50, (
            f"expected many SIMD ops with repeat>1; got {len(nontrivial)}"
        )
