# Tests for Stage-B stories: US-SB-006, US-SB-007, US-SB-008
#
# US-SB-007: Scalar throughput calibration.
# US-SB-008: Two-limit compiler-headroom validation (chunk_kda).
# US-SB-006: accepted seeded-gap counterfactual audit.

import json
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from perfbound.calibration.calib_loader import load_default_calib_db
from perfbound.calibration.constants import CalibrationConstant

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ===========================================================================
# US-SB-007: Scalar throughput calibration
# ===========================================================================

class TestScalarThroughputCalibration:
    """Validates the direct CCE scalar-throughput measurement."""

    def test_scalar_constant_exists_in_db(self):
        """P_scalar_add_sustained constant is present in the calibration DB."""
        db = load_default_calib_db()
        c = db.constants.get("P_scalar_add_sustained")
        assert c is not None, "P_scalar_add_sustained missing from calibration DB"

    def test_scalar_throughput_in_vector_config(self):
        """VectorConfig.scalar_throughput_fp16_tflops is populated (not default 0)."""
        db = load_default_calib_db()
        assert db.vector.scalar_throughput_fp16_tflops > 0, (
            "scalar_throughput_fp16_tflops must be populated in calibration DB"
        )

    def test_scalar_is_directly_measured(self):
        db = load_default_calib_db()
        c = db.constants.get("P_scalar_add_sustained")
        assert c.source == "cce_microbench"
        assert c.n_runs == 30
        assert db.vector.scalar_throughput_measured is True
        assert db.vector.scalar_throughput_fp16_tflops == pytest.approx(c.value / 1000)

    def test_measured_scalar_drives_bound_rate(self):
        db = load_default_calib_db()
        ops_per_us = db.vector.get_scalar_throughput_ops_per_us("fp16")
        assert ops_per_us == pytest.approx(0.0005998078861614121 * 1e6)
        assert 500 < ops_per_us < 700

    def test_unmeasured_scalar_would_use_vector_upper_rate(self):
        """Future unmeasured scalar estimates must not tighten the bound."""
        from perfbound.calibration.constants import VectorConfig
        vc = VectorConfig(
            vec_width_elements=128,
            throughput_fp16_tflops=0.015133235851136873,
            scalar_throughput_fp16_tflops=0.00011822840508700682,
            scalar_throughput_measured=False,
        )
        ops_per_us = vc.get_scalar_throughput_ops_per_us("fp16")
        assert ops_per_us == pytest.approx(0.015133235851136873 * 1e6)

    def test_scalar_ci_propagated(self):
        """Direct scalar measurement has a tight confidence interval."""
        db = load_default_calib_db()
        sca_const = db.constants.get("P_scalar_add_sustained")
        assert sca_const is not None
        assert sca_const.ci_rel < 0.001

    def test_scalar_source_is_cce_microbench(self):
        db = load_default_calib_db()
        c = db.constants.get("P_scalar_add_sustained")
        assert c.source == "cce_microbench"
        assert c.is_valid

    def test_scalar_constant_value_within_range(self):
        """Sanity check: scalar throughput is between 0.05 and 1.0 GFLOPS.

        The measured dependent-FMA rate is ~0.600 GFLOPS.
        """
        db = load_default_calib_db()
        c = db.constants.get("P_scalar_add_sustained")
        assert 0.05 < c.value < 1.0, (
            f"P_scalar = {c.value} GFLOPS is outside sanity range [0.05, 1.0]"
        )


# ===========================================================================
# US-SB-008: Two-limit compiler-headroom validation
# ===========================================================================

KDA_DES_JSON = PROJECT_ROOT / ".omc" / "research" / "hw_runs" / "kda_des.json"

requires_des_json = pytest.mark.skipif(
    not KDA_DES_JSON.exists(), reason="real kda_des.json fixture not present"
)


@requires_des_json
class TestTwoLimitCompilerHeadroom:
    """Validates the two-limit (T_bound_HIVM / T_bound_DSL) for chunk_kda.

    Acceptance (US-SB-008):
    - T_bound_HIVM <= T_bound_DSL <= T_measured computed for chunk_kda
    - Result annotated with Gap-1/3 interpretation
    - Evidence committed

    Note: hand-optimized HIVM compilation is documented as infeasible for
    chunk_kda (bishengir-compile accepts MLIR, not des.json; the compiler
    headroom is only 44.99 µs / 0.04% — too small to validate on hardware
    where measurement noise exceeds the headroom).
    """

    T_MEASURED_US = 104326.0  # from msprof on real 910B3

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

    def test_three_level_soundness(self):
        """T_bound_HIVM <= T_bound_DSL <= T_measured (three-level hierarchy)."""
        report = self._report()
        assert report.t_bound_hivm_us is not None, "T_bound_HIVM must be computed"
        assert report.t_bound_us is not None, "T_bound_DSL must be computed"
        assert report.t_measured_us is not None, "T_measured must be populated"

        assert report.t_bound_hivm_us <= report.t_bound_us, (
            f"T_bound_HIVM ({report.t_bound_hivm_us:.2f}) must be <= "
            f"T_bound_DSL ({report.t_bound_us:.2f})"
        )
        assert report.t_bound_us <= report.t_measured_us, (
            f"T_bound_DSL ({report.t_bound_us:.2f}) must be <= "
            f"T_measured ({report.t_measured_us:.2f})"
        )

    def test_compiler_headroom_is_non_negative(self):
        """compiler_headroom = T_bound_DSL - T_bound_HIVM >= 0."""
        report = self._report()
        assert report.compiler_headroom_us is not None
        assert report.compiler_headroom_us >= 0, (
            f"compiler_headroom must be >= 0, got {report.compiler_headroom_us:.2f}"
        )

    def test_author_headroom_is_positive(self):
        """author_headroom = T_measured - T_bound_DSL > 0 (large gap)."""
        report = self._report()
        assert report.author_headroom_us is not None
        assert report.author_headroom_us > 0, (
            "author_headroom must be positive (model soundness)"
        )

    def test_compiler_headroom_is_small(self):
        """Compiler headroom is small: bishengir's lowering is near-optimal.

        For chunk_kda, the compiler headroom (Gap-1/Gap-3 only) is < 0.1%
        of T_measured. This means the compiler cannot meaningfully improve
        the bound — the dominant gap is author headroom (Triton-level
        rewrite opportunity), not compiler-level suboptimality.
        """
        report = self._report()
        headroom_pct = report.compiler_headroom_us / report.t_measured_us * 100
        assert headroom_pct < 1.0, (
            f"compiler headroom should be < 1% of T_measured, got {headroom_pct:.2f}%"
        )

    def test_dominant_gap_is_author_headroom(self):
        """The dominant gap (T_measured - T_bound_DSL) is author headroom.

        Validates the model's key finding: for chunk_kda, the compiler is
        near-optimal and the performance opportunity is at the DSL/kernel level.
        """
        report = self._report()
        total_gap = report.t_measured_us - report.t_bound_hivm_us
        compiler = report.compiler_headroom_us
        author = report.author_headroom_us
        assert author / total_gap > 0.99, (
            f"Author headroom should dominate (>99% of total gap), "
            f"got {author/total_gap*100:.1f}%"
        )

    def test_gap_interpretation_is_documented(self):
        """The result uses Gap-1/Gap-3 only (resolved decision).

        The spec author resolved the gap interpretation: T_bound_HIVM relaxes
        only Gap-1 (placement) and Gap-3 (avoidable serialization). Gap-2
        (coalescing) and Gap-4 (intra-unit exec) remain as hardware limits.
        """
        report = self._report()
        # Gap-1/Gap-3 relaxation should only produce a small headroom
        assert report.compiler_headroom_us < 100, (
            "Gap-1/3-only relaxation should produce < 100µs headroom "
            f"(got {report.compiler_headroom_us:.2f}µs)"
        )

    def test_binding_is_component_tier(self):
        """chunk_kda binds at Tier 2 (component level), not grid level."""
        report = self._report()
        assert report.binding_tier == "component", (
            f"chunk_kda should bind at component tier, got {report.binding_tier}"
        )

    def test_attribution_gap4_is_largest(self):
        """Gap-4 (intra-unit exec) is the largest attributed gap."""
        report = self._report()
        attr = report.attribution
        gap4 = attr.get("gap4_intra_unit_exec", 0)
        assert gap4 > 0, "Gap-4 must be non-zero"
        for gap_name, gap_val in attr.items():
            if gap_name != "gap4_intra_unit_exec":
                assert gap4 >= gap_val, (
                    f"Gap-4 ({gap4:.4f}) should be >= {gap_name} ({gap_val:.4f})"
                )


# ===========================================================================
# US-SB-006: work-scaling sanity check guard
# ===========================================================================

VECADD_16M_CSV = PROJECT_ROOT / "tests" / "perfbound" / "fixtures" / "vector_add_op_summary_910b3.csv"

requires_vecadd_csv = pytest.mark.skipif(
    not VECADD_16M_CSV.exists(), reason="vector_add op_summary fixture not present"
)


@requires_vecadd_csv
class TestScalarCalibrationSoundness:
    """Soundness guard for future unmeasured scalar estimates."""

    def test_unmeasured_scalar_rate_equals_vector(self):
        """Unmeasured scalar falls back to the Vector rate (no tightening)."""
        from perfbound.calibration.constants import VectorConfig

        vc = VectorConfig(
            throughput_fp16_tflops=0.015,
            scalar_throughput_fp16_tflops=0.0005,
            scalar_throughput_measured=False,
        )
        vec_ops = vc.throughput_fp16_tflops * 1e6
        sca_ops = vc.get_scalar_throughput_ops_per_us("fp16")
        assert sca_ops == pytest.approx(vec_ops), (
            f"unmeasured scalar must use the Vector upper rate, "
            f"got scalar={sca_ops} vector={vec_ops}"
        )


# ===========================================================================
# US-SB-006 fixtures and results (hardware-dependent)
# ===========================================================================

COUNTERFACTUAL_RESULTS = PROJECT_ROOT / ".omc" / "research" / "hw_runs" / "counterfactual_results.json"
COUNTERFACTUAL_GAP_RESULTS = PROJECT_ROOT / ".omc" / "research" / "hw_runs" / "counterfactual_gap_results.json"

requires_counterfactual = pytest.mark.skipif(
    not COUNTERFACTUAL_RESULTS.exists(),
    reason="counterfactual results fixture not present"
)
requires_counterfactual_gap = pytest.mark.skipif(
    not COUNTERFACTUAL_GAP_RESULTS.exists(),
    reason="counterfactual gap audit fixture not present"
)


@requires_counterfactual
class TestWorkScalingSanityCheck:
    """Vector-add work scaling is a sanity check, not US-SB-006 closure.

    Validates that the model correctly predicts the performance change when
    work (data size) doubles for a memory-bound kernel (vector_add).
    """

    @staticmethod
    def _load_results():
        with open(COUNTERFACTUAL_RESULTS) as f:
            return json.load(f)

    def test_counterfactual_result_exists(self):
        """Work-scaling results JSON is present and internally consistent."""
        data = self._load_results()
        assert "kernel_name" in data
        assert "gap_name" in data
        assert "t_before_us" in data
        assert "t_after_us" in data
        assert "predicted_gap_us" in data

    def test_work_scaling_is_not_accepted_us_sb_006_evidence(self):
        """Problem-size scaling must not be counted as seeded-gap evidence."""
        data = self._load_results()
        assert data.get("experiment_kind") == "work_scaling_sanity_check"
        assert data.get("satisfies_us_sb_006") is False
        assert "sanity check" in data.get("satisfies_us_sb_006_note", "").lower()

    def test_output_verified(self):
        """Both kernel variants produce correct output (output_verified=True)."""
        data = self._load_results()
        assert data.get("output_verified") is True, (
            f"output_verified must be True, got {data.get('output_verified')}"
        )

    def test_quantification_error_under_20pct(self):
        """The predicted gap matches the measured delta within 20%."""
        data = self._load_results()
        predicted = data["predicted_gap_us"]
        measured = data["measured_delta_us"]
        assert measured > 0, "measured_delta must be positive"
        error = abs(predicted - measured) / measured
        assert error < 0.20, (
            f"quantification_error = {error:.3f} (must be < 0.20). "
            f"predicted={predicted:.2f}, measured={measured:.2f}"
        )

    def test_soundness_both_kernels_pass(self):
        """Both the baseline and scaled kernel produce sound bounds."""
        data = self._load_results()
        assert data.get("baseline_sound") is True, "baseline must be sound"
        assert data.get("scaled_sound") is True, "scaled kernel must be sound"


@requires_counterfactual_gap
class TestAcceptedSeededGapCounterfactualAudit:
    """US-SB-006/008 accepted counterfactual evidence must be explicit."""

    @staticmethod
    def _load_results():
        with open(COUNTERFACTUAL_GAP_RESULTS) as f:
            return json.load(f)

    def test_no_accepted_counterfactual_is_claimed_without_evidence(self):
        data = self._load_results()
        assert data.get("satisfies_us_sb_006") is False
        assert data.get("accepted_results") == []

    def test_acceptance_contract_excludes_work_scaling(self):
        data = self._load_results()
        contract = data["acceptance_contract"]
        assert contract["requires_seeded_gap_intervention"] is True
        assert contract["requires_compiler_reachable_edit"] is True
        assert contract["requires_output_verified"] is True
        assert contract["work_scaling_sanity_checks_do_not_satisfy"] is True

    def test_attempts_record_actual_blockers(self):
        data = self._load_results()
        attempts = data["attempted_results"]
        assert attempts, "expected attempted counterfactual records"
        assert any(a["intervention_kind"] == "mlir_pipe_barrier_removal" for a in attempts)
        assert any(a["intervention_kind"] == "des_json_raise_repeat" for a in attempts)
        assert any(a["intervention_kind"] == "work_scaling_sanity_check" for a in attempts)
        assert all(a["satisfies_us_sb_006"] is False for a in attempts)

    def test_chunk_kda_pipe_barrier_edit_is_vacuous(self):
        data = self._load_results()
        pipe_edit = next(
            a for a in data["attempted_results"]
            if a["intervention_kind"] == "mlir_pipe_barrier_removal"
        )
        assert pipe_edit["mlir_edit_available"] is True
        assert pipe_edit["local_edit_verified"] is True
        assert pipe_edit["barriers_before"] > pipe_edit["barriers_after"]
        assert pipe_edit["local_bound_delta_us"] == pytest.approx(0.0)

    def test_two_limit_hardware_reachability_not_claimed(self):
        data = self._load_results()
        assert data.get("satisfies_us_sb_008") is False
        assert "large-headroom kernel" in data["next_required"]


# ===========================================================================
# US-SB-005: Multi-kernel validation set (n >= 5)
# ===========================================================================

MULTI_KERNEL_RESULTS = (
    PROJECT_ROOT / ".omc" / "research" / "hw_runs" / "multi_kernel_results.json"
)

requires_multi_kernel = pytest.mark.skipif(
    not MULTI_KERNEL_RESULTS.exists(),
    reason="multi_kernel_results.json fixture not present",
)

SOFTMAX_CSV = PROJECT_ROOT / "tests" / "perfbound" / "fixtures" / "softmax_op_summary_910b3.csv"
LAYERNORM_CSV = PROJECT_ROOT / "tests" / "perfbound" / "fixtures" / "layernorm_op_summary_910b3.csv"
RMSNORM_CSV = PROJECT_ROOT / "tests" / "perfbound" / "fixtures" / "rmsnorm_op_summary_910b3.csv"

requires_softmax_csv = pytest.mark.skipif(
    not SOFTMAX_CSV.exists(), reason="softmax op_summary fixture not present"
)
requires_layernorm_csv = pytest.mark.skipif(
    not LAYERNORM_CSV.exists(), reason="layernorm op_summary fixture not present"
)
requires_rmsnorm_csv = pytest.mark.skipif(
    not RMSNORM_CSV.exists(), reason="rmsnorm op_summary fixture not present"
)


@requires_multi_kernel
class TestMultiKernelValidation:
    """US-SB-005: Multi-kernel soundness validation set (n >= 5).

    Validates that the model produces sound bounds (T_bound <= T_measured)
    across at least 5 distinct kernels profiled on the real 910B3.

    Acceptance (US-SB-005):
    - >= 5 kernels with committed T_bound, T_measured, status, tightness
    - soundness_rate == 1.0 (no BOUND_VIOLATION)
    - CI test loading the fixtures passes
    """

    @staticmethod
    def _load_results():
        with open(MULTI_KERNEL_RESULTS) as f:
            return json.load(f)

    def test_at_least_five_kernels(self):
        """The validation set has >= 5 kernels."""
        data = self._load_results()
        assert data["n_kernels"] >= 5, (
            f"Need >= 5 kernels, got {data['n_kernels']}"
        )

    def test_at_least_five_distinct_kernels(self):
        """>= 5 DISTINCT kernels (vector_add shape variants collapse to one).

        US-SB-005 closure requires distinct kernels, not just shape variants.
        Collapses vector_add_16m/32m to a single 'vector_add' family.
        """
        data = self._load_results()
        families = set()
        for k in data["kernels"]:
            name = k["kernel"]
            if name.startswith("vector_add"):
                name = "vector_add"
            else:
                # strip a trailing shape tag like _8kx2k / _16m
                name = name.rsplit("_", 1)[0] if name.rsplit("_", 1)[-1][:1].isdigit() else name
            families.add(name)
        assert len(families) >= 5, (
            f"Need >= 5 distinct kernels, got {len(families)}: {sorted(families)}"
        )

    def test_soundness_rate_is_one(self):
        """soundness_rate == 1.0 (no BOUND_VIOLATION)."""
        data = self._load_results()
        assert data["soundness_rate"] == 1.0, (
            f"soundness_rate must be 1.0, got {data['soundness_rate']}"
        )

    def test_all_kernels_pass(self):
        """Every kernel in the set has status PASS."""
        data = self._load_results()
        for k in data["kernels"]:
            assert k["status"] == "PASS", (
                f"Kernel {k['kernel']} has status {k['status']}, expected PASS"
            )

    def test_all_kernels_sound(self):
        """T_bound <= T_measured for every kernel (bound soundness)."""
        data = self._load_results()
        for k in data["kernels"]:
            assert k["t_bound_us"] <= k["t_measured_us"], (
                f"Kernel {k['kernel']}: T_bound ({k['t_bound_us']:.2f}) "
                f"> T_measured ({k['t_measured_us']:.2f}) — BOUND VIOLATION"
            )

    def test_all_kernels_have_fixture_csvs(self):
        """Every kernel's fixture CSV file exists on disk."""
        data = self._load_results()
        for k in data["kernels"]:
            csv_path = PROJECT_ROOT / k["fixture"]
            assert csv_path.exists(), (
                f"Kernel {k['kernel']} fixture CSV not found: {csv_path}"
            )

    def test_no_remaining_kernels(self):
        """All target kernels have been profiled (remaining == [])."""
        data = self._load_results()
        assert data.get("remaining", []) == [], (
            f"Still missing kernels: {data.get('remaining')}"
        )

    def test_tightness_reasonable(self):
        """All kernels have tightness between 1x and 100x."""
        data = self._load_results()
        for k in data["kernels"]:
            t = k["tightness"]
            assert 1.0 <= t <= 100.0, (
                f"Kernel {k['kernel']}: tightness {t:.2f}x outside [1, 100]"
            )

    def test_kernel_diversity(self):
        """The set contains both memory-bound and compute-bound kernels."""
        data = self._load_results()
        bound_kinds = {k["bound_kind"] for k in data["kernels"]}
        assert len(bound_kinds) >= 2, (
            f"Expected >= 2 bound kinds, got {bound_kinds}"
        )
        assert "analytic_hbm_floor" in bound_kinds, "Missing memory-bound kernel"
        assert "tier2_des" in bound_kinds, "Missing tier2/compute-bound kernel"


@requires_softmax_csv
class TestSoftmaxKernelSoundness:
    """Softmax-specific soundness: T_measured parsed from CSV >= HBM floor."""

    ROWS = 8192
    N_COLS = 2048
    ELEMENT_SIZE = 4  # fp32
    # HBM BW derived from vector_add calibration: 1.525 TB/s
    HBM_BW_BYTES_PER_US = 1.525e6

    def test_softmax_csv_has_kernel_rows(self):
        """The fixture CSV contains softmax_kernel rows."""
        from perfbound.validate.msprof_parser import parse_kernel_time_us
        result = parse_kernel_time_us(str(SOFTMAX_CSV), op_name_filter="softmax_kernel")
        assert result.t_us > 0, "softmax kernel time must be > 0"

    def test_softmax_hbm_floor_soundness(self):
        """T_bound (HBM floor) <= T_measured for softmax."""
        from perfbound.validate.msprof_parser import parse_kernel_time_us
        result = parse_kernel_time_us(str(SOFTMAX_CSV), op_name_filter="softmax_kernel")
        hbm_bytes = 2 * self.ROWS * self.N_COLS * self.ELEMENT_SIZE
        t_bound = hbm_bytes / self.HBM_BW_BYTES_PER_US
        assert t_bound <= result.t_us, (
            f"softmax HBM floor ({t_bound:.2f} us) > T_measured ({result.t_us:.3f} us)"
        )


@requires_layernorm_csv
class TestLayernormKernelSoundness:
    """Layernorm-specific soundness: T_measured parsed from CSV >= HBM floor."""

    ROWS = 8192
    N_COLS = 2048
    ELEMENT_SIZE = 4  # fp32
    HBM_BW_BYTES_PER_US = 1.525e6

    def test_layernorm_csv_has_kernel_rows(self):
        """The fixture CSV contains layernorm_kernel rows."""
        from perfbound.validate.msprof_parser import parse_kernel_time_us
        result = parse_kernel_time_us(str(LAYERNORM_CSV), op_name_filter="layernorm_kernel")
        assert result.t_us > 0, "layernorm kernel time must be > 0"

    def test_layernorm_hbm_floor_soundness(self):
        """T_bound (HBM floor) <= T_measured for layernorm."""
        from perfbound.validate.msprof_parser import parse_kernel_time_us
        result = parse_kernel_time_us(str(LAYERNORM_CSV), op_name_filter="layernorm_kernel")
        hbm_bytes = 2 * self.ROWS * self.N_COLS * self.ELEMENT_SIZE
        t_bound = hbm_bytes / self.HBM_BW_BYTES_PER_US
        assert t_bound <= result.t_us, (
            f"layernorm HBM floor ({t_bound:.2f} us) > T_measured ({result.t_us:.3f} us)"
        )


@requires_rmsnorm_csv
class TestRmsnormKernelSoundness:
    """Rmsnorm-specific soundness: T_measured parsed from CSV >= HBM floor.

    rmsnorm is the 5th distinct kernel (US-SB-005). RMSNorm forward
    (8192x2048 fp32): mean-of-squares reduction + rsqrt scale + weight.
    """

    ROWS = 8192
    N_COLS = 2048
    ELEMENT_SIZE = 4  # fp32
    HBM_BW_BYTES_PER_US = 1.525e6

    def test_rmsnorm_csv_has_kernel_rows(self):
        """The fixture CSV contains rmsnorm_kernel rows."""
        from perfbound.validate.msprof_parser import parse_kernel_time_us
        result = parse_kernel_time_us(str(RMSNORM_CSV), op_name_filter="rmsnorm_kernel")
        assert result.t_us > 0, "rmsnorm kernel time must be > 0"

    def test_rmsnorm_hbm_floor_soundness(self):
        """T_bound (HBM floor) <= T_measured for rmsnorm."""
        from perfbound.validate.msprof_parser import parse_kernel_time_us
        result = parse_kernel_time_us(str(RMSNORM_CSV), op_name_filter="rmsnorm_kernel")
        hbm_bytes = 2 * self.ROWS * self.N_COLS * self.ELEMENT_SIZE
        t_bound = hbm_bytes / self.HBM_BW_BYTES_PER_US
        assert t_bound <= result.t_us, (
            f"rmsnorm HBM floor ({t_bound:.2f} us) > T_measured ({result.t_us:.3f} us)"
        )
