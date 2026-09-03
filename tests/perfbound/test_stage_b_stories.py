# Tests for Stage-B counterfactual stories: US-SB-006 and US-SB-008
#
# US-SB-008: Two-limit compiler-headroom validation (seeded_serial TTAdapter edit).
# US-SB-006: accepted seeded-gap counterfactual audit.

import json
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    """US-SB-006/008 accepted seeded counterfactual evidence must be explicit."""

    @staticmethod
    def _load_results():
        with open(COUNTERFACTUAL_GAP_RESULTS) as f:
            return json.load(f)

    def test_no_accepted_counterfactual_is_claimed_without_evidence(self):
        data = self._load_results()
        assert data.get("satisfies_us_sb_006") is True
        assert data.get("satisfies_us_sb_008") is True
        accepted = data.get("accepted_results")
        assert accepted, "expected accepted seeded counterfactual evidence"
        assert any(
            r["kernel_name"] == "seeded_serial"
            and r["output_verified"] is True
            and r["quantification_error"] < 0.20
            and r["compiler_ir_profiled"] is True
            and r["satisfies_us_sb_008"] is True
            for r in accepted
        )

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
        assert any(
            a["intervention_kind"] == "edited_npuir_pipe_barrier_removal"
            for a in attempts
        )
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

    def test_two_limit_hardware_reachability_is_claimed_with_evidence(self):
        data = self._load_results()
        assert data.get("satisfies_us_sb_008") is True
        accepted = data["accepted_results"][0]
        assert accepted["t_bound_hivm_us"] <= accepted["t_bound_dsl_us"]
        assert accepted["t_bound_dsl_us"] <= accepted["t_measured_us"]
        assert accepted["local_bound_delta_us"] == pytest.approx(
            accepted["measured_delta_us"], rel=0.20
        )
        assert accepted["compiler_ir_profiled"] is True
        assert accepted["edited_npuir_profiled"] is False
        assert accepted["satisfies_us_sb_008"] is True
        assert data["next_required"] is None


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
