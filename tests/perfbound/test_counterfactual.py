# Tests for counterfactual validation (A.6.2)
#
# Mocked end-to-end: valid (<20%), invalid (output corrupt),
# invalid (error>20%), infra-error path.
#
# Source spec: .omc/plans/a6_2_counterfactual.md §Testability

import pytest
from pathlib import Path

from perfbound.validate.counterfactual import (
    CounterfactualResult,
    run_counterfactual,
)
from perfbound.validate.hivm_edits import HivmEdit, raise_repeat


# ── Fixtures / helpers ─────────────────────────────────────────────

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE_CSV = FIXTURE_DIR / "op_summary_sample.csv"
SAMPLE_HIVM = FIXTURE_DIR / "sample_hivm.json"


def _mock_profile_baseline(**kwargs) -> float:
    """Return a fixed baseline timing."""
    return 1000.0


def _mock_compile_and_profile_ok(**kwargs) -> tuple[float, object]:
    """Return a fixed post-edit timing (improvement) and dummy output."""
    return 780.0, None  # 220 us improvement


def _mock_compile_and_profile_small(**kwargs) -> tuple[float, object]:
    """Return a very small improvement."""
    return 990.0, None  # 10 us improvement (barely any)


def _mock_edit(**kwargs) -> Path:
    """Return a dummy edited path."""
    return Path("/tmp/edited.hivm")


def _mock_verify_true(*args, **kwargs) -> bool:
    return True


def _mock_verify_false(*args, **kwargs) -> bool:
    return False


def _make_edit() -> HivmEdit:
    return HivmEdit(
        gap_name="gap3_avoidable_serial",
        description="Insert ping-pong buffer for MTE_UB serialization",
        apply=lambda p: Path("/tmp/edited.hivm"),
    )


# ── Tests ──────────────────────────────────────────────────────────

class TestCounterfactualResult:
    """CounterfactualResult properties."""

    def test_quantification_error(self):
        r = CounterfactualResult(
            kernel_name="k", gap_name="g",
            predicted_gap_us=200.0,
            t_before_us=1000.0, t_after_us=800.0,
            measured_delta_us=200.0,
            output_verified=True,
        )
        assert r.quantification_error == pytest.approx(0.0)

    def test_quantification_error_10pct(self):
        r = CounterfactualResult(
            kernel_name="k", gap_name="g",
            predicted_gap_us=220.0,
            t_before_us=1000.0, t_after_us=800.0,
            measured_delta_us=200.0,
            output_verified=True,
        )
        assert r.quantification_error == pytest.approx(0.10)

    def test_is_valid_under_20pct(self):
        r = CounterfactualResult(
            kernel_name="k", gap_name="g",
            predicted_gap_us=210.0,
            t_before_us=1000.0, t_after_us=800.0,
            measured_delta_us=200.0,
            output_verified=True,
        )
        # error = |210-200|/200 = 0.05 < 0.20
        assert r.is_valid is True

    def test_is_invalid_over_20pct(self):
        r = CounterfactualResult(
            kernel_name="k", gap_name="g",
            predicted_gap_us=300.0,
            t_before_us=1000.0, t_after_us=800.0,
            measured_delta_us=200.0,
            output_verified=True,
        )
        # error = |300-200|/200 = 0.50 > 0.20
        assert r.is_valid is False

    def test_is_invalid_unverified(self):
        r = CounterfactualResult(
            kernel_name="k", gap_name="g",
            predicted_gap_us=200.0,
            t_before_us=1000.0, t_after_us=800.0,
            measured_delta_us=200.0,
            output_verified=False,
        )
        assert r.is_valid is False

    def test_zero_delta_inf_error(self):
        r = CounterfactualResult(
            kernel_name="k", gap_name="g",
            predicted_gap_us=200.0,
            t_before_us=1000.0, t_after_us=1000.0,
            measured_delta_us=0.0,
            output_verified=True,
        )
        assert r.quantification_error == float("inf")
        assert r.is_valid is False


class TestRunCounterfactual:
    """run_counterfactual with mocked infra."""

    def test_valid_counterfactual_under_20pct(self):
        """predicted=220, measured_delta=220 → error=0% → valid."""
        result = run_counterfactual(
            kernel_name="chunk_kda",
            gap_name="gap3_avoidable_serial",
            predicted_gap_us=220.0,
            hivm_edit=_make_edit(),
            _profile_fn=lambda **kw: 1000.0,
            _compile_and_profile_fn=lambda **kw: (780.0, None),
            _edit_fn=lambda edit: Path("/tmp/edited.hivm"),
            _verify_fn=lambda *a, **kw: True,
        )
        assert result.output_verified is True
        assert result.t_before_us == 1000.0
        assert result.t_after_us == 780.0
        assert result.measured_delta_us == 220.0
        assert result.is_valid is True
        assert result.quantification_error == pytest.approx(0.0)

    def test_invalid_error_over_20pct(self):
        """predicted=500, measured_delta=200 → error=150% → invalid."""
        result = run_counterfactual(
            kernel_name="chunk_kda",
            gap_name="gap3_avoidable_serial",
            predicted_gap_us=500.0,
            hivm_edit=_make_edit(),
            _profile_fn=lambda **kw: 1000.0,
            _compile_and_profile_fn=lambda **kw: (800.0, None),
            _edit_fn=lambda edit: Path("/tmp/edited.hivm"),
            _verify_fn=lambda *a, **kw: True,
        )
        assert result.output_verified is True
        assert result.measured_delta_us == 200.0
        assert result.is_valid is False
        assert result.quantification_error == pytest.approx(1.5)

    def test_invalid_corrupt_output(self):
        """Output verification fails → is_valid=False."""
        result = run_counterfactual(
            kernel_name="chunk_kda",
            gap_name="gap3_avoidable_serial",
            predicted_gap_us=220.0,
            hivm_edit=_make_edit(),
            _profile_fn=lambda **kw: 1000.0,
            _compile_and_profile_fn=lambda **kw: (780.0, None),
            _edit_fn=lambda edit: Path("/tmp/edited.hivm"),
            _verify_fn=lambda *a, **kw: False,
        )
        assert result.output_verified is False
        assert result.is_valid is False
        assert "output verification failed" in result.notes

    def test_infra_error_baseline_fails(self):
        """Baseline profiling fails → non-valid result with notes."""
        def failing_profile(**kw):
            raise RuntimeError("msprof crashed")

        result = run_counterfactual(
            kernel_name="chunk_kda",
            gap_name="gap3_avoidable_serial",
            predicted_gap_us=220.0,
            hivm_edit=_make_edit(),
            _profile_fn=failing_profile,
            _compile_and_profile_fn=lambda **kw: (780.0, None),
            _edit_fn=lambda edit: Path("/tmp/edited.hivm"),
            _verify_fn=lambda *a, **kw: True,
        )
        assert result.output_verified is False
        assert result.measured_delta_us == 0.0
        assert "baseline profiling failed" in result.notes
        assert result.is_valid is False

    def test_infra_error_compile_fails(self):
        """Compile step fails → non-valid result with notes."""
        def failing_compile(**kw):
            raise RuntimeError("bishengir assertion failure")

        result = run_counterfactual(
            kernel_name="chunk_kda",
            gap_name="gap3_avoidable_serial",
            predicted_gap_us=220.0,
            hivm_edit=_make_edit(),
            _profile_fn=lambda **kw: 1000.0,
            _compile_and_profile_fn=failing_compile,
            _edit_fn=lambda edit: Path("/tmp/edited.hivm"),
            _verify_fn=lambda *a, **kw: True,
        )
        assert result.output_verified is False
        assert result.t_before_us == 1000.0  # baseline was measured
        assert result.measured_delta_us == 0.0
        assert "compile/profile failed" in result.notes

    def test_infra_error_edit_fails(self):
        """HIVM edit step fails → non-valid result with notes."""
        def failing_edit(edit):
            raise ValueError("malformed HIVM: no operations key")

        result = run_counterfactual(
            kernel_name="chunk_kda",
            gap_name="gap3_avoidable_serial",
            predicted_gap_us=220.0,
            hivm_edit=_make_edit(),
            _profile_fn=lambda **kw: 1000.0,
            _compile_and_profile_fn=lambda **kw: (780.0, None),
            _edit_fn=failing_edit,
            _verify_fn=lambda *a, **kw: True,
        )
        assert result.output_verified is False
        assert "HIVM edit failed" in result.notes

    def test_never_spurious_small_delta_on_infra_error(self):
        """Infra failure must NOT produce a small measured_delta."""
        def failing_compile(**kw):
            raise RuntimeError("crash")

        result = run_counterfactual(
            kernel_name="chunk_kda",
            gap_name="gap3_avoidable_serial",
            predicted_gap_us=220.0,
            hivm_edit=_make_edit(),
            _profile_fn=lambda **kw: 1000.0,
            _compile_and_profile_fn=failing_compile,
            _edit_fn=lambda edit: Path("/tmp/edited.hivm"),
            _verify_fn=lambda *a, **kw: True,
        )
        # measured_delta is 0.0, NOT a small spurious value
        assert result.measured_delta_us == 0.0
        assert result.is_valid is False


class TestProductionApplyPath:
    """Exercise the real _default_apply_edit (no _edit_fn injection)."""

    def test_real_edit_applied_to_real_hivm(self):
        """With hivm_path + a real edit, the production apply path runs the
        edit against the actual HIVM file (regression for the _default_apply_edit
        wiring bug that passed Path('.'))."""
        edit = HivmEdit(
            gap_name="gap4_intra_unit_exec",
            description="Raise repeat 2x",
            apply=lambda p: raise_repeat(p, factor=2),
        )
        captured = {}

        def _capture_compile(**kw):
            captured["edited_hivm_path"] = kw.get("edited_hivm_path")
            return 780.0, None

        result = run_counterfactual(
            kernel_name="chunk_kda",
            gap_name="gap4_intra_unit_exec",
            predicted_gap_us=220.0,
            hivm_edit=edit,
            hivm_path=SAMPLE_HIVM,                 # real HIVM, real edit
            _profile_fn=lambda **kw: 1000.0,
            _compile_and_profile_fn=_capture_compile,
            _verify_fn=lambda *a, **kw: True,
        )
        # The edit actually produced a real edited file (not Path('.')).
        assert result.measured_delta_us == 220.0
        assert result.is_valid is True
        edited_path = captured["edited_hivm_path"]
        assert edited_path is not None and Path(edited_path).exists()

    def test_missing_hivm_path_surfaces_as_edit_failure(self):
        """Production path with no hivm_path and no _edit_fn → edit step fails
        loudly as an infra error, not a spurious delta."""
        edit = HivmEdit(
            gap_name="gap4_intra_unit_exec",
            description="Raise repeat 2x",
            apply=lambda p: raise_repeat(p, factor=2),
        )
        result = run_counterfactual(
            kernel_name="chunk_kda",
            gap_name="gap4_intra_unit_exec",
            predicted_gap_us=220.0,
            hivm_edit=edit,
            hivm_path=None,                        # missing → _default_apply_edit raises
            _profile_fn=lambda **kw: 1000.0,
            _compile_and_profile_fn=lambda **kw: (780.0, None),
            _verify_fn=lambda *a, **kw: True,
        )
        assert "HIVM edit failed" in result.notes
        assert result.measured_delta_us == 0.0
        assert result.is_valid is False


class TestCounterfactualWithRealCSV:
    """Counterfactual using real sample CSV for baseline timing."""

    def test_baseline_from_real_csv(self):
        """Baseline timing parsed from sample CSV.

        sample CSV has 3 AI_CORE rows for target_kernel (1000, 1050, 5000).
        parse_kernel_time_us with default n_warmup=1 discards the first
        invocation, leaving [1050, 5000]; median = 3025.0.
        """
        result = run_counterfactual(
            kernel_name="test_kernel",
            gap_name="gap3_avoidable_serial",
            predicted_gap_us=500.0,
            hivm_edit=_make_edit(),
            baseline_csv=SAMPLE_CSV,
            profiler_op_name="target_kernel",
            _compile_and_profile_fn=lambda **kw: (2500.0, None),
            _edit_fn=lambda edit: Path("/tmp/edited.hivm"),
            _verify_fn=lambda *a, **kw: True,
        )
        assert result.t_before_us == 3025.0  # from sample CSV (warmup=1)
        assert result.t_after_us == 2500.0
        assert result.measured_delta_us == 525.0
        # predicted=500, measured=525, error = 25/525 = 0.0476 < 0.20
        assert result.is_valid is True


class TestDefaultProfileBaselineRemoteDispatch:
    """Verify _default_profile_baseline dispatches to remote_bench."""

    def test_remote_dispatch_called_when_no_csv(self):
        """When no baseline_csv, _default_profile_baseline calls remote_bench."""
        from unittest.mock import patch, MagicMock
        import sys
        from perfbound.validate.counterfactual import _default_profile_baseline

        mock_csv = Path("/tmp/mock_csv.csv")

        # Build a mock remote_bench module and inject it into sys.modules
        # so that ``importlib.import_module("scripts.remote_bench")`` returns it.
        mock_rb = MagicMock()
        mock_rb.run_remote_bench.return_value = (mock_csv, None)

        # Build a mock parse function and inject it into the source module
        # so that ``from ..validate.msprof_parser import parse_kernel_time_us``
        # picks up the mock via the inline import.
        mock_parse = MagicMock(return_value=MagicMock(t_us=1234.0))

        with patch.dict(sys.modules, {"scripts.remote_bench": mock_rb}), \
             patch("perfbound.validate.msprof_parser.parse_kernel_time_us", mock_parse):
            result = _default_profile_baseline(
                baseline_csv=None,
                profiler_op_name="test_op",
                remote_host="user@host",
                remote_bench_script="scripts/remote_bench.py",
                kernel_name="test_kernel",
            )

        mock_rb.run_remote_bench.assert_called_once()
        call_kwargs = mock_rb.run_remote_bench.call_args[1]
        assert call_kwargs["remote_host"] == "user@host"
        assert call_kwargs["kernel_name"] == "test_kernel"
        assert result == 1234.0

    def test_csv_takes_precedence_over_remote(self):
        """baseline_csv is used even when remote_host is provided."""
        from perfbound.validate.counterfactual import _default_profile_baseline

        # With a real CSV, the remote path should not be invoked
        result = _default_profile_baseline(
            baseline_csv=SAMPLE_CSV,
            profiler_op_name="target_kernel",
            remote_host="user@host",
            remote_bench_script="scripts/remote_bench.py",
            kernel_name="test_kernel",
        )
        # From sample CSV: warmup=1 discards first (1000), remaining [1050, 5000], median=3025
        assert result == 3025.0


class TestDefaultCompileAndProfileRemoteDispatch:
    """Verify _default_compile_and_profile dispatches to remote_bench."""

    def test_remote_dispatch_called(self):
        """_default_compile_and_profile calls remote_bench with hivm_in."""
        from unittest.mock import patch, MagicMock
        import sys
        from perfbound.validate.counterfactual import _default_compile_and_profile

        mock_csv = Path("/tmp/mock_csv.csv")
        mock_npy = Path("/tmp/mock_output.npy")

        mock_rb = MagicMock()
        mock_rb.run_remote_bench.return_value = (mock_csv, mock_npy)

        mock_parse = MagicMock(return_value=MagicMock(t_us=800.0))

        with patch.dict(sys.modules, {"scripts.remote_bench": mock_rb}), \
             patch("perfbound.validate.msprof_parser.parse_kernel_time_us", mock_parse):
            t_us, output = _default_compile_and_profile(
                edited_hivm_path=Path("/tmp/edited.npuir.mlir"),
                remote_host="user@host",
                remote_bench_script="scripts/remote_bench.py",
                kernel_name="test_kernel",
                profiler_op_name="test_op",
            )

        mock_rb.run_remote_bench.assert_called_once()
        call_kwargs = mock_rb.run_remote_bench.call_args[1]
        assert call_kwargs["remote_host"] == "user@host"
        assert call_kwargs["kernel_name"] == "test_kernel"
        assert "hivm_in" in call_kwargs
        assert t_us == 800.0
