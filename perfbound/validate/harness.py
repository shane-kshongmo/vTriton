# M6 — Validation Harness (NOT part of the model)
#
# The only place compilation + execution happen.  Drives the
# remote-bench-910b3 skill for:
#   1. Soundness:  T_bound ≤ T_measured  (binary, must hold 100%)
#   2. Tightness:  T_measured / T_bound   (record median)
#   3. Counterfactual validation
#
# Integration contract with remote-bench-910b3 skill (§7 of A.0 plan).
#
# Source spec: .omc/specs/performance_bound_model.md §A.6

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..combine.bound_combiner import BoundResult
from ..extract.op_classifier import Component
from .counterfactual import CounterfactualResult, run_counterfactual
from .hivm_edits import HivmEdit
from .msprof_parser import parse_kernel_time_us, parse_component_durations


class ValidationStatus(str, Enum):
    """Tri-state validation outcome."""
    PASS = "pass"                          # T_bound ≤ T_measured
    BOUND_VIOLATION = "bound_violation"    # T_bound > T_measured (model bug)
    EXECUTION_ERROR = "execution_error"    # compile/run/profiler infra failure
    CORRECTNESS_FAILURE = "correctness_failure"  # output mismatch (A.6.2)


@dataclass
class ValidationCase:
    """A single kernel to validate against the bound model.

    Required fields:
        kernel_name: Human-readable kernel identifier.
        profiler_op_name: Op name to match in msprof CSV Op Name column.
        bound_result: Precomputed bound from M5 (BoundResult).

    A.6.1 measurement inputs:
        csv_path: Path to local op_summary CSV (already synced from remote).

    A.6.2 compile/correctness inputs (not used in A.6.1):
        kernel_script: Path to kernel compilation/run script.
        reference_fn: Callable that returns reference output for correctness check.
        rtol/atol: Relative and absolute tolerance for correctness check.
        n_warmup: Number of invocations to discard as warmup (default: 1).
    """
    kernel_name: str
    profiler_op_name: str
    bound_result: BoundResult

    # A.6.1 — measurement inputs
    csv_path: Path | None = None

    # A.6.2 — compile/correctness inputs (not used in A.6.1)
    kernel_script: Path | None = None
    reference_fn: Callable | None = None
    rtol: float = 1e-3
    atol: float = 1e-5

    n_warmup: int = 1


@dataclass
class ValidationResult:
    """Soundness and tightness for a single kernel."""
    kernel_name: str
    t_bound_us: float
    t_measured_us: float

    status: ValidationStatus = ValidationStatus.EXECUTION_ERROR
    tightness: float = 0.0               # T_measured / T_bound

    n_invocations: int = 0               # valid invocations used in median
    task_wait_us: float = 0.0             # median queue/dispatch wait
    component_match: bool | None = None  # measured dominant component matches predicted
    msprof_source: str = ""              # path to op_summary CSV
    notes: str = ""

    @property
    def is_sound(self) -> bool:
        """Backward compat: T_bound ≤ T_measured (PASS status)."""
        return self.status == ValidationStatus.PASS


@dataclass
class ValidationSuite:
    """Results for a full validation suite."""
    results: List[ValidationResult] = field(default_factory=list)

    @property
    def soundness_rate(self) -> float:
        """Fraction of valid measurements where T_bound ≤ T_measured.

        Soundness statistics must exclude non-PASS and non-BOUND_VIOLATION rows
        from the denominator. Only PASS + BOUND_VIOLATION count as valid measurements.
        """
        valid = [
            r for r in self.results
            if r.status in (ValidationStatus.PASS, ValidationStatus.BOUND_VIOLATION)
        ]
        if not valid:
            return 0.0
        return sum(1 for r in valid if r.status == ValidationStatus.PASS) / len(valid)

    @property
    def median_tightness(self) -> float:
        """Median T_measured / T_bound (on valid measurements only)."""
        valid = [
            r for r in self.results
            if r.status in (ValidationStatus.PASS, ValidationStatus.BOUND_VIOLATION)
        ]
        if not valid:
            return 0.0
        sorted_t = sorted(r.tightness for r in valid)
        n = len(sorted_t)
        if n % 2 == 0:
            return (sorted_t[n // 2 - 1] + sorted_t[n // 2]) / 2
        return sorted_t[n // 2]

    @property
    def violations(self) -> List[ValidationResult]:
        """Kernels where the bound was violated (unsound)."""
        return [r for r in self.results if r.status == ValidationStatus.BOUND_VIOLATION]

    def summary(self) -> str:
        valid_count = sum(
            1 for r in self.results
            if r.status in (ValidationStatus.PASS, ValidationStatus.BOUND_VIOLATION)
        )
        pass_count = sum(1 for r in self.results if r.status == ValidationStatus.PASS)
        lines = [
            f"=== Validation Suite Results ===",
            f"Total kernels: {len(self.results)}",
            f"Valid measurements: {valid_count}",
            f"Soundness: {self.soundness_rate:.1%} ({pass_count}/{valid_count})",
            f"Median tightness: {self.median_tightness:.3f}",
        ]
        if self.violations:
            lines.append(f"BOUND VIOLATIONS ({len(self.violations)}):")
            for v in self.violations:
                lines.append(
                    f"  {v.kernel_name}: T_bound={v.t_bound_us:.2f} > T_measured={v.t_measured_us:.2f}"
                )
        # Component match summary
        comp_results = [r for r in self.results if r.component_match is not None]
        if comp_results:
            match_count = sum(1 for r in comp_results if r.component_match)
            lines.append(f"Component match: {match_count}/{len(comp_results)}")
        return "\n".join(lines)


# ── Component match helper ─────────────────────────────────────────────

_COMPONENT_TO_CATEGORY = {
    Component.CUBE: "aicore",
    Component.VECTOR: "aicore",
    Component.MTE_GM: "mte",
    Component.MTE_L1: "mte",
    Component.MTE_UB: "mte",
    Component.SCALAR: "aicpu",
}


def _check_component_match(
    comp_durations: dict[str, float],
    predicted_component: Component | None,
) -> bool | None:
    """Check if measured dominant component matches predicted.

    Args:
        comp_durations: Dict from parse_component_durations().
        predicted_component: BoundResult.binding_component (or None).

    Returns:
        True if dominant measured matches expected, False if mismatch,
        None if task_type fields are all empty (old CSV).
    """
    if predicted_component is None:
        return None

    # Check if all durations are zero (old CSV without task_type)
    total = sum(comp_durations.values())
    if total <= 0:
        return None

    # Find dominant measured category
    dominant_measured = max(comp_durations, key=comp_durations.get)

    # Map predicted component to expected category
    expected_category = _COMPONENT_TO_CATEGORY.get(predicted_component)
    if expected_category is None:
        return None

    return dominant_measured == expected_category


# ── CSV-based validation (Level A, no hardware) ────────────────────────

def validate_from_csv(case: ValidationCase) -> ValidationResult:
    """Validate a single kernel from a local msprof CSV.

    Args:
        case: ValidationCase with csv_path and bound_result.

    Returns:
        ValidationResult with PASS/BOUND_VIOLATION/EXECUTION_ERROR status.
    """
    # Check CSV path
    if case.csv_path is None or not Path(case.csv_path).exists():
        return ValidationResult(
            kernel_name=case.kernel_name,
            t_bound_us=case.bound_result.t_bound_us,
            t_measured_us=0.0,
            status=ValidationStatus.EXECUTION_ERROR,
            notes="csv_path missing or not found",
            msprof_source=str(case.csv_path) if case.csv_path else "",
        )

    # Check bound validity
    if case.bound_result.t_bound_us <= 0:
        return ValidationResult(
            kernel_name=case.kernel_name,
            t_bound_us=case.bound_result.t_bound_us,
            t_measured_us=0.0,
            status=ValidationStatus.EXECUTION_ERROR,
            notes=f"invalid bound: t_bound_us={case.bound_result.t_bound_us} <= 0",
            msprof_source=str(case.csv_path),
        )

    # Parse timing
    try:
        timing = parse_kernel_time_us(
            case.csv_path, case.profiler_op_name, case.n_warmup
        )
    except (ValueError, OSError) as e:
        return ValidationResult(
            kernel_name=case.kernel_name,
            t_bound_us=case.bound_result.t_bound_us,
            t_measured_us=0.0,
            status=ValidationStatus.EXECUTION_ERROR,
            notes=str(e),
            msprof_source=str(case.csv_path),
        )

    # Check bound violation
    is_violation = case.bound_result.t_bound_us > timing.t_us

    # Check component match (filtered to this kernel's rows)
    try:
        comp_durations = parse_component_durations(
            case.csv_path, op_name_filter=case.profiler_op_name
        )
        component_match = _check_component_match(
            comp_durations, case.bound_result.binding_component
        )
    except (ValueError, OSError):
        component_match = None

    return ValidationResult(
        kernel_name=case.kernel_name,
        t_bound_us=case.bound_result.t_bound_us,
        t_measured_us=timing.t_us,
        n_invocations=timing.n_invocations,
        task_wait_us=timing.median_task_wait_us,
        status=ValidationStatus.BOUND_VIOLATION if is_violation else ValidationStatus.PASS,
        tightness=timing.t_us / case.bound_result.t_bound_us,
        msprof_source=str(case.csv_path),
        component_match=component_match,
    )


# ── Remote validation (Level B, hardware) ──────────────────────────────

def run_validation(
    cases: List[ValidationCase],
    remote_host: str | None = None,
    remote_bench_script: str | None = None,
) -> ValidationSuite:
    """Compile, run, profile, and validate a list of kernels.

    Level B (hardware): delegates to scripts/remote_bench.py for remote
    sync + profile. For each case: sync → run msprof → sync back CSV →
    call validate_from_csv(case).

    Args:
        cases: List of ValidationCase to validate.
        remote_host: Remote 910B3 host (SSH). If None, assumes CSVs are local.
        remote_bench_script: Path to scripts/remote_bench.py.

    Returns:
        ValidationSuite with per-kernel soundness/tightness.
    """
    suite = ValidationSuite()

    for case in cases:
        if remote_host and remote_bench_script:
            # Remote execution: sync, profile, fetch CSV
            try:
                import subprocess
                import tempfile

                with tempfile.TemporaryDirectory() as tmpdir:
                    csv_path = Path(tmpdir) / f"{case.kernel_name}_op_summary.csv"

                    # Call remote_bench.py
                    cmd = [
                        "python3", remote_bench_script,
                        "--host", remote_host,
                        "--kernel", case.kernel_name,
                        "--output", str(csv_path),
                    ]
                    if case.kernel_script:
                        cmd.extend(["--script", str(case.kernel_script)])

                    subprocess.run(cmd, check=True, capture_output=True, text=True)

                    # Update case with fetched CSV
                    case.csv_path = csv_path
                    result = validate_from_csv(case)

            except Exception as e:
                # Infrastructure failure → EXECUTION_ERROR (never BOUND_VIOLATION)
                result = ValidationResult(
                    kernel_name=case.kernel_name,
                    t_bound_us=case.bound_result.t_bound_us,
                    t_measured_us=0.0,
                    status=ValidationStatus.EXECUTION_ERROR,
                    notes=f"remote execution failed: {e}",
                )
        else:
            # Local CSV already present
            result = validate_from_csv(case)

        suite.results.append(result)

    return suite


# ── Counterfactual validation (A.6.2) ────────────────────────────────

@dataclass
class CounterfactualCase:
    """A single counterfactual experiment definition.

    Carries the kernel identity, the gap being tested, the predicted gap
    value from the model, the HIVM edit to apply, baseline CSV for
    timing, and optional reference function for correctness checking.
    """
    kernel_name: str
    gap_name: str                        # e.g. "gap3_avoidable_serial"
    predicted_gap_us: float              # from A.5 Attribution
    hivm_edit: HivmEdit                  # edit to apply
    hivm_path: Path | None = None        # original HIVM the edit applies to
    baseline_csv: Path | None = None     # baseline msprof CSV
    profiler_op_name: str | None = None  # op name filter
    reference_fn: Callable | None = None # reference output generator
    reference_args: tuple = ()           # args for reference_fn
    rtol: float = 1e-3
    atol: float = 1e-5


@dataclass
class CounterfactualSuite:
    """Results for a full counterfactual validation suite."""
    results: List[CounterfactualResult] = field(default_factory=list)

    @property
    def valid_count(self) -> int:
        return sum(1 for r in self.results if r.is_valid)

    @property
    def valid_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.valid_count / len(self.results)

    @property
    def median_quantification_error(self) -> float:
        """Median quantification error across valid results."""
        valid = [r.quantification_error for r in self.results if r.output_verified]
        if not valid:
            return float("inf")
        sorted_err = sorted(valid)
        n = len(sorted_err)
        if n % 2 == 0:
            return (sorted_err[n // 2 - 1] + sorted_err[n // 2]) / 2
        return sorted_err[n // 2]

    def summary(self) -> str:
        lines = [
            f"=== Counterfactual Suite Results ===",
            f"Total cases: {len(self.results)}",
            f"Valid: {self.valid_count} ({self.valid_rate:.1%})",
            f"Median quantification error: {self.median_quantification_error:.3f}",
        ]
        for r in self.results:
            status = "VALID" if r.is_valid else "INVALID"
            lines.append(
                f"  {r.kernel_name}/{r.gap_name}: {status} "
                f"(predicted={r.predicted_gap_us:.2f}, "
                f"measured={r.measured_delta_us:.2f}, "
                f"error={r.quantification_error:.3f})"
            )
        return "\n".join(lines)


def run_counterfactual_suite(
    cases: List[CounterfactualCase],
    remote_host: str | None = None,
    remote_bench_script: str | None = None,
) -> CounterfactualSuite:
    """Run a suite of counterfactual experiments.

    Args:
        cases: List of CounterfactualCase to run.
        remote_host: Remote 910B3 SSH host (None for local).
        remote_bench_script: Path to scripts/remote_bench.py.

    Returns:
        CounterfactualSuite with per-case results.
    """
    suite = CounterfactualSuite()

    for case in cases:
        result = run_counterfactual(
            kernel_name=case.kernel_name,
            gap_name=case.gap_name,
            predicted_gap_us=case.predicted_gap_us,
            hivm_edit=case.hivm_edit,
            hivm_path=case.hivm_path,
            baseline_csv=case.baseline_csv,
            profiler_op_name=case.profiler_op_name,
            remote_host=remote_host,
            remote_bench_script=remote_bench_script,
            reference_fn=case.reference_fn,
            reference_args=case.reference_args,
            rtol=case.rtol,
            atol=case.atol,
        )
        suite.results.append(result)

    return suite
