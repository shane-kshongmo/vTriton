# M5 — Per-Kernel Report (text + JSON)
#
# Deliverable: bound, binding tier/component, five-way attribution,
# two-limit gap, single recommended action.
#
# Source spec: .omc/specs/performance_bound_model.md §A.5

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .bound_combiner import BoundResult, BindingTier
from .two_limit import TwoLimitResult

if TYPE_CHECKING:
    from ..calibration.constants import CalibrationDB


_RECOMMENDATIONS = {
    "grid": "Fix grid partitioning — increase occupancy or load balance",
    "gap1_wrong_unit": "Fix DSL types — move ops to eligible unit",
    "gap2_coalescing": "Merge transfers — increase transfer size to reduce amortization",
    "gap3_avoidable_serial": "Add ping-pong buffer to overlap this handoff",
    "gap4_intra_unit_exec": "Increase SIMD repeat/mask utilization",
}

# Threshold below which all gaps are considered negligible ("at bound")
_AT_BOUND_EPS = 1e-4


@dataclass
class KernelReport:
    """Complete per-kernel performance bound report."""
    kernel_name: str

    # Bound
    t_bound_us: float
    binding_tier: str
    binding_component: Optional[str] = None

    # Decomposed
    t_grid_floor_us: float = 0.0
    t_core_floor_us: float = 0.0
    t_serial_irreducible_us: float = 0.0

    # Two-limit (A.7)
    t_bound_hivm_us: Optional[float] = None
    t_bound_dsl_us: Optional[float] = None
    t_measured_us: Optional[float] = None
    compiler_headroom_us: Optional[float] = None
    author_headroom_us: Optional[float] = None

    # Measurement metadata
    msprof_source: Optional[str] = None
    n_invocations: Optional[int] = None
    component_match: Optional[bool] = None

    # Calibration provenance
    calibration_source: Optional[str] = None
    calibration_version: Optional[str] = None
    calibration_hardware_name: Optional[str] = None
    calibration_measured_constant_count: int = 0
    calibration_derived_constant_count: int = 0
    calibration_max_measured_ci_rel: Optional[float] = None
    calibration_p0_complete: Optional[bool] = None
    calibration_p0_violations: list[str] = field(default_factory=list)
    calibration_warnings: list[str] = field(default_factory=list)
    calibration_fallbacks: list[str] = field(default_factory=list)

    # Profile diagnosis (from profile_utilization)
    profile_diagnosis: Optional[str] = None
    profile_dominant_component: Optional[str] = None
    exposed_control_frac_measured: Optional[float] = None
    exposed_control_frac_model: Optional[float] = None
    exposed_control_deficit_pts: Optional[float] = None
    exposed_control_deficit_us: Optional[float] = None
    n_sync_ops: Optional[int] = None

    # Attainable-headroom assessment. The measured-minus-bound residual is not
    # itself a realizable speedup claim; without a correctness-verified
    # counterfactual, report only a diagnostic interval and no point estimate.
    headroom_status: str = "unavailable"
    recoverable_headroom_lower_us: Optional[float] = None
    recoverable_headroom_upper_us: Optional[float] = None
    recoverable_headroom_estimate_us: Optional[float] = None
    headroom_confidence: str = "none"
    headroom_method: str = (
        "No correctness-verified counterfactual measurement is available."
    )
    potential_speedup_upper: Optional[float] = None

    # Attribution (five-way, fractions of T_bound)
    attribution: dict[str, float] = field(default_factory=dict)
    attribution_us: dict[str, float] = field(default_factory=dict)

    # Recommendation
    recommended_action: str = "unknown"

    def to_dict(self) -> dict:
        base = {
            "kernel_name": self.kernel_name,
            "t_bound_us": self.t_bound_us,
            "binding_tier": self.binding_tier,
            "binding_component": self.binding_component,
            "t_grid_floor_us": self.t_grid_floor_us,
            "t_core_floor_us": self.t_core_floor_us,
            "t_serial_irreducible_us": self.t_serial_irreducible_us,
            "t_bound_hivm_us": self.t_bound_hivm_us,
            "t_measured_us": self.t_measured_us,
            "compiler_headroom_us": self.compiler_headroom_us,
            "author_headroom_us": self.author_headroom_us,
            "attribution": self.attribution,
            "attribution_us": self.attribution_us,
            "calibration": {
                "source": self.calibration_source,
                "version": self.calibration_version,
                "hardware_name": self.calibration_hardware_name,
                "measured_constant_count": self.calibration_measured_constant_count,
                "derived_constant_count": self.calibration_derived_constant_count,
                "max_measured_ci_rel": self.calibration_max_measured_ci_rel,
                "p0_complete": self.calibration_p0_complete,
                "p0_violations": self.calibration_p0_violations,
                "warnings": self.calibration_warnings,
                "fallbacks": self.calibration_fallbacks,
            },
            "profile": {
                "diagnosis": self.profile_diagnosis,
                "dominant_component": self.profile_dominant_component,
                "exposed_control_frac_measured": self.exposed_control_frac_measured,
                "exposed_control_frac_model": self.exposed_control_frac_model,
                "exposed_control_deficit_pts": self.exposed_control_deficit_pts,
                "exposed_control_deficit_us": self.exposed_control_deficit_us,
                "n_sync_ops": self.n_sync_ops,
            } if self.profile_diagnosis is not None else None,
            "headroom_assessment": {
                "status": self.headroom_status,
                "lower_us": self.recoverable_headroom_lower_us,
                "upper_us": self.recoverable_headroom_upper_us,
                "point_estimate_us": self.recoverable_headroom_estimate_us,
                "confidence": self.headroom_confidence,
                "method": self.headroom_method,
                "potential_speedup_upper": self.potential_speedup_upper,
            },
            "recommended_action": self.recommended_action,
        }
        # A.6.1 reachability block
        is_violation = (
            self.t_measured_us is not None
            and self.t_bound_dsl_us is not None
            and self.t_bound_dsl_us > self.t_measured_us
        )
        base["reachability"] = {
            "t_bound_hivm_us": self.t_bound_hivm_us,
            "t_bound_dsl_us": self.t_bound_dsl_us,
            "t_measured_us": self.t_measured_us,
            "compiler_headroom_us": self.compiler_headroom_us,
            "author_headroom_us": self.author_headroom_us,
            "author_residual_us": self.author_headroom_us,
            "is_violation": is_violation,
            "msprof_source": self.msprof_source,
            "n_invocations": self.n_invocations,
            "component_match": self.component_match,
        }
        return base

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialize to JSON string, optionally writing to a file."""
        text = json.dumps(self.to_dict(), indent=2)
        if path:
            Path(path).write_text(text)
        return text

    def to_text(self) -> str:
        """Human-readable text report."""
        lines = [
            f"=== Performance Bound Report: {self.kernel_name} ===",
            f"",
            f"T_bound:   {self.t_bound_us:.2f} us",
            f"  Tier 1 (grid):      {self.t_grid_floor_us:.2f} us",
            f"  Tier 2 (component): {self.t_core_floor_us:.2f} us",
            f"  Serial irreducible: {self.t_serial_irreducible_us:.2f} us",
            f"",
            f"Binding: {self.binding_tier}",
        ]
        if self.binding_component:
            lines.append(f"  Component: {self.binding_component}")

        if self.calibration_version or self.calibration_source:
            lines.extend([
                "",
                "Calibration:",
                f"  source:   {self.calibration_source or 'unknown'}",
                f"  version:  {self.calibration_version or 'unknown'}",
                f"  hardware: {self.calibration_hardware_name or 'unknown'}",
                (
                    "  P0 status: complete"
                    if self.calibration_p0_complete
                    else "  P0 status: incomplete"
                ),
                (
                    "  measured constants: "
                    f"{self.calibration_measured_constant_count}"
                    f" (max relative 95% CI: "
                    f"{self.calibration_max_measured_ci_rel:.2%})"
                    if self.calibration_max_measured_ci_rel is not None
                    else (
                        "  measured constants: "
                        f"{self.calibration_measured_constant_count}"
                    )
                ),
            ])
            for violation in self.calibration_p0_violations:
                lines.append(f"  P0 violation: {violation}")
            for warning in self.calibration_warnings:
                lines.append(f"  warning: {warning}")
            for fallback in self.calibration_fallbacks:
                lines.append(f"  diagnostic fallback: {fallback}")

        lines.append(f"")
        lines.append(f"Attribution (absolute and fraction of T_bound):")
        for gap_name, frac in sorted(self.attribution.items(), key=lambda x: -x[1]):
            gap_us = self.attribution_us.get(gap_name, 0.0)
            lines.append(f"  {gap_name}: {gap_us:.2f} us ({frac:.3f})")

        # Reachability Hierarchy (three-level)
        lines.append(f"")
        lines.append(f"Reachability Hierarchy:")
        lines.append(f"  1. Hardware floor  (T_bound_HIVM):  "
                     f"{self.t_bound_hivm_us:.2f} us" if self.t_bound_hivm_us is not None
                     else "  1. Hardware floor  (T_bound_HIVM):  N/A")

        dsl_line = f"  2. DSL bound       (T_bound_DSL):   "
        if self.t_bound_dsl_us is not None:
            dsl_line += f"{self.t_bound_dsl_us:.2f} us"
            if self.compiler_headroom_us is not None:
                dsl_line += f"   [compiler headroom: {self.compiler_headroom_us:.2f} us]"
        else:
            dsl_line += "N/A"
        lines.append(dsl_line)

        meas_line = f"  3. Measured        (T_measured):    "
        if self.t_measured_us is not None:
            # Check for bound violation
            if (self.t_bound_dsl_us is not None
                    and self.t_bound_dsl_us > self.t_measured_us):
                meas_line += (
                    f"{self.t_measured_us:.2f} us   "
                    f"*** BOUND VIOLATION: T_bound={self.t_bound_dsl_us:.2f} > T_measured ***"
                )
            else:
                meas_line += f"{self.t_measured_us:.2f} us"
                if self.author_headroom_us is not None:
                    meas_line += (
                        "   [author residual, not proven attainable: "
                        f"{self.author_headroom_us:.2f} us]"
                    )
            lines.append(meas_line)
            # Source + invocations
            source_line = ""
            if self.msprof_source:
                source_line += f"     source: {self.msprof_source}"
            if self.n_invocations is not None:
                source_line += f"  n={self.n_invocations} invocations"
            if source_line:
                lines.append(source_line)
            # Component match
            if self.component_match is not None:
                match_sym = "✓" if self.component_match else "✗"
                pred = self.binding_component if self.binding_component else "unknown"
                lines.append(
                    f"     coarse task-category match: predicted={pred}, match={match_sym}"
                )
        else:
            meas_line += "not yet measured"
            lines.append(meas_line)

        if self.profile_diagnosis:
            lines.append(f"")
            lines.append(f"Profile Diagnosis:")
            lines.append(f"  diagnosis:          {self.profile_diagnosis}")
            if self.profile_dominant_component:
                lines.append(f"  dominant_component: {self.profile_dominant_component}")
            if self.exposed_control_frac_measured is not None:
                lines.append(
                    f"  scalar_frac_meas:   {self.exposed_control_frac_measured:.1%}"
                )
            if self.exposed_control_deficit_pts is not None:
                lines.append(
                    f"  control_deficit:    +{self.exposed_control_deficit_pts * 100:.1f} pts"
                )
            if self.exposed_control_deficit_us is not None:
                lines.append(
                    f"  deficit_us:         ~{self.exposed_control_deficit_us:.0f} us"
                )
            if self.n_sync_ops is not None:
                lines.append(f"  n_sync_ops:         {self.n_sync_ops}")

        lines.extend([
            "",
            "Attainable Headroom Assessment:",
            f"  status:     {self.headroom_status}",
            f"  confidence: {self.headroom_confidence}",
        ])
        if (
            self.recoverable_headroom_lower_us is not None
            and self.recoverable_headroom_upper_us is not None
        ):
            lines.append(
                "  diagnostic range: "
                f"{self.recoverable_headroom_lower_us:.2f}.."
                f"{self.recoverable_headroom_upper_us:.2f} us"
            )
        if self.recoverable_headroom_estimate_us is None:
            lines.append("  point estimate: unavailable")
        else:
            lines.append(
                f"  point estimate: {self.recoverable_headroom_estimate_us:.2f} us"
            )
        if self.potential_speedup_upper is not None:
            lines.append(
                f"  diagnostic speedup upper bound: {self.potential_speedup_upper:.2f}x"
            )
        lines.append(f"  method: {self.headroom_method}")

        lines.append(f"")
        lines.append(f"Recommended action: {self.recommended_action}")

        return "\n".join(lines)

    @classmethod
    def from_bound(cls, result: BoundResult,
                   two_limit: Optional[TwoLimitResult] = None) -> "KernelReport":
        """Create a report from a BoundResult."""
        dominant_name, dominant_frac = result.attribution.dominant_gap()

        # At-bound detection: when all gap fractions are below ε, the kernel
        # is at its analytical bound — no actionable software gap remains.
        total_gap_frac = (
            result.attribution.grid_gap_frac
            + result.attribution.gap1_frac
            + result.attribution.gap2_frac
            + result.attribution.gap3_frac
            + result.attribution.gap4_frac
        )
        if total_gap_frac < _AT_BOUND_EPS:
            action = (
                "At component bound — no actionable software gap "
                "(consider algorithmic redesign: fusion/precision/less traffic)"
            )
        else:
            action = _RECOMMENDATIONS.get(dominant_name, "Profile to identify bottleneck")

        return cls(
            kernel_name=result.kernel_name,
            t_bound_us=result.t_bound_us,
            binding_tier=result.binding_tier.value,
            binding_component=result.binding_component.value if result.binding_component else None,
            t_grid_floor_us=result.t_grid_floor_us,
            t_core_floor_us=result.t_core_floor_us,
            t_serial_irreducible_us=result.t_serial_irreducible_us,
            t_bound_hivm_us=two_limit.t_bound_hivm_us if two_limit else None,
            t_bound_dsl_us=two_limit.t_bound_dsl_us if two_limit else result.t_bound_us,
            t_measured_us=two_limit.t_measured_us if two_limit else None,
            compiler_headroom_us=two_limit.compiler_headroom_us if two_limit else None,
            author_headroom_us=two_limit.author_headroom_us if two_limit else None,
            attribution={
                "grid": result.attribution.grid_gap_frac,
                "gap1_wrong_unit": result.attribution.gap1_frac,
                "gap2_coalescing": result.attribution.gap2_frac,
                "gap3_avoidable_serial": result.attribution.gap3_frac,
                "gap4_intra_unit_exec": result.attribution.gap4_frac,
            },
            attribution_us={
                "grid": result.attribution.grid_gap_us,
                "gap1_wrong_unit": result.attribution.gap1_wrong_unit_us,
                "gap2_coalescing": result.attribution.gap2_coalescing_us,
                "gap3_avoidable_serial": result.attribution.gap3_avoidable_serial_us,
                "gap4_intra_unit_exec": result.attribution.gap4_intra_unit_exec_us,
            },
            recommended_action=action,
        )

    def merge_validation(
        self,
        t_measured_us: float,
        msprof_source: str = "",
        n_invocations: int = 0,
        component_match: bool | None = None,
    ) -> None:
        """Merge measurement provenance from ValidationResult into this report.

        Called by run_report when a measured CSV is provided.  Copies the
        three provenance fields (msprof_source, n_invocations, component_match)
        and updates t_measured_us + author_headroom_us.

        Args:
            t_measured_us: Measured kernel time from msprof.
            msprof_source: Path to op_summary CSV.
            n_invocations: Valid invocations used in median.
            component_match: Whether dominant measured component matches predicted.
        """
        self.t_measured_us = t_measured_us
        self.msprof_source = msprof_source or None
        self.n_invocations = n_invocations or None
        self.component_match = component_match
        # Recompute author headroom
        if self.t_bound_dsl_us is not None:
            self.author_headroom_us = t_measured_us - self.t_bound_dsl_us

    def merge_calibration(
        self,
        db: "CalibrationDB",
        source: str,
    ) -> None:
        """Attach the measured calibration provenance used by the model."""
        from ..calibration.calib_loader import validate_calibration

        measured = [
            constant
            for constant in db.constants.values()
            if constant.source == "cce_microbench"
        ]
        derived = [
            constant
            for constant in db.constants.values()
            if constant.source.startswith("derived")
        ]
        self.calibration_source = source
        self.calibration_version = db.version
        self.calibration_hardware_name = db.hardware_name
        self.calibration_measured_constant_count = len(measured)
        self.calibration_derived_constant_count = len(derived)
        self.calibration_max_measured_ci_rel = (
            max(constant.ci_rel for constant in measured) if measured else None
        )
        self.calibration_p0_violations = db.validate_p0_constants()
        self.calibration_p0_complete = not self.calibration_p0_violations
        all_warnings = validate_calibration(db)
        self.calibration_warnings = [
            warning
            for warning in all_warnings
            if warning not in self.calibration_p0_violations
        ]
        missing_startups = [
            component
            for component in ("vector", "cube")
            if component not in db.startup_latency
        ]
        self.calibration_fallbacks = []
        if missing_startups:
            self.calibration_fallbacks.append(
                "Gap 4 startup latency uses hard-coded diagnostic defaults for "
                + ", ".join(missing_startups)
                + "; attribution is not fully calibration-backed"
            )

        # Check for missing P0 bandwidth constants that affect model accuracy
        l0c_gm = db.constants.get("BW_l0c_to_gm_sustained")
        if l0c_gm is None or l0c_gm.value <= 0:
            self.calibration_fallbacks.append(
                "BW_l0c_to_gm_sustained not measured — "
                "MTE_UB component uses UB→GM (MTE3) rate as fallback; "
                "Cube-output (FixPipe) transfers may be mis-estimated"
            )

        hbm_allcore = db.constants.get("BW_hbm_allcore_sustained")
        if hbm_allcore is None or hbm_allcore.value <= 0:
            self.calibration_fallbacks.append(
                "BW_hbm_allcore_sustained not measured — "
                "grid floor uses single-core GM→UB rate; "
                "memory-bound grid floor may be optimistic under full-core contention"
            )

    def merge_profile(self, profile_report) -> None:
        """Merge OperatorBottleneckReport; overrides recommended_action when author headroom dominates."""
        from ..extract.op_classifier import Component

        self.profile_diagnosis = profile_report.diagnosis
        self.profile_dominant_component = (
            profile_report.dominant_component.value
            if profile_report.dominant_component else None
        )
        self.exposed_control_frac_measured = profile_report.exposed_control_frac_measured
        self.exposed_control_frac_model = profile_report.exposed_control_frac_model
        self.exposed_control_deficit_pts = profile_report.exposed_control_deficit_pts
        self.exposed_control_deficit_us = profile_report.exposed_control_deficit_us
        self.n_sync_ops = profile_report.n_sync_ops

        if (
            self.author_headroom_us is not None
            and self.author_headroom_us > 0
            and self.exposed_control_deficit_us is not None
            and self.exposed_control_deficit_us > 0
        ):
            upper_us = min(
                self.author_headroom_us,
                self.exposed_control_deficit_us,
            )
            self.headroom_status = "diagnostic_upper_bound"
            self.recoverable_headroom_lower_us = 0.0
            self.recoverable_headroom_upper_us = upper_us
            self.recoverable_headroom_estimate_us = None
            self.headroom_confidence = "low"
            self.headroom_method = (
                "Range upper bound from msprof same-core scalar residency minus "
                "DES exposed-control overlap, capped by the measured-minus-DSL "
                "residual. No point estimate is claimed until a "
                "correctness-verified counterfactual is measured."
            )
            optimized_floor_us = self.t_measured_us - upper_us
            if self.t_measured_us and optimized_floor_us > 0:
                self.potential_speedup_upper = (
                    self.t_measured_us / optimized_floor_us
                )

        # Only override when author headroom is the dominant gap (>15% of T_measured).
        # Below the threshold the model's five-gap attribution is still the better signal.
        if not (
            self.author_headroom_us is not None
            and self.t_measured_us is not None
            and self.t_measured_us > 0
            and self.author_headroom_us / self.t_measured_us > 0.15
        ):
            return

        diag = profile_report.diagnosis
        comp = profile_report.dominant_component

        if diag == "Insufficient Parallelism" and comp == Component.SCALAR:
            n_sync = profile_report.n_sync_ops or 0
            pts = profile_report.exposed_control_deficit_pts
            pts_str = f", +{pts * 100:.0f} pts exposed" if pts is not None else ""
            self.recommended_action = (
                f"Profile-guided hypothesis: investigate sync barriers "
                f"({n_sync} ops{pts_str}); validate any PIPE_S/pipe_barrier "
                f"change with a correctness-checked hardware counterfactual"
            )
        elif diag == "Insufficient Parallelism":
            self.recommended_action = (
                "Increase parallelism — all hardware units underutilized"
            )
        elif diag in ("Compute Bound", "MTE Bound"):
            comp_str = comp.value if comp else "unknown"
            self.recommended_action = (
                f"Kernel is {diag} on {comp_str} — "
                f"increase arithmetic intensity or reduce transfers"
            )
        elif diag in ("Inefficient Compute", "Inefficient MTE"):
            comp_str = comp.value if comp else "unknown"
            self.recommended_action = f"{diag} on {comp_str} — reduce per-element overhead"
