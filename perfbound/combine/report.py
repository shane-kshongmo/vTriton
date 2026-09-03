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

from .bound_combiner import BoundResult
from .two_limit import TwoLimitResult

if TYPE_CHECKING:
    from ..calibration.constants import CalibrationDB


REPORT_SCHEMA_VERSION = "perfbound_report_v2"

_RECOMMENDATIONS = {
    "grid": "Fix grid partitioning — increase occupancy or load balance",
    "gap1_wrong_unit": "Fix DSL types — move ops to eligible unit",
    "gap2_coalescing": "Merge transfers — increase transfer size to reduce amortization",
    "gap3_avoidable_serial": "Add ping-pong buffer to overlap this handoff",
    "gap4_intra_unit_exec": "Increase SIMD repeat/mask utilization",
}

_OPPORTUNITY_SCOPES = {
    "grid": "launch_and_partitioning",
    "gap1_wrong_unit": "compiler_or_dsl_placement",
    "gap2_coalescing": "kernel_or_lowering_transfers",
    "gap3_avoidable_serial": "compiler_or_kernel_overlap",
    "gap4_intra_unit_exec": "kernel_intra_unit_utilization",
}

_MODELING_PURPOSE = (
    "use semantic IR, HIVM IR, and hardware profiling values to rank "
    "optimization opportunities and calculate sound theoretical ceilings"
)
_COUNTERFACTUAL_REQUIREMENT = (
    "A correctness-verified counterfactual measurement is required before "
    "claiming achievable headroom."
)

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
    t_body_bound_us: float = 0.0
    t_launch_overhead_us: float = 0.0

    # Two-limit (A.7)
    t_bound_hivm_us: Optional[float] = None
    t_bound_dsl_us: Optional[float] = None
    t_measured_us: Optional[float] = None
    compiler_headroom_us: Optional[float] = None
    author_headroom_us: Optional[float] = None

    # Measurement metadata
    msprof_source: Optional[str] = None
    n_invocations: Optional[int] = None
    task_wait_us: Optional[float] = None
    component_match: Optional[bool] = None
    measurement_metric: Optional[str] = None
    event_elapsed_us: Optional[float] = None
    model_coverage: Optional[dict] = None

    # Calibration provenance
    calibration_source: Optional[str] = None
    calibration_version: Optional[str] = None
    calibration_hardware_name: Optional[str] = None
    calibration_measured_constant_count: int = 0
    calibration_max_measured_ci_rel: Optional[float] = None
    calibration_p0_complete: Optional[bool] = None
    calibration_p0_violations: list[str] = field(default_factory=list)
    calibration_warnings: list[str] = field(default_factory=list)
    calibration_fallbacks: list[str] = field(default_factory=list)

    # Loop-resolution diagnostics (visibility into how loose t_bound_us may
    # be due to unresolved, data-dependent scf.for trip counts). None when
    # the DES JSON predates this feature or the kernel has no loops.
    # t_bound_worst_case_us is diagnostic-only — NOT a sound lower bound —
    # and must never be conflated with the primary t_bound_us.
    loop_resolution: Optional[dict] = None

    # DES critical-path event-wait attribution (from des_event_wait_analyzer).
    # This is Gap-3 (avoidable serialization) attribution *inside* the elapsed
    # DES critical path — it is NEVER added to t_bound (spec §2.2/§3: event
    # waits explain where elapsed time went, they are not an extra bound term).
    # None when the DES graph was static-scheduled (empty critical path) or
    # predates the critical-path feature; check des_event_wait["populated"].
    des_event_wait: Optional[dict] = None

    # Profile diagnosis (from profile_utilization)
    profile_diagnosis: Optional[str] = None
    profile_dominant_component: Optional[str] = None
    exposed_control_frac_measured: Optional[float] = None
    exposed_control_frac_model: Optional[float] = None
    exposed_control_deficit_pts: Optional[float] = None
    exposed_control_deficit_us: Optional[float] = None
    n_sync_ops: Optional[int] = None

    # Attribution (five-way, fractions of T_bound)
    attribution: dict[str, float] = field(default_factory=dict)
    attribution_us: dict[str, float] = field(default_factory=dict)

    # Recommendation
    recommended_action: str = "unknown"

    def _build_modeling_output(self) -> dict:
        """Build the canonical interpretation of the model's output.

        Attribution ranks where to investigate. Measured-to-bound differences
        are sound upper ceilings when coverage, calibration, and bound ordering
        are valid. Neither is an achievable point estimate by itself.
        """
        coverage = self.model_coverage if isinstance(self.model_coverage, dict) else {}
        coverage_status = coverage.get("status", "unknown")
        trace_timing_status = coverage.get("trace_timing_status", "unknown")
        semantic_overlay = coverage.get("semantic_overlay", {})
        semantic_applied = (
            isinstance(semantic_overlay, dict)
            and bool(semantic_overlay.get("applied"))
        )
        basis = ["hivm_ir"]
        if semantic_applied:
            basis.insert(0, "semantic_ir")
        if self.t_measured_us is not None or self.profile_diagnosis is not None:
            basis.append("hardware_profile_values")
        basis.append("perf_bound_theory")

        bounds_available = (
            self.t_bound_hivm_us is not None
            and self.t_bound_dsl_us is not None
            and self.t_bound_hivm_us > 0
            and self.t_bound_dsl_us > 0
        )
        bound_order_valid: Optional[bool] = None
        if bounds_available:
            bound_order_valid = self.t_bound_hivm_us <= self.t_bound_dsl_us
            if self.t_measured_us is not None:
                bound_order_valid = (
                    bound_order_valid
                    and self.t_bound_dsl_us <= self.t_measured_us
                )

        calibration_valid = (
            self.calibration_p0_complete is True
            and not self.calibration_p0_violations
        )
        analytical_ready = (
            coverage_status == "complete"
            and bounds_available
            and calibration_valid
            and bound_order_valid is not False
        )

        if coverage_status != "complete":
            status = (
                "coverage_unknown"
                if coverage_status == "unknown"
                else "model_incomplete"
            )
        elif not bounds_available:
            status = "bounds_unavailable"
        elif self.calibration_p0_complete is None:
            status = "calibration_unknown"
        elif not calibration_valid:
            status = "calibration_incomplete"
        elif bound_order_valid is False:
            status = "bound_violation"
        elif self.t_measured_us is None:
            status = "measurement_required"
        elif self.measurement_metric != "msprof_task_duration":
            status = "task_measurement_required"
        else:
            status = "sound_theoretical_ceiling"

        opportunities = []
        if status == "sound_theoretical_ceiling":
            ranked = sorted(
                (
                    (name, max(float(gap_us), 0.0))
                    for name, gap_us in self.attribution_us.items()
                    if float(gap_us) > _AT_BOUND_EPS
                ),
                key=lambda item: (-item[1], item[0]),
            )
            for rank, (name, gap_us) in enumerate(ranked, start=1):
                opportunities.append({
                    "rank": rank,
                    "name": name,
                    "modeled_gap_us": gap_us,
                    "fraction_of_bound": self.attribution.get(name, 0.0),
                    "scope": _OPPORTUNITY_SCOPES.get(name, "unknown"),
                    "action": _RECOMMENDATIONS.get(
                        name, "Profile to identify bottleneck"
                    ),
                })

        compiler_floor_shift_us = None
        if analytical_ready:
            compiler_floor_shift_us = max(
                self.t_bound_dsl_us - self.t_bound_hivm_us, 0.0
            )

        ceilings = {
            "to_realized_dsl_floor_us": None,
            "to_idealized_hivm_floor_us": None,
            "speedup_to_realized_dsl_floor_upper": None,
            "speedup_to_idealized_hivm_floor_upper": None,
        }
        if status == "sound_theoretical_ceiling":
            ceilings = {
                "to_realized_dsl_floor_us": max(
                    self.t_measured_us - self.t_bound_dsl_us, 0.0
                ),
                "to_idealized_hivm_floor_us": max(
                    self.t_measured_us - self.t_bound_hivm_us, 0.0
                ),
                "speedup_to_realized_dsl_floor_upper": (
                    self.t_measured_us / self.t_bound_dsl_us
                ),
                "speedup_to_idealized_hivm_floor_upper": (
                    self.t_measured_us / self.t_bound_hivm_us
                ),
            }

        return {
            "schema_version": "modeling_output_v1",
            "purpose": _MODELING_PURPOSE,
            "basis": basis,
            "bounds": {
                "hivm_floor_us": self.t_bound_hivm_us,
                "dsl_floor_us": self.t_bound_dsl_us,
            },
            "screening_metrics": {
                "event_elapsed_us": self.event_elapsed_us,
            },
            "profile_inputs": {
                "task_duration_us": self.t_measured_us,
                "measurement_metric": self.measurement_metric,
                "msprof_source": self.msprof_source,
                "invocations": self.n_invocations,
                "task_wait_us": self.task_wait_us,
                "component_match": self.component_match,
                "diagnosis": self.profile_diagnosis,
                "dominant_component": self.profile_dominant_component,
                "exposed_control_fraction_measured": (
                    self.exposed_control_frac_measured
                ),
                "exposed_control_fraction_model": self.exposed_control_frac_model,
                "exposed_control_deficit_points": self.exposed_control_deficit_pts,
                "exposed_control_deficit_us": self.exposed_control_deficit_us,
                "sync_operations": self.n_sync_ops,
            },
            "simulator_trace": {
                "role": "calibration_and_validation_only",
                "used_as_kernel_model_input": False,
            },
            "status": status,
            "compiler_floor_shift_us": compiler_floor_shift_us,
            "theoretical_ceilings": ceilings,
            "opportunity_ranking": opportunities,
            "opportunity_ranking_semantics": (
                "Diagnostic priority only; modeled gaps are not additive or "
                "independently attainable savings."
            ),
            "achievable_headroom": {
                "status": "not_established",
                "point_estimate_us": None,
                "evidence_required": _COUNTERFACTUAL_REQUIREMENT,
            },
            "validity_gates": {
                "model_coverage_status": coverage_status,
                "modeled_trace_timing_status": trace_timing_status,
                "calibration_p0_complete": self.calibration_p0_complete,
                "measurement_metric": self.measurement_metric,
                "bound_order_valid": bound_order_valid,
            },
        }

    def to_dict(self) -> dict:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "kernel_name": self.kernel_name,
            "t_bound_us": self.t_bound_us,
            "binding_tier": self.binding_tier,
            "binding_component": self.binding_component,
            "t_grid_floor_us": self.t_grid_floor_us,
            "t_core_floor_us": self.t_core_floor_us,
            "t_serial_irreducible_us": self.t_serial_irreducible_us,
            "t_body_bound_us": self.t_body_bound_us,
            "t_launch_overhead_us": self.t_launch_overhead_us,
            "model_coverage": self.model_coverage,
            "attribution": self.attribution,
            "attribution_us": self.attribution_us,
            "calibration": {
                "source": self.calibration_source,
                "version": self.calibration_version,
                "hardware_name": self.calibration_hardware_name,
                "measured_constant_count": self.calibration_measured_constant_count,
                "max_measured_ci_rel": self.calibration_max_measured_ci_rel,
                "p0_complete": self.calibration_p0_complete,
                "p0_violations": self.calibration_p0_violations,
                "warnings": self.calibration_warnings,
                "fallbacks": self.calibration_fallbacks,
            },
            "modeling_output": self._build_modeling_output(),
            "loop_resolution": self.loop_resolution,
            "des_event_wait": self.des_event_wait,
            "recommended_action": self.recommended_action,
        }

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
            "",
            f"T_bound:   {self.t_bound_us:.2f} us",
            f"  Tier 1 (grid):      {self.t_grid_floor_us:.2f} us",
            f"  Tier 2 (component): {self.t_core_floor_us:.2f} us",
            f"  Serial irreducible: {self.t_serial_irreducible_us:.2f} us",
            f"  Launch overhead:     {self.t_launch_overhead_us:.2f} us",
            "",
            f"Binding: {self.binding_tier}",
        ]
        if self.binding_component:
            lines.append(f"  Component: {self.binding_component}")

        if self.model_coverage is not None:
            lines.extend([
                "",
                "Model coverage:",
                f"  status: {self.model_coverage.get('status', 'unknown')}",
                f"  outlined calls: {self.model_coverage.get('outlined_calls', 0)} "
                f"(summarized: {self.model_coverage.get('summarized_outlined_calls', 0)})",
                f"  zero-byte transfers: {self.model_coverage.get('zero_byte_transfers', 0)}",
            ])

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

        if self.loop_resolution and self.loop_resolution.get("unresolved", 0) > 0:
            lines.extend([
                "",
                "Loop resolution:",
                f"  {self.loop_resolution['unresolved']}/{self.loop_resolution['total']} "
                "loop(s) have data-dependent trip counts (loop_multiplier=1, "
                "the sound minimum, applied to t_bound_us)",
            ])
            worst = self.loop_resolution.get("t_bound_worst_case_us")
            if worst is not None:
                lines.append(
                    f"  diagnostic worst-case T_bound (NOT sound, NOT the "
                    f"primary bound): {worst:.2f} us"
                )

        if self.des_event_wait is not None:
            dew = self.des_event_wait
            lines.append("")
            lines.append(
                "DES critical-path serialization (Gap-3 attribution, "
                "NOT added to T_bound):"
            )
            if not dew.get("populated"):
                lines.append(
                    "  critical path not populated — regenerate the DES graph "
                    "with --scheduler des for event-wait attribution"
                )
            else:
                lines.append(
                    f"  critical path: {dew['critical_path_cycles']} cyc "
                    f"({dew['critical_path_us']:.2f} us); "
                    f"issue {dew['critical_path_issue_cycles']} cyc, "
                    f"event-wait {dew['critical_path_event_wait_cycles']} cyc "
                    f"({dew['critical_path_event_wait_us']:.2f} us)"
                )
                for grp in dew.get("top_wait_groups", [])[:5]:
                    lines.append(
                        f"    wait {grp['wait_us']:.2f} us  "
                        f"({grp['wait_cycles']} cyc, {grp['ops']} ops)  {grp['key']}"
                    )

        lines.append("")
        lines.append("Attribution (absolute and fraction of T_bound):")
        for gap_name, frac in sorted(self.attribution.items(), key=lambda x: -x[1]):
            gap_us = self.attribution_us.get(gap_name, 0.0)
            lines.append(f"  {gap_name}: {gap_us:.2f} us ({frac:.3f})")

        if self.event_elapsed_us is not None:
            lines.append(
                f"  End-to-end event elapsed: {self.event_elapsed_us:.2f} us "
                "(not used for task-duration headroom)"
            )

        modeling = self._build_modeling_output()
        ceilings = modeling["theoretical_ceilings"]
        lines.extend([
            "",
            "Modeling Output:",
            f"  basis: {' + '.join(modeling['basis'])}",
            f"  purpose: {modeling['purpose']}",
            "  simulator trace: calibration/validation only; not a kernel model input",
            f"  status: {modeling['status']}",
        ])
        bounds = modeling["bounds"]
        if bounds["hivm_floor_us"] is not None:
            lines.append(
                f"  idealized HIVM floor: {bounds['hivm_floor_us']:.2f} us"
            )
        if bounds["dsl_floor_us"] is not None:
            lines.append(
                f"  realized DSL floor: {bounds['dsl_floor_us']:.2f} us"
            )
        profile_inputs = modeling["profile_inputs"]
        if profile_inputs["task_duration_us"] is not None:
            lines.append(
                "  hardware profile task duration: "
                f"{profile_inputs['task_duration_us']:.2f} us"
            )
            if profile_inputs["msprof_source"]:
                lines.append(
                    f"  hardware profile source: {profile_inputs['msprof_source']}"
                )
            if profile_inputs["invocations"] is not None:
                lines.append(
                    "  hardware profile invocations: "
                    f"{profile_inputs['invocations']}"
                )
            if profile_inputs["task_wait_us"] is not None:
                lines.append(
                    f"  median task wait: {profile_inputs['task_wait_us']:.2f} us "
                    "(reported separately)"
                )
            if profile_inputs["component_match"] is not None:
                match = "yes" if profile_inputs["component_match"] else "no"
                lines.append(f"  coarse task-category match: {match}")
        if profile_inputs["diagnosis"]:
            lines.append(f"  profile diagnosis: {profile_inputs['diagnosis']}")
        if profile_inputs["dominant_component"]:
            lines.append(
                "  profile dominant component: "
                f"{profile_inputs['dominant_component']}"
            )
        if profile_inputs["exposed_control_fraction_measured"] is not None:
            lines.append(
                "  measured scalar fraction: "
                f"{profile_inputs['exposed_control_fraction_measured']:.1%}"
            )
        if profile_inputs["exposed_control_deficit_points"] is not None:
            lines.append(
                "  exposed control deficit: "
                f"+{profile_inputs['exposed_control_deficit_points'] * 100:.1f} "
                "points"
            )
        if profile_inputs["exposed_control_deficit_us"] is not None:
            lines.append(
                "  exposed control deficit time: "
                f"~{profile_inputs['exposed_control_deficit_us']:.0f} us"
            )
        if profile_inputs["sync_operations"] is not None:
            lines.append(f"  sync operations: {profile_inputs['sync_operations']}")
        if modeling["compiler_floor_shift_us"] is not None:
            lines.append(
                "  modeled compiler floor shift: "
                f"{modeling['compiler_floor_shift_us']:.2f} us"
            )
        if ceilings["to_realized_dsl_floor_us"] is not None:
            lines.extend([
                "  sound ceiling to realized DSL floor: "
                f"{ceilings['to_realized_dsl_floor_us']:.2f} us "
                f"({ceilings['speedup_to_realized_dsl_floor_upper']:.2f}x)",
                "  sound ceiling to idealized HIVM floor: "
                f"{ceilings['to_idealized_hivm_floor_us']:.2f} us "
                f"({ceilings['speedup_to_idealized_hivm_floor_upper']:.2f}x)",
            ])
        else:
            lines.append("  theoretical ceilings: suppressed by validity gates")
        lines.append("  achievable point estimate: not established")
        if modeling["opportunity_ranking"]:
            lines.append("  ranked opportunities:")
            for opportunity in modeling["opportunity_ranking"]:
                lines.append(
                    f"    {opportunity['rank']}. {opportunity['name']}: "
                    f"{opportunity['modeled_gap_us']:.2f} us "
                    f"({opportunity['scope']})"
                )
        else:
            if modeling["status"] == "sound_theoretical_ceiling":
                lines.append("  ranked opportunities: none above threshold")
            else:
                lines.append("  ranked opportunities: unavailable")
        lines.append(
            "  ranking semantics: diagnostic priority, not additive savings"
        )

        lines.append("")
        lines.append(f"Recommended action: {self.recommended_action}")

        return "\n".join(lines)

    @classmethod
    def from_bound(cls, result: BoundResult,
                   two_limit: Optional[TwoLimitResult] = None) -> "KernelReport":
        """Create a report from a BoundResult."""
        dominant_name, _ = result.attribution.dominant_gap()

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
            t_body_bound_us=(
                result.t_body_bound_us
                if result.t_body_bound_us is not None
                else result.t_bound_us
            ),
            t_launch_overhead_us=result.t_launch_overhead_us,
            t_bound_hivm_us=two_limit.t_bound_hivm_us if two_limit else None,
            t_bound_dsl_us=two_limit.t_bound_dsl_us if two_limit else result.t_bound_us,
            t_measured_us=two_limit.t_measured_us if two_limit else None,
            measurement_metric=(
                "msprof_task_duration"
                if two_limit and two_limit.t_measured_us is not None
                else None
            ),
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
        task_wait_us: float | None = None,
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
        self.measurement_metric = "msprof_task_duration"
        self.msprof_source = msprof_source or None
        self.n_invocations = n_invocations or None
        self.component_match = component_match
        self.task_wait_us = task_wait_us
        # Recompute author headroom
        coverage_status = (
            self.model_coverage.get("status") if self.model_coverage else None
        )
        if (
            self.t_bound_dsl_us is not None
            and coverage_status in {None, "complete"}
        ):
            self.author_headroom_us = t_measured_us - self.t_bound_dsl_us

    def merge_event_elapsed(self, elapsed_us: float) -> None:
        """Attach end-to-end event latency without treating it as task time."""
        self.event_elapsed_us = elapsed_us
        self.measurement_metric = "event_elapsed"

    def merge_model_coverage(self, coverage: dict) -> None:
        """Attach extraction coverage and suppress claims on partial models."""
        self.model_coverage = dict(coverage)
        status = self.model_coverage.get("status", "legacy_unknown")
        if status == "complete":
            return
        self.compiler_headroom_us = None
        self.author_headroom_us = None
        self.recommended_action = (
            "Model coverage incomplete — restore semantic work before optimization advice"
        )

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
        self.calibration_source = source
        self.calibration_version = db.version
        self.calibration_hardware_name = db.hardware_name
        self.calibration_measured_constant_count = len(measured)
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
        """Merge profile evidence and update the recommendation when warranted."""
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
            self.model_coverage
            and self.model_coverage.get("status") != "complete"
        ):
            return

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
                f"Profile-guided hypothesis: reduce scalar/control exposure "
                f"from loop/tile structure and sync barriers ({n_sync} sync ops"
                f"{pts_str}); try safe tile-size or exact-shape fast-path "
                f"variants before scalar-throughput tuning, and validate each "
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
