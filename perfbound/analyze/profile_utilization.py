"""根据 profiling 汇总信息计算实际 component 利用率。

本文件用于把粗粒度 profiling 统计量和
``perfbound.model.component_model`` 算出的理论 component floor 做对比。

当前先假设输入已经是很简单的统计数据：每个 component 一行汇总。
后续接入真实 msprof/CSV 时，可以另外加 parser，不需要改这里的计算公式。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from typing import Optional

from ..calibration.calib_loader import load_calibration
from ..calibration.constants import CalibrationDB, DType
from ..extract.hivm_extractor import extract_hivm
from ..extract.op_classifier import Component, Precision
from ..model.component_model import ComponentBound, compute_component_floor_from_db


_EPSILON = 1e-12
_COMPUTE_COMPONENTS = (Component.CUBE, Component.VECTOR, Component.SCALAR)
_MTE_COMPONENTS = (Component.MTE_GM, Component.MTE_L1, Component.MTE_UB)


@dataclass
class WorkBreakdownItem:
    """component 内部的 operator-aware work 项。

    对 Compute 来说，``label`` 通常是 precision，例如 fp16/bf16/int8。
    对 MTE 来说，``label`` 通常是 transfer path，例如 gm->ub/l1->l0a。

    ``peak_rate`` 的单位必须和 work 对齐：
    Compute 使用 ops/us，MTE 使用 bytes/us。
    """

    label: str
    work: float
    peak_rate: float


@dataclass
class ProfileComponentStats:
    """单个硬件 component 的粗粒度 profiling 统计。

    ``work_done`` 必须和理论 bound 使用同一种单位：
    Cube/Vector/Scalar 使用 ops 或 FLOPs，MTE component 使用 bytes。
    """

    component: Component
    work_done: float
    active_time_us: float
    work_breakdown: list[WorkBreakdownItem] = field(default_factory=list)


@dataclass
class KernelProfileStats:
    """一个 kernel 或一个 core 的最小 profiling 汇总输入。"""

    kernel_name: str
    elapsed_time_us: float
    components: list[ProfileComponentStats] = field(default_factory=list)


@dataclass
class ComponentUtilization:
    """单个 component 的 A/I/U/R/E 分析结果。"""

    component: Component
    work_done: float
    bound_work: float
    elapsed_time_us: float
    active_time_us: float
    actual_performance: float
    ideal_performance: float
    e_efficiency: float
    r_residency: float
    u_utilization: float
    dominant_item: Optional[str]
    dominant_share: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class UtilizationReport:
    """一个 kernel 的实际利用率与理论 component bound 对比结果。"""

    kernel_name: str
    elapsed_time_us: float
    t_core_floor_us: float
    binding_component: Component
    component_results: dict[str, ComponentUtilization] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class OperatorBottleneckReport:
    """论文第四章后半部分的算子级瓶颈分析结果。"""

    kernel_name: str
    elapsed_time_us: float
    diagnosis: str
    bound_kind: Optional[str]
    dominant_component: Optional[Component]
    dominant_item: Optional[str]
    dominant_share: float
    component_results: dict[str, ComponentUtilization] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def compute_realized_utilization(
    profile: KernelProfileStats,
    component_bound: ComponentBound,
    work_tolerance: float = 0.10,
) -> UtilizationReport:
    """根据 profiling 汇总信息计算 A、I、U、R、E。

    定义：
      A_c = work_done_c / elapsed_time
      I_c = operator-aware ideal performance
      U_c = A_c / I_c
      R_c = active_time_c / elapsed_time
      E_c = U_c / R_c = work_done_c / (active_time_c * I_c)

    参数：
        profile: 实测 kernel 的粗粒度 profiling 统计。
        component_bound: compute_component_floor 算出来的理论 component floor。
        work_tolerance: profiling 里的 work_done 和理论 bound 里的 work
            允许有多大的相对误差；超过后会产生 warning。
    """

    elapsed = profile.elapsed_time_us
    report = UtilizationReport(
        kernel_name=profile.kernel_name,
        elapsed_time_us=elapsed,
        t_core_floor_us=component_bound.t_core_floor_us,
        binding_component=component_bound.binding_component,
    )

    if elapsed <= 0:
        report.warnings.append("elapsed_time_us 必须是正数")
        return report

    for stats in profile.components:
        comp_key = stats.component.value
        bound_work = _bound_work_for_component(component_bound, stats.component)
        ideal_performance, ideal_source = _ideal_performance_for_component(
            stats,
            component_bound,
        )

        warnings: list[str] = []
        if bound_work <= 0:
            warnings.append("profiling 中有该 component，但理论 bound 中没有对应 work")
        if ideal_performance <= 0:
            warnings.append("该 component 没有正数的 ideal performance")
        if ideal_source == "component_bound":
            warnings.append("未提供 operator-aware work_breakdown，已回退到 component_bound 的 I_c")

        actual_performance = stats.work_done / elapsed
        utilization = (
            actual_performance / ideal_performance
            if ideal_performance > _EPSILON
            else 0.0
        )
        r_residency = stats.active_time_us / elapsed
        e_efficiency = (
            utilization / r_residency
            if r_residency > _EPSILON
            else 0.0
        )

        if stats.active_time_us < 0:
            warnings.append("active_time_us 是负数")
        if stats.active_time_us > elapsed * (1.0 + _EPSILON):
            warnings.append("active_time_us 大于 elapsed_time_us")
        if e_efficiency > 1.05:
            warnings.append("E > 1.05；请检查单位、profiling 统计或校准值")
        if utilization > 1.05:
            warnings.append("U > 1.05；请检查 A/I 的单位或 ideal performance 来源")
        if r_residency > 1.05:
            warnings.append("R > 1.05；active_time 和 elapsed_time 的统计口径可能不同")
        if bound_work > 0:
            rel_mismatch = abs(stats.work_done - bound_work) / bound_work
            if rel_mismatch > work_tolerance:
                warnings.append(
                    f"profiling work 和理论 bound work 相差 {rel_mismatch:.1%}"
                )

        dominant_item, dominant_share = _dominant_breakdown_item(stats)
        result = ComponentUtilization(
            component=stats.component,
            work_done=stats.work_done,
            bound_work=bound_work,
            elapsed_time_us=elapsed,
            active_time_us=stats.active_time_us,
            actual_performance=actual_performance,
            ideal_performance=ideal_performance,
            e_efficiency=e_efficiency,
            r_residency=r_residency,
            u_utilization=utilization,
            dominant_item=dominant_item,
            dominant_share=dominant_share,
            warnings=warnings,
        )
        report.component_results[comp_key] = result
        report.warnings.extend(f"{comp_key}: {warning}" for warning in warnings)

    return report


def analyze_operator_bottleneck(
    profile: KernelProfileStats,
    component_bound: ComponentBound,
    u_threshold: float = 0.80,
    r_threshold: float = 0.50,
    work_tolerance: float = 0.10,
) -> OperatorBottleneckReport:
    """按论文第四章后半部分流程分析算子瓶颈。

    流程：
      1. 对每个 component 计算 A、I、U。
      2. 若存在 U >= u_threshold 的 component，则认为达到 roofline ceiling。
         Compute component -> Compute Bound；MTE component -> MTE Bound。
      3. 若所有 component 都未达到 ceiling，则进入 Underutilization Analysis。
      4. 计算 R、E。如果所有有效 component 的 R 都低于 r_threshold，
         判定为 Insufficient Parallelism。
      5. 否则选出高 R 且低 E 的主导 component：
         Compute component -> Inefficient Compute；MTE component -> Inefficient MTE。

    ``work_breakdown`` 用于定位主导 precision 或 transfer path；如果没有提供，
    主导项会是 None，但 component 级诊断仍然可用。
    """

    utilization = compute_realized_utilization(
        profile,
        component_bound,
        work_tolerance=work_tolerance,
    )
    warnings = list(utilization.warnings)
    valid = [
        result
        for result in utilization.component_results.values()
        if result.work_done > 0 and result.ideal_performance > 0
    ]

    if not valid:
        return OperatorBottleneckReport(
            kernel_name=profile.kernel_name,
            elapsed_time_us=profile.elapsed_time_us,
            diagnosis="Insufficient Data",
            bound_kind=None,
            dominant_component=None,
            dominant_item=None,
            dominant_share=0.0,
            component_results=utilization.component_results,
            warnings=warnings + ["没有可用于分析的有效 component"],
        )

    ceiling_candidates = [
        result for result in valid if result.u_utilization >= u_threshold
    ]
    if ceiling_candidates:
        dominant = max(ceiling_candidates, key=lambda item: item.u_utilization)
        bound_kind = (
            "Compute Bound"
            if dominant.component in _COMPUTE_COMPONENTS
            else "MTE Bound"
        )
        return OperatorBottleneckReport(
            kernel_name=profile.kernel_name,
            elapsed_time_us=profile.elapsed_time_us,
            diagnosis=bound_kind,
            bound_kind=bound_kind,
            dominant_component=dominant.component,
            dominant_item=dominant.dominant_item,
            dominant_share=dominant.dominant_share,
            component_results=utilization.component_results,
            warnings=warnings,
        )

    if all(result.r_residency < r_threshold for result in valid):
        return OperatorBottleneckReport(
            kernel_name=profile.kernel_name,
            elapsed_time_us=profile.elapsed_time_us,
            diagnosis="Insufficient Parallelism",
            bound_kind=None,
            dominant_component=None,
            dominant_item=None,
            dominant_share=0.0,
            component_results=utilization.component_results,
            warnings=warnings,
        )

    # 高 R 但低 E 的 component 是 Inefficient Component 的主要嫌疑。
    high_r = [result for result in valid if result.r_residency >= r_threshold]
    candidates = high_r if high_r else valid
    dominant = max(
        candidates,
        key=lambda item: item.r_residency * max(0.0, 1.0 - item.e_efficiency),
    )

    if dominant.component in _MTE_COMPONENTS:
        diagnosis = "Inefficient MTE"
    else:
        diagnosis = "Inefficient Compute"

    return OperatorBottleneckReport(
        kernel_name=profile.kernel_name,
        elapsed_time_us=profile.elapsed_time_us,
        diagnosis=diagnosis,
        bound_kind=None,
        dominant_component=dominant.component,
        dominant_item=dominant.dominant_item,
        dominant_share=dominant.dominant_share,
        component_results=utilization.component_results,
        warnings=warnings,
    )


def _bound_work_for_component(
    component_bound: ComponentBound,
    component: Component,
) -> float:
    comp_key = component.value
    if component in (Component.MTE_GM, Component.MTE_L1, Component.MTE_UB):
        return component_bound.total_bytes.get(comp_key, 0.0)
    return component_bound.total_ops.get(comp_key, 0.0)


def _ideal_performance_for_component(
    stats: ProfileComponentStats,
    component_bound: ComponentBound,
) -> tuple[float, str]:
    """按 operator-aware 公式计算 I。

    Compute:
      I = Sum(Ops_prec) / Sum(Ops_prec / Peak_prec)

    MTE:
      I = Sum(Bytes_path) / Sum(Bytes_path / Bandwidth_path)

    如果 profiling/HIVM 暂时没有提供 breakdown 和 peak/bandwidth，
    就回退到 component_bound 中已有的 I_c。
    """

    numerator = 0.0
    denominator = 0.0
    for item in stats.work_breakdown:
        if item.work <= 0 or item.peak_rate <= 0:
            continue
        numerator += item.work
        denominator += item.work / item.peak_rate

    if numerator > 0 and denominator > _EPSILON:
        return numerator / denominator, "operator_aware"

    comp_key = stats.component.value
    bound_work = _bound_work_for_component(component_bound, stats.component)
    ideal_time = component_bound.per_component_us.get(comp_key, 0.0)
    if bound_work > 0 and ideal_time > _EPSILON:
        return bound_work / ideal_time, "component_bound"
    return 0.0, "missing"


def _dominant_breakdown_item(
    stats: ProfileComponentStats,
) -> tuple[Optional[str], float]:
    """返回主导 precision 或 transfer path 及其占比。"""

    total = sum(item.work for item in stats.work_breakdown if item.work > 0)
    if total <= 0:
        return None, 0.0
    dominant = max(stats.work_breakdown, key=lambda item: item.work)
    return dominant.label, dominant.work / total


def run_from_files(
    op_summary_path: str | Path,
    desgraph_path: str | Path,
    calibration_path: str | Path | None = None,
    *,
    kernel_name: str | None = None,
    u_threshold: float = 0.80,
    r_threshold: float = 0.50,
    work_tolerance: float = 0.10,
) -> OperatorBottleneckReport:
    """从 op_summary、DES graph 和 calibration 文件端到端运行分析。"""

    op_row = _read_op_summary_row(op_summary_path, kernel_name)
    profile_name = kernel_name or _cell(op_row, "Op Name", "op_name", "Name") or "unknown"
    elapsed_time_us = _float_cell(
        op_row,
        "Task Duration(us)",
        "duration(us)",
        "Duration(us)",
        "duration_us",
    )
    active_time_us = {
        Component.CUBE: _float_cell(op_row, "aic_mac_time(us)"),
        Component.VECTOR: _float_cell(op_row, "aiv_vec_time(us)"),
        Component.SCALAR: (
            _float_cell(op_row, "aic_scalar_time(us)")
            + _float_cell(op_row, "aiv_scalar_time(us)")
        ),
        Component.MTE_L1: _float_cell(op_row, "aic_mte1_time(us)"),
        Component.MTE_GM: (
            _float_cell(op_row, "aic_mte2_time(us)")
            + _float_cell(op_row, "aiv_mte2_time(us)")
        ),
        Component.MTE_UB: (
            _float_cell(op_row, "aic_fixpipe_time(us)")
            + _float_cell(op_row, "aiv_mte3_time(us)")
        ),
    }

    extract = extract_hivm(desgraph_path)
    db = load_calibration(calibration_path)
    component_bound = compute_component_floor_from_db(extract, db)

    work_done: dict[Component, float] = defaultdict(float)
    breakdown: dict[Component, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for op in extract.operations:
        comp = op.component
        if comp in _COMPUTE_COMPONENTS:
            work = float(op.elements) * float(op.loop_multiplier)
            label = op.precision.value if op.precision else "unknown"
        elif comp in _MTE_COMPONENTS:
            work = float(op.bytes_transferred) * float(op.loop_multiplier)
            src, dst = op.src_space.lower(), op.dst_space.lower()
            label = f"{src}->{dst}" if src and dst else comp.value
        else:
            continue
        if work <= 0:
            continue
        work_done[comp] += work
        breakdown[comp][label] += work

    components: list[ProfileComponentStats] = []
    for comp in Component:
        if work_done.get(comp, 0.0) <= 0 and active_time_us.get(comp, 0.0) <= 0:
            continue
        items = [
            WorkBreakdownItem(
                label=label,
                work=work,
                peak_rate=_peak_rate_for_label(comp, label, db),
            )
            for label, work in sorted(breakdown.get(comp, {}).items())
        ]
        components.append(
            ProfileComponentStats(
                component=comp,
                work_done=work_done.get(comp, 0.0),
                active_time_us=active_time_us.get(comp, 0.0),
                work_breakdown=items,
            )
        )

    profile = KernelProfileStats(
        kernel_name=profile_name,
        elapsed_time_us=elapsed_time_us,
        components=components,
    )
    return analyze_operator_bottleneck(
        profile,
        component_bound,
        u_threshold=u_threshold,
        r_threshold=r_threshold,
        work_tolerance=work_tolerance,
    )


def report_to_dict(report: OperatorBottleneckReport) -> dict:
    """把分析结果转成可 JSON 序列化的 dict。"""

    return {
        "kernel_name": report.kernel_name,
        "elapsed_time_us": report.elapsed_time_us,
        "diagnosis": report.diagnosis,
        "bound_kind": report.bound_kind,
        "dominant_component": (
            report.dominant_component.value if report.dominant_component else None
        ),
        "dominant_item": report.dominant_item,
        "dominant_share": report.dominant_share,
        "components": {
            key: {
                "component": result.component.value,
                "work_done": result.work_done,
                "bound_work": result.bound_work,
                "elapsed_time_us": result.elapsed_time_us,
                "active_time_us": result.active_time_us,
                "actual_performance": result.actual_performance,
                "ideal_performance": result.ideal_performance,
                "u_utilization": result.u_utilization,
                "r_residency": result.r_residency,
                "e_efficiency": result.e_efficiency,
                "dominant_item": result.dominant_item,
                "dominant_share": result.dominant_share,
                "warnings": result.warnings,
            }
            for key, result in report.component_results.items()
        },
        "warnings": report.warnings,
    }


def _read_op_summary_row(path: str | Path, kernel_name: str | None) -> dict[str, str]:
    with open(path) as f:
        reader = csv.DictReader(line for line in f if not line.strip().startswith("#"))
        rows = list(reader)
    if not rows:
        raise ValueError(f"op_summary 中没有可读取的行: {path}")
    if kernel_name is None:
        return rows[0]
    for row in rows:
        if _cell(row, "Op Name", "op_name", "Name") == kernel_name:
            return row
    raise ValueError(f"op_summary 中找不到 kernel: {kernel_name}")


def _cell(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value.strip()
    return ""


def _float_cell(row: dict[str, str], *names: str) -> float:
    value = _cell(row, *names)
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return float(value.split()[0])


def _peak_rate_for_label(
    component: Component,
    label: str,
    db: CalibrationDB,
) -> float:
    if component == Component.CUBE:
        return _cube_peak_rate(label, db)
    if component == Component.VECTOR:
        return _vector_peak_rate(label, db)
    if component in _MTE_COMPONENTS:
        src, sep, dst = label.partition("->")
        if not sep:
            return 0.0
        try:
            bw, _ = db.memory.lookup_bw(src, dst)
            return bw
        except KeyError:
            return 0.0
    return 0.0


def _cube_peak_rate(label: str, db: CalibrationDB) -> float:
    try:
        dtype = DType.from_str(label)
    except KeyError:
        return 0.0
    tflops = db.cube.throughput.get(dtype, 0.0)
    return tflops * 1e6 if tflops > 0 else 0.0


def _vector_peak_rate(label: str, db: CalibrationDB) -> float:
    try:
        precision = Precision(label)
    except ValueError:
        return 0.0
    if precision in (Precision.FP16, Precision.BF16):
        tflops = db.vector.throughput_fp16_tflops
    else:
        tflops = db.vector.throughput_fp32_tflops
    return tflops * 1e6 if tflops > 0 else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run profile utilization analysis from op_summary, DES graph, and calibration files."
    )
    parser.add_argument("--op-summary", required=True, help="Path to msprof op_summary CSV")
    parser.add_argument("--des-graph", required=True, help="Path to tritonsim-hivm DES graph JSON")
    parser.add_argument("--calibration", help="Path to calibration JSON; defaults to calib_910b3_v1.json")
    parser.add_argument("--kernel-name", help="Optional Op Name to select from op_summary")
    parser.add_argument("--u-threshold", type=float, default=0.80)
    parser.add_argument("--r-threshold", type=float, default=0.50)
    parser.add_argument("--work-tolerance", type=float, default=0.10)
    parser.add_argument(
        "--output-file",
        default="data/profile_utilization_inputs/profile_utilization_report.json",
        help="Path to write the JSON report",
    )
    args = parser.parse_args(argv)

    report = run_from_files(
        args.op_summary,
        args.des_graph,
        args.calibration,
        kernel_name=args.kernel_name,
        u_threshold=args.u_threshold,
        r_threshold=args.r_threshold,
        work_tolerance=args.work_tolerance,
    )

    output = json.dumps(report_to_dict(report), ensure_ascii=False, indent=2)
    Path(args.output_file).write_text(output + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
