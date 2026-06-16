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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from ..calibration.calib_loader import load_calibration
from ..calibration.constants import CalibrationDB, DType
from ..extract.hivm_extractor import extract_hivm
from ..extract.op_classifier import Component, Precision
from ..model.component_model import ComponentBound, compute_component_floor_from_db
from .hivm_bottleneck_diagnosis import (
    HIVMBottleneckReport,
    diagnose_hivm_bottleneck_from_des_ops,
    hivm_bottleneck_report_to_dict,
    read_des_graph_metadata,
)


_EPSILON = 1e-12
_COMPUTE_COMPONENTS = (Component.CUBE, Component.VECTOR, Component.SCALAR)
_MTE_COMPONENTS = (Component.MTE_GM, Component.MTE_L1, Component.MTE_UB)

# DES 调度时间线分类（用于量化暴露控制/同步赤字）。
_OVL_COMPUTE = {"PIPE_V", "PIPE_M"}
_OVL_MEMORY = {"PIPE_MTE2_V", "PIPE_MTE2_C", "PIPE_MTE3", "PIPE_MTE1", "PIPE_FIX"}
_OVL_CONTROL = {"PIPE_S", "PIPE_ALL"}
_SYNC_OPS = {"wait_flag", "set_flag", "pipe_barrier", "sync_block_wait", "sync_block_set"}


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

    # —— 对论文的补充：暴露控制/同步赤字的量化 ——
    # 仅当判定为 Insufficient Parallelism 且主导为暴露的 Scalar 控制时填充。
    # 模型(DES 调度)假设控制能和计算重叠到 exposed_control_frac_model；硬件实测的
    # 同核 scalar 占用 exposed_control_frac_measured 远高于它，差值即赤字（论文的
    # 二分类只给定性结论，这里给定量）。diagnostic-only，不改变任何 bound。
    exposed_control_frac_model: Optional[float] = None      # DES 关键路径暴露控制比例
    exposed_control_frac_measured: Optional[float] = None   # 同核实测 scalar 占比 (aiv)
    exposed_control_deficit_pts: Optional[float] = None     # measured - model（百分点/小数）
    exposed_control_deficit_us: Optional[float] = None      # 估计 µs，封顶到 author headroom
    n_sync_ops: Optional[int] = None
    hivm_bottleneck: Optional[HIVMBottleneckReport] = None


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

    # —— 对论文的补充：暴露的控制/同步开销 ——
    # 论文的欠利用分类只有两类，且用 (∀R<threshold) 判定 Insufficient
    # Parallelism。但这无法处理"某个高驻留 component 其实是暴露的控制/同步"的
    # 情形：在 Ascend 上 Scalar 还负责向其他单元派发指令并插入 pipe_barrier，
    # 这类 op 几乎没有算术/传输 work，却可能独占时间线。由于 work_done<=0，它
    # 会被 valid 过滤排除，于是论文会把算子误判为 Inefficient Component（让你去
    # 优化 scalar 的算术效率，方向错误）。我们补充：高驻留(R>=threshold)但无
    # 算术/传输 work 的 component 视为暴露的控制/同步，归因为 Insufficient
    # Parallelism（控制路径没能和计算重叠 = 并行不足），并把它标为主导 locus。
    exposed_control = sorted(
        (
            result
            for result in utilization.component_results.values()
            if result.r_residency >= r_threshold and result.work_done <= 0
        ),
        key=lambda item: item.r_residency,
        reverse=True,
    )
    top_valid_residency = max(
        (result.r_residency for result in valid), default=0.0
    )
    valid_high_r = [result for result in valid if result.r_residency >= r_threshold]

    if not valid_high_r or (
        exposed_control and exposed_control[0].r_residency >= top_valid_residency
    ):
        locus = exposed_control[0] if exposed_control else None
        ip_warnings = list(warnings)
        if locus is not None:
            ip_warnings.append(
                f"{locus.component.value}: 高驻留 R={locus.r_residency:.2f} 但无算术/传输 "
                "work，判定为暴露的控制/同步开销（exposed control/sync），归因为并行不足"
            )
        return OperatorBottleneckReport(
            kernel_name=profile.kernel_name,
            elapsed_time_us=profile.elapsed_time_us,
            diagnosis="Insufficient Parallelism",
            bound_kind=None,
            dominant_component=locus.component if locus else None,
            dominant_item=locus.dominant_item if locus else None,
            dominant_share=locus.dominant_share if locus else 0.0,
            component_results=utilization.component_results,
            warnings=ip_warnings,
        )

    # 高 R 但低 E 的 component 是 Inefficient Component 的主要嫌疑。
    high_r = valid_high_r
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


def _ovl_category(op) -> str:
    """把一个 DES op 归类为 control_sync / compute / memory / other。

    OpRecord 不带 is_sync/is_barrier，所以用 pipe + op_name 判定。
    """
    if op.op_name in _SYNC_OPS or op.pipe in _OVL_CONTROL:
        return "control_sync"
    if op.pipe in _OVL_COMPUTE:
        return "compute"
    if op.pipe in _OVL_MEMORY:
        return "memory"
    return "other"


def _exposed_control_overlap(operations) -> tuple[float, int, int]:
    """扫描 DES 调度，计算"模型预测的暴露控制比例"。

    对 start_cycle/end_cycle 做 +1/-1 扫描，统计 control_sync 活跃且**没有**任何
    compute/memory 重叠的周期，占关键路径的比例。这是模型假设的控制暴露下界——
    硬件实测的同核 scalar 占用远高于它，差值即暴露控制/同步赤字。

    返回 (model_exposed_frac, n_sync_ops, critical_path_cycles)；调度退化时返回 0。
    """
    n_sync = sum(1 for op in operations if op.op_name in _SYNC_OPS)
    if not operations:
        return 0.0, n_sync, 0
    critical_path = max(op.end_cycle for op in operations)
    if critical_path <= 0:
        return 0.0, n_sync, 0

    events: dict[int, Counter] = defaultdict(Counter)
    for op in operations:
        start, end = op.start_cycle, op.end_cycle
        if end <= start:
            continue
        cat = _ovl_category(op)
        if cat == "other":
            continue
        events[start][cat] += 1
        events[end][cat] -= 1

    active: Counter = Counter()
    state_cycles: Counter = Counter()
    prev = None
    for t in sorted(events):
        if prev is not None and t > prev:
            cats = frozenset(c for c, n in active.items() if n > 0)
            state_cycles[cats] += t - prev
        for cat, delta in events[t].items():
            active[cat] += delta
        prev = t

    exposed = sum(
        cyc
        for cats, cyc in state_cycles.items()
        if "control_sync" in cats and "compute" not in cats and "memory" not in cats
    )
    return exposed / critical_path, n_sync, critical_path



def run_from_files(
    op_summary_path: str | Path,
    desgraph_path: str | Path,
    calibration_path: str | Path | None = None,
    *,
    kernel_name: str | None = None,
    u_threshold: float = 0.80,
    r_threshold: float = 0.50,
    work_tolerance: float = 0.10,
    t_bound_us: float | None = None,
    calibration_db: CalibrationDB | None = None,
) -> OperatorBottleneckReport:
    """从 op_summary、DES graph 和 calibration 文件端到端运行分析。

    ``t_bound_us`` 是该 kernel 的 sound 紧 bound（loop-scaled 两级调度 bound，
    例如 46,109.91 µs），仅用于把暴露控制/同步赤字的 µs 估计封顶到 author
    headroom (elapsed - t_bound_us)。注意：本模块自身的 component throughput
    floor（compute_component_floor）只是吞吐下界，对 chunk_kda 这类非吞吐受限的
    kernel 过松（~225 µs），不能用作 author headroom 的基准；DES 原始 makespan
    又是 per-iteration（未 loop 放大）。因此紧 bound 必须由调用方提供，不提供时
    deficit_pts 仍给出（与分母无关），但 deficit_us 留空。"""

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

    des_metadata, des_input_warnings = read_des_graph_metadata(desgraph_path)
    extract = extract_hivm(desgraph_path)
    db = calibration_db if calibration_db is not None else load_calibration(calibration_path)
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
    report = analyze_operator_bottleneck(
        profile,
        component_bound,
        u_threshold=u_threshold,
        r_threshold=r_threshold,
        work_tolerance=work_tolerance,
    )
    report.hivm_bottleneck = diagnose_hivm_bottleneck_from_des_ops(
        extract.operations,
        db,
        des_metadata=des_metadata,
        input_warnings=des_input_warnings,
    )
    report.warnings.extend(
        f"hivm_bottleneck: {warning}"
        for warning in report.hivm_bottleneck.warnings
    )

    # —— 对论文的补充：量化暴露控制/同步赤字 ——
    # 当判定为暴露的 Scalar 控制导致的 Insufficient Parallelism 时，用 DES 调度的
    # 暴露控制比例（模型）和同核实测 scalar 占比（aiv_scalar/aiv_time）求差，给出
    # 定量赤字。两个分母都归一到单核时间线，匹配可比；µs 估计封顶到 author
    # headroom (elapsed - t_core_floor)。纯诊断，不改变 bound。
    if (
        report.diagnosis == "Insufficient Parallelism"
        and report.dominant_component == Component.SCALAR
    ):
        model_frac, n_sync, _ = _exposed_control_overlap(extract.operations)
        aiv_time = _float_cell(op_row, "aiv_time(us)")
        aiv_scalar = _float_cell(op_row, "aiv_scalar_time(us)")
        measured_frac = aiv_scalar / aiv_time if aiv_time > _EPSILON else 0.0
        report.exposed_control_frac_model = model_frac
        report.exposed_control_frac_measured = measured_frac
        report.exposed_control_deficit_pts = measured_frac - model_frac
        report.n_sync_ops = n_sync
        # µs 估计需要 sound 紧 bound 才能诚实封顶；只有调用方提供 t_bound_us 时才给。
        if t_bound_us is not None:
            author_headroom_us = elapsed_time_us - t_bound_us
            if author_headroom_us > 0:
                raw_us = max(0.0, measured_frac - model_frac) * elapsed_time_us
                report.exposed_control_deficit_us = min(raw_us, author_headroom_us)

    return report


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
        "exposed_control_frac_model": report.exposed_control_frac_model,
        "exposed_control_frac_measured": report.exposed_control_frac_measured,
        "exposed_control_deficit_pts": report.exposed_control_deficit_pts,
        "exposed_control_deficit_us": report.exposed_control_deficit_us,
        "n_sync_ops": report.n_sync_ops,
        "hivm_bottleneck": hivm_bottleneck_report_to_dict(report.hivm_bottleneck),
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


def emit_perfetto_trace_from_des(
    desgraph_path: str | Path,
    output_path: str | Path,
    *,
    clock_ghz: float | None = None,
) -> None:
    """Emit a Perfetto-compatible trace from a DES graph JSON.

    This mirrors the C++ ``HIVMAnalysisReport::emitPerfettoTrace()`` path,
    but starts from an already emitted DES graph instead of the in-memory
    ``HIVMAnalysisReport``.
    """

    with open(desgraph_path) as f:
        data = json.load(f)

    raw_ops = data.get("operations")
    if raw_ops is None:
        raw_ops = data.get("nodes", [])
    if not isinstance(raw_ops, list):
        raise ValueError(f"DES graph operations/nodes 不是数组: {desgraph_path}")

    effective_clock = clock_ghz or float(data.get("clock_ghz") or 1.85)
    events: list[dict] = []

    _append_perfetto_metadata(events)
    for op in raw_ops:
        if not isinstance(op, dict):
            continue
        if op.get("name") in {"pointer_cast", "convert_layout"}:
            continue
        events.append(_perfetto_duration_event(op, effective_clock))

    events.extend(_perfetto_sync_flow_events(raw_ops, effective_clock))
    trace = {"traceEvents": events, "displayTimeUnit": "us"}
    Path(output_path).write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")


def _append_perfetto_metadata(events: list[dict]) -> None:
    events.extend(
        [
            {
                "ph": "M",
                "pid": 1,
                "tid": 0,
                "name": "process_name",
                "args": {"name": "AIC (Cube Core)"},
            },
            {
                "ph": "M",
                "pid": 2,
                "tid": 0,
                "name": "process_name",
                "args": {"name": "AIV (Vector Core)"},
            },
            {
                "ph": "M",
                "pid": 3,
                "tid": 0,
                "name": "process_name",
                "args": {"name": "Shared"},
            },
        ]
    )
    for pipe in ("Cube", "MTE1", "CubeMTE2", "FixPipe"):
        events.append(
            {
                "ph": "M",
                "pid": 1,
                "tid": _perfetto_pipe_tid(pipe),
                "name": "thread_name",
                "args": {"name": pipe},
            }
        )
    events.append(
        {
            "ph": "M",
            "pid": 1,
            "tid": _perfetto_pipe_tid("Scalar"),
            "name": "thread_name",
            "args": {"name": "Scalar"},
        }
    )
    for pipe in ("Vector", "VectorMTE2", "MTE3"):
        events.append(
            {
                "ph": "M",
                "pid": 2,
                "tid": _perfetto_pipe_tid(pipe),
                "name": "thread_name",
                "args": {"name": pipe},
            }
        )
    events.append(
        {
            "ph": "M",
            "pid": 2,
            "tid": _perfetto_pipe_tid("Scalar"),
            "name": "thread_name",
            "args": {"name": "Scalar"},
        }
    )
    events.append(
        {
            "ph": "M",
            "pid": 3,
            "tid": _perfetto_pipe_tid("All"),
            "name": "thread_name",
            "args": {"name": "All"},
        }
    )


def _perfetto_duration_event(op: dict, clock_ghz: float) -> dict:
    pipe = _canonical_perfetto_pipe(str(op.get("pipe", "")))
    duration = _perfetto_op_duration(op)
    return {
        "ph": "X",
        "pid": _perfetto_pipe_pid(pipe, str(op.get("core_type", ""))),
        "tid": _perfetto_pipe_tid(pipe),
        "ts": round(_cycles_to_trace_us(_int_field(op, "start_cycle"), clock_ghz), 3),
        "dur": round(_cycles_to_trace_us(duration, clock_ghz), 3),
        "name": str(op.get("name", "")),
        "args": {
            "line": _int_field(op, "line"),
            "cycles": duration,
            "loop_multiplier": _int_field(op, "loop_multiplier", default=1),
            "bytes": _int_field(op, "bytes"),
            "elements": _int_field(op, "elements"),
            "event_id": str(op.get("event_id", "")),
            "event_generation": _int_field(op, "event_generation"),
            "sender_pipe": _canonical_perfetto_pipe(str(op.get("sender_pipe", "Unknown"))),
            "receiver_pipe": _canonical_perfetto_pipe(str(op.get("receiver_pipe", "Unknown"))),
            "read_buffers": _join_trace_values(op.get("read_buffers", [])),
            "write_buffers": _join_trace_values(op.get("write_buffers", [])),
            "read_versions": _join_trace_values(op.get("read_versions", [])),
            "write_versions": _join_trace_values(op.get("write_versions", [])),
            "core_type": str(op.get("core_type", "")),
            "sync": bool(op.get("is_sync", False)),
            "barrier": bool(op.get("is_barrier", False)),
            "src_space": str(op.get("src_space", "")),
            "dst_space": str(op.get("dst_space", "")),
            "elem_type": str(op.get("elem_type", "")),
        },
    }


def _perfetto_sync_flow_events(raw_ops: list, clock_ghz: float) -> list[dict]:
    set_ops: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    wait_ops: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for op in raw_ops:
        if not isinstance(op, dict) or not op.get("event_id"):
            continue
        event_id = str(op.get("event_id", ""))
        generation = _int_field(op, "event_generation")
        core = _perfetto_core_kind(str(op.get("core_type", "")))
        if op.get("name") == "sync_block_set":
            set_ops[(event_id, generation, core)].append(op)
        elif op.get("name") == "sync_block_wait":
            wait_source = "AIV" if core == "AIC" else "AIC"
            wait_ops[(event_id, generation, wait_source)].append(op)

    events: list[dict] = []
    flow_id = 0
    for key, sets in set_ops.items():
        waits = wait_ops.get(key, [])
        for set_op, wait_op in zip(sets, waits):
            set_pipe = _canonical_perfetto_pipe(str(set_op.get("pipe", "")))
            wait_pipe = _canonical_perfetto_pipe(str(wait_op.get("pipe", "")))
            events.append(
                {
                    "ph": "s",
                    "id": flow_id,
                    "pid": _perfetto_pipe_pid(set_pipe, str(set_op.get("core_type", ""))),
                    "tid": _perfetto_pipe_tid(set_pipe),
                    "ts": round(
                        _cycles_to_trace_us(
                            _int_field(set_op, "start_cycle") + _perfetto_op_duration(set_op),
                            clock_ghz,
                        ),
                        3,
                    ),
                    "name": "sync",
                    "cat": "sync",
                }
            )
            events.append(
                {
                    "ph": "f",
                    "id": flow_id,
                    "pid": _perfetto_pipe_pid(wait_pipe, str(wait_op.get("core_type", ""))),
                    "tid": _perfetto_pipe_tid(wait_pipe),
                    "ts": round(
                        _cycles_to_trace_us(_int_field(wait_op, "start_cycle"), clock_ghz),
                        3,
                    ),
                    "name": "sync",
                    "cat": "sync",
                    "bp": "e",
                }
            )
            flow_id += 1
    return events


def _canonical_perfetto_pipe(pipe: str) -> str:
    return {
        "PIPE_CUBE": "Cube",
        "PIPE_M": "Cube",
        "PIPE_MTE1": "MTE1",
        "PIPE_MTE2_C": "CubeMTE2",
        "PIPE_FIX": "FixPipe",
        "PIPE_S": "Scalar",
        "PIPE_V": "Vector",
        "PIPE_MTE2_V": "VectorMTE2",
        "PIPE_MTE3": "MTE3",
        "PIPE_ALL": "All",
        "PIPE_UNKNOWN": "Unknown",
    }.get(pipe, pipe or "Unknown")


def _perfetto_pipe_tid(pipe: str) -> int:
    return {
        "Cube": 1,
        "MTE1": 2,
        "CubeMTE2": 3,
        "FixPipe": 4,
        "Scalar": 5,
        "Vector": 1,
        "VectorMTE2": 2,
        "MTE3": 3,
        "All": 1,
        "Unknown": 5,
    }.get(pipe, 5)


def _perfetto_pipe_pid(pipe: str, core_type: str) -> int:
    if pipe in {"Cube", "MTE1", "CubeMTE2", "FixPipe"}:
        return 1
    if pipe in {"Vector", "VectorMTE2", "MTE3"}:
        return 2
    core = _perfetto_core_kind(core_type)
    if core == "AIC":
        return 1
    if core == "AIV":
        return 2
    return 3


def _perfetto_core_kind(core_type: str) -> str:
    if core_type in {"CUBE", "AIC"}:
        return "AIC"
    if core_type in {"VECTOR", "AIV"}:
        return "AIV"
    return "Shared"


def _cycles_to_trace_us(cycles: int | float, clock_ghz: float) -> float:
    if clock_ghz <= 0:
        clock_ghz = 1.85
    return float(cycles) / (clock_ghz * 1000.0)


def _perfetto_op_duration(op: dict) -> int:
    duration = _int_field(op, "duration")
    if duration:
        return duration
    return max(0, _int_field(op, "end_cycle") - _int_field(op, "start_cycle"))


def _join_trace_values(values) -> str:
    if not isinstance(values, list):
        return ""
    return ";".join(str(value) for value in values)


def _int_field(op: dict, field_name: str, *, default: int = 0) -> int:
    try:
        return int(op.get(field_name, default) or default)
    except (TypeError, ValueError):
        return default


def format_text_report(report: OperatorBottleneckReport) -> str:
    """Format the complete report for stdout."""

    parts = [_format_operator_bottleneck_report(report)]
    if report.hivm_bottleneck is not None:
        parts.append(_format_hivm_bottleneck_report(report.hivm_bottleneck))
    return "\n".join(part.rstrip() for part in parts if part).rstrip() + "\n"


def _format_operator_bottleneck_report(report: OperatorBottleneckReport) -> str:
    dominant = (
        report.component_results.get(report.dominant_component.value)
        if report.dominant_component
        else None
    )
    evidence_components = _operator_evidence_components(report, dominant)

    lines = [
        "=== Profile Utilization Diagnosis ===",
        f"结论: {report.diagnosis}",
        "证据:",
        f"  - kernel={report.kernel_name}, elapsed={_fmt_us(report.elapsed_time_us)} us",
    ]

    if dominant is not None:
        lines.append(
            "  - 主导 component="
            f"{dominant.component.value}, item={dominant.dominant_item or 'None'}, "
            f"U={dominant.u_utilization * 100.0:.1f}%, "
            f"R={dominant.r_residency * 100.0:.1f}%, "
            f"E={dominant.e_efficiency * 100.0:.1f}%"
        )
    elif evidence_components:
        max_r = max(item.r_residency for item in evidence_components)
        lines.append(
            "  - 没有 component 达到高驻留/高利用阈值；"
            f"最高 R={max_r * 100.0:.1f}%"
        )
    else:
        lines.append("  - 未找到可用于 A/I/U/R/E 分析的有效 component")

    if report.exposed_control_frac_model is not None:
        deficit_pts = (report.exposed_control_deficit_pts or 0.0) * 100.0
        measured = (report.exposed_control_frac_measured or 0.0) * 100.0
        model = report.exposed_control_frac_model * 100.0
        lines.append(
            "  - 暴露控制/同步: "
            f"model={model:.1f}%, measured={measured:.1f}%, "
            f"deficit={deficit_pts:.1f} pts, n_sync_ops={report.n_sync_ops}"
        )
        if report.exposed_control_deficit_us is not None:
            lines.append(
                f"  - 暴露控制/同步赤字约 {_fmt_us(report.exposed_control_deficit_us)} us"
            )

    lines.append("处理建议:")
    for suggestion in _operator_suggestions(report):
        lines.append(f"  -> {suggestion}")

    if evidence_components:
        lines.append("关键指标:")
        for result in evidence_components[:3]:
            lines.append(
                "  - "
                f"{result.component.value}: "
                f"U={result.u_utilization * 100.0:.1f}%, "
                f"R={result.r_residency * 100.0:.1f}%, "
                f"E={result.e_efficiency * 100.0:.1f}%, "
                f"active={_fmt_us(result.active_time_us)} us, "
                f"work={_fmt_number(result.work_done)}"
            )
            if result.dominant_item:
                lines[-1] += f", item={result.dominant_item}"

    if report.warnings:
        lines.append("Warnings:")
        for warning in report.warnings:
            lines.append(f"  - {warning}")

    return "\n".join(lines)


def _operator_evidence_components(
    report: OperatorBottleneckReport,
    dominant: ComponentUtilization | None,
) -> list[ComponentUtilization]:
    if dominant is not None:
        return [dominant]
    return sorted(
        report.component_results.values(),
        key=lambda item: (
            item.r_residency,
            item.u_utilization,
            item.active_time_us,
        ),
        reverse=True,
    )


def _operator_suggestions(report: OperatorBottleneckReport) -> list[str]:
    diagnosis = report.diagnosis
    if diagnosis == "Compute Bound":
        return [
            "计算侧已接近理论 ceiling；优先减少计算量、调整精度或提升 arithmetic intensity",
            "检查 tile/MN/K 形状和算子融合，避免额外启动和同步开销",
        ]
    if diagnosis == "MTE Bound":
        return [
            "数据搬运侧已接近理论 ceiling；优先减少 bytes 或增加片上复用",
            "用 multi-buffer/software pipeline 让搬运与计算重叠",
        ]
    if diagnosis == "Inefficient Compute":
        return [
            "计算单元驻留较高但效率低；检查 vector/cube 指令形态、mask/repeat 和 tile 是否过小",
            "对照 DES 中主导 compute op，优先处理高 R 低 E 的 component",
        ]
    if diagnosis == "Inefficient MTE":
        return [
            "MTE 驻留较高但有效带宽低；检查传输路径、对齐、burst/packet size 和 tile 粒度",
            "合并小搬运或提高 reuse，减少单位有效 bytes 的启动成本",
        ]
    if diagnosis == "Insufficient Parallelism":
        suggestions = [
            "整体驻留不足；优先增加 pipeline depth、multi-buffer 或并行 tile 数",
            "结合 HIVM Bottleneck Diagnosis 查看是 pipe 不均衡、同步等待还是全局 barrier 导致",
        ]
        if report.dominant_component == Component.SCALAR:
            suggestions.insert(
                0,
                "高驻留但无 work 的 Scalar 指向暴露控制/同步；优先减少 barrier/wait 或让控制与计算/搬运重叠",
            )
        return suggestions
    if diagnosis == "Insufficient Data":
        return [
            "补齐 op_summary、DES work 和 calibration 后再判断",
            "检查 elapsed_time、active_time、bytes/flops/elements 的单位是否一致",
        ]
    return [
        "检查主导 component 的 U/R/E，并结合 HIVM 结构诊断定位下一步优化方向",
    ]


def _format_hivm_bottleneck_report(report: HIVMBottleneckReport) -> str:
    """Format PR17 diagnosis like HIVMBottleneckReport::print()."""

    lines = [
        "",
        "=== Bottleneck Diagnosis ===",
        f"Global root cause: {report.global_root_cause}",
        f"  Evidence: {report.global_evidence}",
        f"  {report.global_explanation}",
    ]
    for suggestion in report.global_suggestions:
        lines.append(f"  -> {suggestion}")

    lines.extend(
        [
            "",
            "Sync/barrier overhead (from syncCycles/barrierCycles/oneIterationCycles):",
            "  syncCycles/oneIterationCycles = "
            f"{report.sync_overhead_ratio:.1f}%",
            "  barrierCycles/oneIterationCycles = "
            f"{report.barrier_overhead_ratio:.1f}%",
            "",
            "Pipeline diagnosis:",
        ]
    )

    pipeline = report.pipeline_diagnosis
    if pipeline is None:
        lines.extend(
            [
                "  Evidence: ",
                "  Bottleneck pipe: Unknown",
                "  Imbalance ratio (weightedPipeCycles max/min): 0.0x",
                "  Insufficient data to determine pipeline root cause",
            ]
        )
    else:
        lines.extend(
            [
                f"  Evidence: {pipeline.evidence}",
                f"  Bottleneck pipe: {pipeline.bottleneck_pipe}",
                "  Imbalance ratio (weightedPipeCycles max/min): "
                f"{pipeline.imbalance_ratio:.1f}x",
                f"  {pipeline.explanation}",
            ]
        )
        for suggestion in pipeline.suggestions:
            lines.append(f"  -> {suggestion}")

    sorted_ops = sorted(
        (
            diag
            for diag in report.op_diagnoses
            if diag.root_cause != "SyncOverhead"
        ),
        key=lambda item: item.actual_cycles,
        reverse=True,
    )
    limit = min(10, len(sorted_ops))
    lines.append("")
    lines.append(
        f"Per-op diagnosis (top {limit} by duration, excluding sync ops):"
    )
    for diag in sorted_ops[:limit]:
        lines.extend(
            [
                "  line "
                f"{diag.line_number} {diag.op_name} [{diag.pipe}]: "
                f"{diag.root_cause}",
                f"    Evidence: {diag.evidence}",
                f"    {diag.explanation}",
                "    duration="
                f"{_fmt_cycles(diag.actual_cycles)} cyc, theoretical_min="
                f"{_fmt_cycles(diag.theoretical_min_cycles)} cyc, overhead="
                f"{diag.overhead_ratio * 100.0:.1f}%",
            ]
        )
        for suggestion in diag.suggestions:
            lines.append(f"    -> {suggestion}")

    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in report.warnings:
            lines.append(f"  - {warning}")

    return "\n".join(lines)


def _fmt_number(value: float) -> str:
    if abs(value) >= 1000.0:
        return f"{value:.0f}"
    return f"{value:.6g}"


def _fmt_us(value: float) -> str:
    if abs(value) >= 100.0:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.6g}"


def _fmt_cycles(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _row_duration_us(row: dict[str, str]) -> float:
    return _float_cell(
        row,
        "Task Duration(us)",
        "duration(us)",
        "Duration(us)",
        "duration_us",
    )


def _read_op_summary_row(path: str | Path, kernel_name: str | None) -> dict[str, str]:
    """选取要分析的 op_summary 行。

    一个 grid kernel 在 msprof 里会有多条 shard 行（每个 AI core 一条），它们
    在不同 core 上**并发**执行。因此墙钟时间应取最长的那条（关键 core），而不是
    第一条或求和——求和会把并发时间错误地累加。``_float_cell`` 已容忍 'N/A'，
    所以汇总/表头行的时长会变成 0，不会被误选。
    """
    with open(path) as f:
        reader = csv.DictReader(line for line in f if not line.strip().startswith("#"))
        rows = list(reader)
    if not rows:
        raise ValueError(f"op_summary 中没有可读取的行: {path}")
    if kernel_name is None:
        candidates = rows
    else:
        normalized = kernel_name.lower()
        candidates = [
            row
            for row in rows
            if normalized in _cell(row, "Op Name", "op_name", "Name").lower()
        ]
        if not candidates:
            raise ValueError(f"op_summary 中找不到 kernel: {kernel_name}")
    return max(candidates, key=_row_duration_us)


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
        # 容忍 'N/A'、空白以及带单位后缀的单元格（例如 '12.3 us'）。
        # msprof 的汇总/表头行常含 'N/A'，按缺失(0.0)处理而不是抛异常。
        try:
            return float(value.split()[0])
        except (ValueError, IndexError):
            return 0.0


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
        "--t-bound-us",
        type=float,
        default=None,
        help="Sound loop-scaled bound (us) used only to cap the exposed-control "
        "deficit us estimate at author headroom (elapsed - t_bound_us).",
    )
    parser.add_argument(
        "--output-file",
        default="data/profile_utilization_inputs/profile_utilization_report.json",
        help="Path to write the JSON report",
    )
    parser.add_argument(
        "--perfetto-trace-file",
        help="Optional path to write a Perfetto-compatible trace JSON from the DES timeline",
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
        t_bound_us=args.t_bound_us,
    )

    output = json.dumps(report_to_dict(report), ensure_ascii=False, indent=2)
    Path(args.output_file).write_text(output + "\n")
    if args.perfetto_trace_file:
        emit_perfetto_trace_from_des(args.des_graph, args.perfetto_trace_file)
    print(format_text_report(report), end="")
    if args.perfetto_trace_file:
        print(f"\nPerfetto trace: {args.perfetto_trace_file}")
    print(f"\nJSON report: {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
