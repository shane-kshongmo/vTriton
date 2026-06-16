"""HIVM bottleneck diagnosis for already extracted DES operations.

This module consumes already extracted DES operations plus the calibration DB.
It does not parse raw HIVM MLIR, rerun scheduling, or re-estimate durations.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..calibration.constants import CalibrationDB, DType


_EPSILON = 1e-12
_SYNC_OPS = {"wait_flag", "set_flag", "pipe_barrier", "sync_block_wait", "sync_block_set"}

_MODEL_TRANSFER_PIPES = {
    "PIPE_MTE2_V", "PIPE_MTE2_C", "PIPE_MTE3",
    "VectorMTE2", "CubeMTE2", "MTE3",
}
_MODEL_COMPUTE_PIPES = {"PIPE_V", "PIPE_M", "PIPE_MTE1", "Vector", "Cube", "MTE1"}
_MODEL_MEMORY_PIPES = set(_MODEL_TRANSFER_PIPES)
_MODEL_GLOBAL_COMPUTE_PIPES = {"PIPE_V", "PIPE_M", "Vector", "Cube"}
_MODEL_FIXPIPE_PIPES = {"PIPE_FIX", "FixPipe"}
_ALL_PIPES = {"PIPE_ALL", "All"}
_UNKNOWN_PIPES = {"PIPE_UNKNOWN", "Unknown", ""}
_MODEL_STARTUP_FALLBACK = {
    "vector": 35.0,
    "mte2": 50.0,
    "mte3": 40.0,
    "cube": 20.0,
    "fixpipe": 30.0,
    "mte1": 25.0,
    "scalar": 1.0,
    "pipe_barrier": 64.0,
}
_REQUIRED_DES_FIELDS = {
    "id", "name", "pipe", "duration", "start_cycle", "end_cycle",
    "depends_on", "is_sync", "is_barrier", "bytes", "elements",
    "flops", "loop_multiplier", "src_space", "dst_space", "elem_type",
}
_OPTIONAL_DES_FIELDS = {"repeat", "mask", "line", "core_type", "event_id"}
_EMPTY_DES_FIELDS = ("src_space", "dst_space", "elem_type")
_MEMORY_FALLBACK_PATHS = {
    "PIPE_MTE2_V": [("gm", "ub")],
    "PIPE_MTE2_C": [("gm", "l1")],
    "PIPE_MTE3": [("ub", "gm")],
    "PIPE_FIX": [("l0c", "gm"), ("ub", "gm")],
    "VectorMTE2": [("gm", "ub")],
    "CubeMTE2": [("gm", "l1")],
    "MTE3": [("ub", "gm")],
    "FixPipe": [("l0c", "gm"), ("ub", "gm")],
}
_PIPE_STARTUP_KEYS = {
    "PIPE_V": "vector",
    "Vector": "vector",
    "PIPE_MTE2_V": "mte2",
    "PIPE_MTE2_C": "mte2",
    "VectorMTE2": "mte2",
    "CubeMTE2": "mte2",
    "PIPE_MTE3": "mte3",
    "MTE3": "mte3",
    "PIPE_FIX": "fixpipe",
    "FixPipe": "fixpipe",
    "PIPE_M": "cube",
    "Cube": "cube",
    "PIPE_MTE1": "mte1",
    "MTE1": "cxx_default",
    "PIPE_S": "scalar",
    "Scalar": "scalar",
    "PIPE_ALL": "pipe_barrier",
    "All": "pipe_barrier",
}


@dataclass
class HIVMOpDiagnosis:
    """Python counterpart of C++ OpDiagnosis."""

    op_id: int
    line_number: int
    op_name: str
    pipe: str
    root_cause: str
    evidence: str
    explanation: str
    suggestions: list[str] = field(default_factory=list)
    theoretical_min_cycles: float = 0.0
    actual_cycles: float = 0.0
    overhead_ratio: float = 0.0


@dataclass
class HIVMPipelineDiagnosis:
    """Python counterpart of C++ PipelineDiagnosis."""

    root_cause: str
    bottleneck_pipe: str = "Unknown"
    imbalance_ratio: float = 0.0
    evidence: str = ""
    explanation: str = ""
    suggestions: list[str] = field(default_factory=list)


@dataclass
class HIVMBottleneckReport:
    """Python counterpart of C++ HIVMBottleneckReport."""

    global_root_cause: str
    global_evidence: str
    global_explanation: str
    global_suggestions: list[str] = field(default_factory=list)
    sync_overhead_ratio: float = 0.0
    barrier_overhead_ratio: float = 0.0
    one_iteration_cycles: float = 0.0
    weighted_cycles: float = 0.0
    pipe_busy_cycles: dict[str, float] = field(default_factory=dict)
    weighted_pipe_cycles: dict[str, float] = field(default_factory=dict)
    pipeline_diagnosis: HIVMPipelineDiagnosis | None = None
    op_diagnoses: list[HIVMOpDiagnosis] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def diagnose_hivm_bottleneck_from_des_ops(
    operations,
    db: CalibrationDB,
    *,
    des_metadata: dict[int, dict] | None = None,
    input_warnings: list[str] | None = None,
    top_k: int = 10,
) -> HIVMBottleneckReport:
    """从已解析 DES graph 复现 HIVMBottleneckDiagnoser 的瓶颈诊断。

    这部分刻意消费 ``extract_hivm()`` 的 OpRecord 和现有 CalibrationDB，
    不重新解析 HIVM MLIR、不重新 estimateDuration、不重新调度。
    因此它回答的是：给定 DES graph 里已有的模型时间线，HIVM bottleneck
    diagnosis 会如何解释瓶颈。
    """

    operations = list(operations)
    metadata = des_metadata or {}
    warnings = list(input_warnings or [])
    warn_keys: set[str] = set()
    if des_metadata is None:
        _warn_once(
            warnings,
            warn_keys,
            "missing_des_metadata",
            "未提供原始 DES metadata；line/is_sync/is_barrier 只能从 op name/pipe 推断",
        )

    summary = _summarize_model_operations(
        operations,
        metadata,
        warnings,
        warn_keys,
    )
    pipeline = _diagnose_model_pipeline(summary)
    global_root = _diagnose_model_global(summary, pipeline)
    global_evidence, global_explanation, global_suggestions = _model_global_details(
        global_root,
        summary,
        pipeline,
    )

    op_diagnoses = [
        _diagnose_model_op(op, db, metadata, warnings, warn_keys)
        for op in operations
        if _model_op_duration(op, metadata, warnings, warn_keys) > _EPSILON
    ]
    op_diagnoses.sort(key=lambda item: item.actual_cycles, reverse=True)

    return HIVMBottleneckReport(
        global_root_cause=global_root,
        global_evidence=global_evidence,
        global_explanation=global_explanation,
        global_suggestions=global_suggestions,
        sync_overhead_ratio=summary["sync_ratio"],
        barrier_overhead_ratio=summary["barrier_ratio"],
        one_iteration_cycles=summary["one_iteration_cycles"],
        weighted_cycles=summary["weighted_cycles"],
        pipe_busy_cycles=summary["pipe_busy_cycles"],
        weighted_pipe_cycles=summary["weighted_pipe_cycles"],
        pipeline_diagnosis=pipeline,
        op_diagnoses=op_diagnoses[:top_k],
        warnings=warnings,
    )


def read_des_graph_metadata(path: str | Path) -> tuple[dict[int, dict], list[str]]:
    """读取原始 DES JSON 字段质量信息，不改变 extract_hivm 的输入路径。"""

    with open(path) as f:
        data = json.load(f)

    warnings: list[str] = []
    if "schedule_truncated" not in data:
        warnings.append("DES graph 缺少顶层 schedule_truncated；无法确认调度是否完整")
    elif data.get("schedule_truncated"):
        warnings.append("DES graph 标记 schedule_truncated=true；模型诊断结果不可靠")

    raw_ops = data.get("operations")
    if raw_ops is None:
        raw_ops = data.get("nodes", [])
        warnings.append("DES graph 使用 legacy nodes 字段；建议统一为 operations")
    if not isinstance(raw_ops, list):
        return {}, warnings + ["DES graph operations/nodes 不是数组；无法做 HIVM bottleneck diagnosis"]

    missing_counts: Counter[str] = Counter()
    empty_counts: Counter[str] = Counter()
    metadata: dict[int, dict] = {}

    for idx, op in enumerate(raw_ops):
        if not isinstance(op, dict):
            warnings.append(f"DES operation[{idx}] 不是 object；已跳过 metadata")
            continue
        for field_name in _REQUIRED_DES_FIELDS | _OPTIONAL_DES_FIELDS:
            if field_name not in op:
                missing_counts[field_name] += 1
        for field_name in _EMPTY_DES_FIELDS:
            if op.get(field_name, "") == "":
                empty_counts[field_name] += 1
        try:
            metadata[int(op.get("id", idx))] = op
        except (TypeError, ValueError):
            warnings.append(f"DES operation[{idx}] id 无法转成 int；line/is_sync metadata 可能丢失")

    n_ops = len(raw_ops)
    for field_name in sorted(_REQUIRED_DES_FIELDS):
        count = missing_counts.get(field_name, 0)
        if count:
            warnings.append(f"DES graph 有 {count}/{n_ops} 个 op 缺少必需字段 {field_name}")
    for field_name in sorted(_OPTIONAL_DES_FIELDS):
        count = missing_counts.get(field_name, 0)
        if count:
            warnings.append(f"DES graph 有 {count}/{n_ops} 个 op 缺少可选字段 {field_name}")

    if n_ops:
        compute_like_ops = [
            op
            for op in raw_ops
            if isinstance(op, dict)
            and (op.get("pipe", "") in _MODEL_COMPUTE_PIPES or op.get("elements", 0) > 0)
        ]
        if compute_like_ops and all(op.get("flops", 0) <= 0 for op in compute_like_ops):
            warnings.append("DES graph 中 flops 全部为 0；compute theoretical_min 需要回退或依赖 extractor 推断")
        for field_name in ("src_space", "dst_space"):
            count = empty_counts.get(field_name, 0)
            if count / n_ops > 0.5:
                warnings.append(
                    f"DES graph 中 {field_name} 为空的 op 占比 {count / n_ops:.1%}；memory path 诊断会使用带宽回退"
                )

    return metadata, warnings


def hivm_bottleneck_report_to_dict(report: HIVMBottleneckReport | None) -> dict | None:
    if report is None:
        return None
    return asdict(report)


def _summarize_model_operations(
    operations,
    des_metadata: dict[int, dict],
    warnings: list[str],
    warn_keys: set[str],
) -> dict:
    pipe_busy: dict[str, float] = defaultdict(float)
    weighted_pipe: dict[str, float] = defaultdict(float)
    total_busy = 0.0
    sync_cycles = 0.0
    barrier_cycles = 0.0
    one_iteration = 0.0
    global_barrier_weighted = 0.0

    for op in operations:
        duration = _model_op_duration(op, des_metadata, warnings, warn_keys)
        end_cycle = _model_op_end_cycle(op, des_metadata, warnings, warn_keys)
        loop_multiplier = max(1.0, float(getattr(op, "loop_multiplier", 1) or 1))
        pipe = getattr(op, "pipe", "") or "Unknown"

        total_busy += duration
        one_iteration = max(one_iteration, end_cycle)
        is_barrier = _model_op_is_barrier(op, des_metadata)
        if _model_op_is_sync(op, des_metadata):
            sync_cycles += duration
        if is_barrier:
            barrier_cycles += duration
        if pipe not in _ALL_PIPES and pipe not in _UNKNOWN_PIPES:
            pipe_busy[pipe] += duration
            weighted_pipe[pipe] += duration * loop_multiplier
        if is_barrier and pipe in _ALL_PIPES:
            global_barrier_weighted += duration * loop_multiplier

    weighted_cycles = max(weighted_pipe.values(), default=0.0) + global_barrier_weighted
    if weighted_cycles <= _EPSILON:
        weighted_cycles = one_iteration
    if one_iteration <= _EPSILON and operations:
        _warn_once(
            warnings,
            warn_keys,
            "zero_critical_path",
            "DES graph 的 start/end_cycle 无法形成正数关键路径；模型利用率和 sync ratio 可能无效",
        )

    return {
        "total_busy_cycles": total_busy,
        "sync_cycles": sync_cycles,
        "barrier_cycles": barrier_cycles,
        "one_iteration_cycles": one_iteration,
        "weighted_cycles": weighted_cycles,
        "pipe_busy_cycles": dict(pipe_busy),
        "weighted_pipe_cycles": dict(weighted_pipe),
        "sync_ratio": _safe_ratio(sync_cycles, one_iteration) * 100.0,
        "barrier_ratio": _safe_ratio(barrier_cycles, one_iteration) * 100.0,
    }


def _diagnose_model_op(
    op,
    db: CalibrationDB,
    des_metadata: dict[int, dict],
    warnings: list[str],
    warn_keys: set[str],
) -> HIVMOpDiagnosis:
    op_id = int(getattr(op, "op_id", 0))
    op_name = getattr(op, "op_name", "")
    pipe = getattr(op, "pipe", "") or "Unknown"
    duration = _model_op_duration(op, des_metadata, warnings, warn_keys)
    theoretical_min = _model_theoretical_min_cycles(
        op,
        db,
        des_metadata,
        warnings,
        warn_keys,
    )
    overhead_ratio = _safe_ratio(duration - theoretical_min, duration)
    line_number = int(des_metadata.get(op_id, {}).get("line", 0) or 0)
    is_sync = _model_op_is_sync(op, des_metadata)
    is_barrier = _model_op_is_barrier(op, des_metadata)
    is_control = is_sync or is_barrier
    bytes_transferred = getattr(op, "bytes_transferred", 0)
    elements = getattr(op, "elements", 0)
    flops = getattr(op, "flops", 0)

    if is_control:
        root = "SyncOverhead"
        evidence = (
            f"[is_sync={is_sync}, "
            f"is_barrier={is_barrier}, "
            f"duration={duration:.0f} cyc]"
        )
        explanation = f"{op_name} on {pipe}: synchronization/control op with no arithmetic/transfer work"
        suggestions = []
        if op_name == "pipe_barrier" and pipe in _ALL_PIPES:
            suggestions.append("用更细粒度 set_flag/wait_flag 替代 PIPE_ALL barrier")
            suggestions.append("增加 multi-buffer pipeline depth 以降低 barrier 频率")
        elif op_name in {"wait_flag", "sync_block_wait"}:
            suggestions.append("调整 producer/consumer 排序，减少 wait stall 时间")
    else:
        startup = _startup_cycles_for_pipe(pipe, db, warnings, warn_keys)
        suggestions = []
        root = "LowParallelism"
        evidence = f"[duration={duration:.0f} cyc, pipe={pipe}]"
        explanation = f"{op_name} on {pipe}: unclassified or scalar/control-like op"

    if not is_control and pipe in _MODEL_TRANSFER_PIPES and bytes_transferred > 0:
        transfer_only = duration - startup
        if transfer_only <= 0 or startup > transfer_only:
            root = "StartupOverhead"
            evidence = (
                f"[duration={duration:.0f} cyc, startup_latency={startup:.0f} cyc, "
                f"bytes={bytes_transferred}, "
                f"startup/duration={_safe_ratio(startup, duration) * 100.0:.1f}%]"
            )
            explanation = (
                f"{op_name} on {pipe}: startup latency dominates transfer work; "
                f"bytes={bytes_transferred}"
            )
            suggestions = ["增大 tile 或合并相邻 transfer，摊薄每次 DMA startup"]
        else:
            root = "BandwidthBound"
            evidence = (
                f"[duration={duration:.0f} cyc, transfer_cycles={transfer_only:.0f} cyc, "
                f"bytes={bytes_transferred}, "
                f"src_space={getattr(op, 'src_space', '')}, dst_space={getattr(op, 'dst_space', '')}]"
            )
            explanation = f"{op_name} on {pipe}: data movement dominates modeled duration"
            suggestions = ["减少数据搬运量或提高 tile reuse", "用 software pipeline 将 transfer 与 compute 重叠"]

    elif not is_control and pipe in _MODEL_COMPUTE_PIPES:
        compute_only = duration - startup
        if startup > compute_only and compute_only > 0:
            root = "StartupOverhead"
            evidence = (
                f"[duration={duration:.0f} cyc, startup_latency={startup:.0f} cyc, "
                f"compute_work={compute_only:.0f} cyc, elements={elements}]"
            )
            explanation = f"{op_name} on {pipe}: startup latency dominates compute work"
            suggestions = ["增大 tile size，摊薄 compute startup"]
        else:
            root = "ComputeBound"
            evidence = (
                f"[duration={duration:.0f} cyc, elements={elements}, "
                f"flops={flops}, compute_work={compute_only:.0f} cyc]"
            )
            explanation = f"{op_name} on {pipe}: compute work dominates modeled duration"
            suggestions = ["提高 arithmetic intensity", "将 compute 与 MTE prefetch 重叠"]

    elif not is_control and pipe in _MODEL_FIXPIPE_PIPES and bytes_transferred > 0:
        root = "BandwidthBound"
        evidence = (
            f"[duration={duration:.0f} cyc, bytes={bytes_transferred}, "
            f"dst_space={getattr(op, 'dst_space', '')}]"
        )
        explanation = (
            f"fixpipe: draining {bytes_transferred} bytes from L0C to "
            f"{getattr(op, 'dst_space', '')} (duration={duration:.0f} cyc)"
        )
        suggestions = ["增大 tile size，在 draining 前填满 Cube pipeline"]

    return HIVMOpDiagnosis(
        op_id=op_id,
        line_number=line_number,
        op_name=op_name,
        pipe=pipe,
        root_cause=root,
        evidence=evidence,
        explanation=explanation,
        suggestions=suggestions,
        theoretical_min_cycles=theoretical_min,
        actual_cycles=duration,
        overhead_ratio=overhead_ratio,
    )


def _diagnose_model_pipeline(summary: dict) -> HIVMPipelineDiagnosis:
    weighted = summary["weighted_pipe_cycles"]
    if not weighted:
        return HIVMPipelineDiagnosis(
            root_cause="LowParallelism",
            explanation="No pipe activity recorded in DES graph",
        )

    max_pipe, max_weighted = max(weighted.items(), key=lambda item: item[1])
    positive = [(pipe, value) for pipe, value in weighted.items() if value > _EPSILON]
    min_pipe, min_weighted = min(positive, key=lambda item: item[1]) if positive else (max_pipe, max_weighted)
    imbalance = _safe_ratio(max_weighted, min_weighted)

    busy = summary["pipe_busy_cycles"]
    one_iter = summary["one_iteration_cycles"]
    utils = [_safe_ratio(value, one_iter) * 100.0 for value in busy.values() if value > 0]
    max_util = max(utils, default=0.0)
    min_util = min(utils, default=0.0)

    if imbalance > 3.0:
        return HIVMPipelineDiagnosis(
            root_cause="PipelineImbalance",
            bottleneck_pipe=max_pipe,
            imbalance_ratio=imbalance,
            evidence=f"[weightedPipeCycles[{max_pipe}]={max_weighted:.0f}, weightedPipeCycles[{min_pipe}]={min_weighted:.0f}, ratio={imbalance:.1f}x]",
            explanation=f"{max_pipe} weighted cycles dominate {min_pipe} by {imbalance:.1f}x",
            suggestions=[
                f"将 {max_pipe} 工作与其他 pipe 通过 multi-buffer/software pipeline 重叠",
                "检查 Cube/Vector split 是否能更均衡利用 AIC/AIV",
            ],
        )

    if max_util > 60.0 and 0.0 < min_util < 20.0:
        return HIVMPipelineDiagnosis(
            root_cause="PipelineImbalance",
            bottleneck_pipe=max_pipe,
            imbalance_ratio=imbalance,
            evidence=f"[pipe utilization max={max_util:.1f}%, min={min_util:.1f}%]",
            explanation="one pipe is busy while another is mostly idle",
            suggestions=["增加 software pipeline depth 填充 idle pipe slots"],
        )

    if max_pipe in _MODEL_MEMORY_PIPES:
        return HIVMPipelineDiagnosis(
            root_cause="BandwidthBound",
            bottleneck_pipe=max_pipe,
            imbalance_ratio=imbalance,
            evidence=f"[max weightedPipeCycles on {max_pipe}={max_weighted:.0f}]",
            explanation=f"memory pipe {max_pipe} has the highest weighted cycles",
            suggestions=["减少数据搬运或提高 tile reuse"],
        )
    if max_pipe in _MODEL_GLOBAL_COMPUTE_PIPES:
        return HIVMPipelineDiagnosis(
            root_cause="ComputeBound",
            bottleneck_pipe=max_pipe,
            imbalance_ratio=imbalance,
            evidence=f"[max weightedPipeCycles on {max_pipe}={max_weighted:.0f}]",
            explanation=f"compute pipe {max_pipe} has the highest weighted cycles",
            suggestions=["提高 compute-to-data ratio"],
        )
    return HIVMPipelineDiagnosis(
        root_cause="BandwidthBound",
        bottleneck_pipe=max_pipe,
        imbalance_ratio=imbalance,
        evidence=f"[max weightedPipeCycles on {max_pipe}={max_weighted:.0f}]",
        explanation=f"bottleneck pipe is {max_pipe}",
    )


def _diagnose_model_global(summary: dict, pipeline: HIVMPipelineDiagnosis) -> str:
    if summary["sync_ratio"] > 20.0 or summary["barrier_ratio"] > 15.0:
        return "SyncOverhead"
    if pipeline.root_cause == "PipelineImbalance":
        return "PipelineImbalance"

    weighted = summary["weighted_pipe_cycles"]
    if weighted:
        max_pipe = max(weighted.items(), key=lambda item: item[1])[0]
        if max_pipe in _MODEL_MEMORY_PIPES:
            return "BandwidthBound"
        if max_pipe in _MODEL_GLOBAL_COMPUTE_PIPES:
            return "ComputeBound"
    return pipeline.root_cause


def _model_global_details(
    root: str,
    summary: dict,
    pipeline: HIVMPipelineDiagnosis,
) -> tuple[str, str, list[str]]:
    if root == "SyncOverhead":
        evidence = (
            f"[syncCycles={summary['sync_cycles']:.0f} cyc, "
            f"barrierCycles={summary['barrier_cycles']:.0f} cyc, "
            f"oneIterationCycles={summary['one_iteration_cycles']:.0f} cyc, "
            f"sync={summary['sync_ratio']:.1f}%, barrier={summary['barrier_ratio']:.1f}%]"
        )
        explanation = "sync/barrier ratio exceeds HIVMBottleneckDiagnoser thresholds (sync>20% or barrier>15%)"
        suggestions = [
            "减少全局 barrier，优先使用局部 set_flag/wait_flag",
            "增加 multi-buffer pipeline depth 以隐藏同步等待",
        ]
        return evidence, explanation, suggestions
    if root == "PipelineImbalance":
        return pipeline.evidence, pipeline.explanation, list(pipeline.suggestions)

    weighted = summary["weighted_pipe_cycles"]
    max_pipe = "Unknown"
    max_value = 0.0
    if weighted:
        max_pipe, max_value = max(weighted.items(), key=lambda item: item[1])
    evidence = f"[max weightedPipeCycles on {max_pipe}={max_value:.0f}]"
    if root == "BandwidthBound":
        return evidence, f"memory pipe {max_pipe} dominates modeled weighted cycles", [
            "减少 data movement 或增加 tile reuse",
            "用 software pipeline 将 transfer 与 compute 重叠",
        ]
    if root == "ComputeBound":
        return evidence, f"compute pipe {max_pipe} dominates modeled weighted cycles", [
            "提高 arithmetic intensity",
            "融合相邻 compute ops 以减少启动和同步开销",
        ]
    return evidence, "insufficient modeled activity to classify root cause", []


def _model_theoretical_min_cycles(
    op,
    db: CalibrationDB,
    des_metadata: dict[int, dict],
    warnings: list[str],
    warn_keys: set[str],
) -> float:
    pipe = getattr(op, "pipe", "") or "Unknown"
    bytes_transferred = float(getattr(op, "bytes_transferred", 0) or 0)
    elements = float(getattr(op, "elements", 0) or 0)
    duration = _model_op_duration(op, des_metadata, warnings, warn_keys)

    if bytes_transferred <= 0 and elements <= 0:
        return 0.0

    if pipe in _MODEL_TRANSFER_PIPES:
        if bytes_transferred <= 0:
            return 0.0
        bw = _model_memory_bw_bytes_per_cycle(op, db, warnings, warn_keys)
        return math.ceil(bytes_transferred / bw) if bw > _EPSILON else duration

    if pipe in {"PIPE_V", "Vector"}:
        if elements <= 0:
            return 0.0
        width = max(1, int(db.vector.vec_width_bytes / 4))
        return _startup_cycles_for_pipe(pipe, db, warnings, warn_keys) + math.ceil(elements / width)

    if pipe in {"PIPE_M", "Cube"}:
        flops = float(getattr(op, "flops", 0) or 0)
        if flops <= 0:
            _warn_once(
                warnings,
                warn_keys,
                "cube_flops_missing",
                "Cube op 的 flops 缺失或为 0；HIVM bottleneck theoretical_min 对 Cube 回退到 actual duration",
            )
            return duration
        rate = _model_compute_ops_per_cycle(op, db, warnings, warn_keys)
        return _startup_cycles_for_pipe(pipe, db, warnings, warn_keys) + (
            math.ceil(flops / rate) if rate > _EPSILON else duration
        )

    if pipe in _MODEL_FIXPIPE_PIPES:
        if bytes_transferred <= 0:
            return 0.0
        return _startup_cycles_for_pipe(pipe, db, warnings, warn_keys) + math.ceil(
            bytes_transferred / float(max(1, db.vector.vec_width_bytes))
        )

    if pipe in {"PIPE_MTE1", "MTE1"}:
        return duration

    return duration


def _model_compute_ops_per_cycle(
    op,
    db: CalibrationDB,
    warnings: list[str],
    warn_keys: set[str],
) -> float:
    precision = getattr(getattr(op, "precision", None), "value", "") or "fp16"
    if getattr(op, "pipe", "") in {"PIPE_M", "Cube"}:
        rate_us = _cube_peak_rate(precision, db)
    else:
        rate_us = _vector_peak_rate(precision, db)
    if rate_us <= _EPSILON:
        _warn_once(
            warnings,
            warn_keys,
            f"missing_compute_peak_{getattr(op, 'pipe', '')}_{precision}",
            f"缺少 {getattr(op, 'pipe', '')}/{precision} 的 compute peak；theoretical_min 可能回退",
        )
        return 0.0
    return rate_us / max(db.core.cycles_per_us, _EPSILON)


def _model_memory_bw_bytes_per_cycle(
    op,
    db: CalibrationDB,
    warnings: list[str],
    warn_keys: set[str],
) -> float:
    pipe = getattr(op, "pipe", "") or "Unknown"
    src = (getattr(op, "src_space", "") or "").lower()
    dst = (getattr(op, "dst_space", "") or "").lower()

    candidates = []
    if src and dst:
        candidates.append((src, dst))
    candidates.extend(_MEMORY_FALLBACK_PATHS.get(pipe, []))

    for candidate_src, candidate_dst in candidates:
        try:
            bw_us, _ = db.memory.lookup_bw(candidate_src, candidate_dst)
            if bw_us > _EPSILON:
                if (candidate_src, candidate_dst) != (src, dst):
                    _warn_once(
                        warnings,
                        warn_keys,
                        f"fallback_bw_{pipe}",
                        f"{pipe} 缺少 src_space/dst_space 或无对应带宽；使用 {candidate_src}->{candidate_dst} 带宽回退",
                    )
                return bw_us / max(db.core.cycles_per_us, _EPSILON)
        except KeyError:
            continue

    _warn_once(
        warnings,
        warn_keys,
        f"missing_bw_{pipe}",
        f"缺少 {pipe} 的内存带宽校准；使用 vector_width_bytes/cycle 回退",
    )
    return float(max(1, db.vector.vec_width_bytes))


def _startup_cycles_for_pipe(
    pipe: str,
    db: CalibrationDB,
    warnings: list[str],
    warn_keys: set[str],
) -> float:
    key = _PIPE_STARTUP_KEYS.get(pipe, "scalar")
    if key == "cxx_default":
        return 1.0
    value = db.startup_latency.get(key)
    if value is None and key == "mte1":
        value = db.startup_latency.get("mte2", 0.0) / 2.0
    if value is not None and value > 0:
        return float(value)

    fallback = _MODEL_STARTUP_FALLBACK.get(key, 1.0)
    _warn_once(
        warnings,
        warn_keys,
        f"startup_fallback_{key}",
        f"calibration 缺少 startup_latency['{key}']；HIVM bottleneck diagnosis 使用 {fallback:g} cycles 回退值",
    )
    return fallback


def _model_op_duration(
    op,
    des_metadata: dict[int, dict],
    warnings: list[str],
    warn_keys: set[str],
) -> float:
    duration = float(getattr(op, "duration_cycles", 0) or 0)
    if duration > _EPSILON:
        return duration
    op_id = int(getattr(op, "op_id", 0))
    meta = des_metadata.get(op_id, {})
    raw_duration = float(meta.get("duration", 0) or 0)
    if raw_duration > _EPSILON:
        return raw_duration
    start = float(getattr(op, "start_cycle", 0) or meta.get("start_cycle", 0) or 0)
    end = float(getattr(op, "end_cycle", 0) or meta.get("end_cycle", 0) or 0)
    if end > start:
        _warn_once(
            warnings,
            warn_keys,
            "duration_from_start_end",
            "部分 DES op 缺少 duration；已用 end_cycle-start_cycle 补齐",
        )
        return end - start
    return 0.0


def _model_op_end_cycle(
    op,
    des_metadata: dict[int, dict],
    warnings: list[str],
    warn_keys: set[str],
) -> float:
    end = float(getattr(op, "end_cycle", 0) or 0)
    if end > _EPSILON:
        return end
    op_id = int(getattr(op, "op_id", 0))
    meta = des_metadata.get(op_id, {})
    raw_end = float(meta.get("end_cycle", 0) or 0)
    if raw_end > _EPSILON:
        return raw_end
    start = float(getattr(op, "start_cycle", 0) or meta.get("start_cycle", 0) or 0)
    duration = _model_op_duration(op, des_metadata, warnings, warn_keys)
    return start + duration


def _model_op_is_sync(op, des_metadata: dict[int, dict]) -> bool:
    meta = des_metadata.get(int(getattr(op, "op_id", 0)), {})
    if "is_sync" in meta:
        return bool(meta["is_sync"])
    return getattr(op, "op_name", "") in _SYNC_OPS


def _model_op_is_barrier(op, des_metadata: dict[int, dict]) -> bool:
    meta = des_metadata.get(int(getattr(op, "op_id", 0)), {})
    if "is_barrier" in meta:
        return bool(meta["is_barrier"])
    return (
        getattr(op, "op_name", "") in {"pipe_barrier", "sync_block_wait"}
        or getattr(op, "pipe", "") in _ALL_PIPES
    )


def _cube_peak_rate(label: str, db: CalibrationDB) -> float:
    try:
        dtype = DType.from_str(label)
    except KeyError:
        return 0.0
    tflops = db.cube.throughput.get(dtype, 0.0)
    return tflops * 1e6 if tflops > 0 else 0.0


def _vector_peak_rate(label: str, db: CalibrationDB) -> float:
    try:
        dtype = DType.from_str(label)
    except KeyError:
        dtype = DType.FP16
    if dtype in (DType.FP16, DType.BF16):
        tflops = db.vector.throughput_fp16_tflops
    else:
        tflops = db.vector.throughput_fp32_tflops
    return tflops * 1e6 if tflops > 0 else 0.0


def _safe_ratio(num: float, den: float) -> float:
    return num / den if den > _EPSILON else 0.0


def _warn_once(
    warnings: list[str],
    warn_keys: set[str],
    key: str,
    message: str,
) -> None:
    if key in warn_keys:
        return
    warn_keys.add(key)
    warnings.append(message)
