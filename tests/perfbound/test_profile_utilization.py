"""Tests for paper-style profiling bottleneck analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[2]))

from perfbound.calibration.constants import CalibrationDB
from perfbound.extract.op_classifier import Component, Precision
from perfbound.model.component_model import ComponentBound
from perfbound.analyze.hivm_bottleneck_diagnosis import (
    diagnose_hivm_bottleneck_from_des_ops,
)
from perfbound.analyze.profile_utilization import (
    KernelProfileStats,
    ProfileComponentStats,
    WorkBreakdownItem,
    analyze_operator_bottleneck,
    compute_realized_utilization,
)


TOL = 1e-9


def _ideal_rate(items: list[WorkBreakdownItem]) -> float:
    """按论文公式计算 operator-aware ideal performance。"""
    total_work = sum(item.work for item in items)
    weighted_time = sum(item.work / item.peak_rate for item in items)
    return total_work / weighted_time


def _elapsed_for_u(work: float, ideal_rate: float, utilization: float) -> float:
    """由目标 U 反推 elapsed，便于构造可解释的测试数据。"""
    return work / (utilization * ideal_rate)


def _active_for_e(work: float, ideal_rate: float, efficiency: float) -> float:
    """由目标 E 反推 active_time，保证 E 不超过 1。"""
    return work / (efficiency * ideal_rate)


def _assert_close(actual: float, expected: float) -> None:
    assert abs(actual - expected) < TOL


def _assert_no_warnings(report) -> None:
    assert report.warnings == []
    for result in report.component_results.values():
        assert result.warnings == []


def _bound_for_profile(profile: KernelProfileStats) -> ComponentBound:
    """构造和 profile work 对齐的 ComponentBound，避免正常路径误报 warning。"""
    total_ops: dict[str, float] = {}
    total_bytes: dict[str, float] = {}
    per_component_us: dict[str, float] = {}

    for stats in profile.components:
        comp_key = stats.component.value
        if stats.component in (Component.CUBE, Component.VECTOR, Component.SCALAR):
            total_ops[comp_key] = stats.work_done
        else:
            total_bytes[comp_key] = stats.work_done

        if stats.work_breakdown:
            ideal = _ideal_rate(stats.work_breakdown)
            per_component_us[comp_key] = stats.work_done / ideal
        else:
            per_component_us[comp_key] = stats.work_done

    binding_key, t_core_floor = max(
        per_component_us.items(),
        key=lambda item: item[1],
    )
    binding_component = Component(binding_key)

    return ComponentBound(
        t_core_floor_us=t_core_floor,
        binding_component=binding_component,
        total_ops=total_ops,
        total_bytes=total_bytes,
        per_component_us=per_component_us,
    )


def _component_bound(
    *,
    total_ops: dict[str, float] | None = None,
    total_bytes: dict[str, float] | None = None,
    per_component_us: dict[str, float] | None = None,
) -> ComponentBound:
    """用于 warning 测试的手写 ComponentBound。"""
    per_component_us = per_component_us or {"cube": 10.0}
    binding_key, t_core_floor = max(
        per_component_us.items(),
        key=lambda item: item[1],
    )
    return ComponentBound(
        t_core_floor_us=t_core_floor,
        binding_component=Component(binding_key),
        total_ops=total_ops or {},
        total_bytes=total_bytes or {},
        per_component_us=per_component_us,
    )


def test_operator_aware_ideal_performance_uses_precision_breakdown():
    """I = sum(ops) / sum(ops / peak)，并能找出主导精度。"""
    items = [
        WorkBreakdownItem("fp16", work=1234.5, peak_rate=157.25),
        WorkBreakdownItem("bf16", work=789.25, peak_rate=143.5),
        WorkBreakdownItem("int8", work=456.75, peak_rate=311.125),
    ]
    work = sum(item.work for item in items)
    ideal = _ideal_rate(items)
    elapsed = _elapsed_for_u(work, ideal, utilization=0.62)
    active = _active_for_e(work, ideal, efficiency=0.775)

    profile = KernelProfileStats(
        kernel_name="mixed_cube",
        elapsed_time_us=elapsed,
        components=[
            ProfileComponentStats(
                component=Component.CUBE,
                work_done=work,
                active_time_us=active,
                work_breakdown=items,
            ),
        ],
    )

    report = compute_realized_utilization(profile, _bound_for_profile(profile))
    cube = report.component_results["cube"]

    _assert_no_warnings(report)
    _assert_close(cube.ideal_performance, ideal)
    _assert_close(cube.actual_performance, work / elapsed)
    _assert_close(cube.u_utilization, 0.62)
    _assert_close(cube.r_residency, active / elapsed)
    _assert_close(cube.e_efficiency, 0.775)
    assert cube.dominant_item == "fp16"
    _assert_close(cube.dominant_share, 1234.5 / work)


def test_component_metrics_match_hand_calculation_for_compute_and_mte():
    """同时验证 Compute 和 MTE 的 A/I/U/R/E 数值。"""
    cube_items = [
        WorkBreakdownItem("fp16", work=1379.25, peak_rate=211.5),
        WorkBreakdownItem("bf16", work=842.75, peak_rate=188.25),
        WorkBreakdownItem("fp32", work=317.5, peak_rate=72.125),
        WorkBreakdownItem("int8", work=529.25, peak_rate=401.75),
    ]
    mte_items = [
        WorkBreakdownItem("gm->ub", work=4096.5, peak_rate=236.75),
        WorkBreakdownItem("gm->l1", work=1536.25, peak_rate=184.5),
        WorkBreakdownItem("l1->ub", work=768.75, peak_rate=128.25),
    ]
    cube_work = sum(item.work for item in cube_items)
    mte_work = sum(item.work for item in mte_items)
    cube_ideal = _ideal_rate(cube_items)
    mte_ideal = _ideal_rate(mte_items)

    # 两个 component 共用同一个 operator elapsed_time。
    elapsed = max(
        _elapsed_for_u(cube_work, cube_ideal, utilization=0.51),
        _elapsed_for_u(mte_work, mte_ideal, utilization=0.37),
    )
    cube_active = _active_for_e(cube_work, cube_ideal, efficiency=0.68)
    mte_active = _active_for_e(mte_work, mte_ideal, efficiency=0.74)

    profile = KernelProfileStats(
        kernel_name="numeric_check",
        elapsed_time_us=elapsed,
        components=[
            ProfileComponentStats(Component.CUBE, cube_work, cube_active, cube_items),
            ProfileComponentStats(Component.MTE_GM, mte_work, mte_active, mte_items),
        ],
    )

    report = compute_realized_utilization(profile, _bound_for_profile(profile))
    cube = report.component_results["cube"]
    mte = report.component_results["mte_gm"]

    _assert_no_warnings(report)

    _assert_close(cube.ideal_performance, cube_ideal)
    _assert_close(cube.actual_performance, cube_work / elapsed)
    _assert_close(cube.u_utilization, (cube_work / elapsed) / cube_ideal)
    _assert_close(cube.r_residency, cube_active / elapsed)
    _assert_close(cube.e_efficiency, cube.u_utilization / cube.r_residency)

    _assert_close(mte.ideal_performance, mte_ideal)
    _assert_close(mte.actual_performance, mte_work / elapsed)
    _assert_close(mte.u_utilization, (mte_work / elapsed) / mte_ideal)
    _assert_close(mte.r_residency, mte_active / elapsed)
    _assert_close(mte.e_efficiency, mte.u_utilization / mte.r_residency)


def test_compute_bound_when_compute_utilization_reaches_threshold():
    """Cube 的 U 达到阈值时，诊断为 Compute Bound。"""
    cube_items = [
        WorkBreakdownItem("fp16", 1640.5, 202.25),
        WorkBreakdownItem("bf16", 915.25, 176.5),
        WorkBreakdownItem("int8", 608.75, 392.125),
    ]
    mte_items = [
        WorkBreakdownItem("gm->ub", 3880.5, 980.0),
        WorkBreakdownItem("gm->l1", 1412.25, 760.75),
    ]
    cube_work = sum(item.work for item in cube_items)
    mte_work = sum(item.work for item in mte_items)
    cube_ideal = _ideal_rate(cube_items)
    mte_ideal = _ideal_rate(mte_items)
    elapsed = _elapsed_for_u(cube_work, cube_ideal, utilization=0.83)

    profile = KernelProfileStats(
        kernel_name="compute_bound",
        elapsed_time_us=elapsed,
        components=[
            ProfileComponentStats(
                Component.CUBE,
                cube_work,
                _active_for_e(cube_work, cube_ideal, efficiency=0.92),
                cube_items,
            ),
            ProfileComponentStats(
                Component.MTE_GM,
                mte_work,
                _active_for_e(mte_work, mte_ideal, efficiency=0.80),
                mte_items,
            ),
        ],
    )

    report = analyze_operator_bottleneck(
        profile,
        _bound_for_profile(profile),
        u_threshold=0.80,
    )

    _assert_no_warnings(report)
    assert report.diagnosis == "Compute Bound"
    assert report.bound_kind == "Compute Bound"
    assert report.dominant_component == Component.CUBE
    assert report.dominant_item == "fp16"
    _assert_close(report.component_results["cube"].u_utilization, 0.83)


def test_mte_bound_when_mte_utilization_reaches_threshold():
    """MTE 的 U 达到阈值时，诊断为 MTE Bound。"""
    cube_items = [
        WorkBreakdownItem("fp16", 511.25, 196.0),
        WorkBreakdownItem("bf16", 377.5, 162.0),
        WorkBreakdownItem("fp32", 128.75, 71.0),
    ]
    mte_items = [
        WorkBreakdownItem("gm->ub", 4333.5, 224.25),
        WorkBreakdownItem("gm->l1", 2331.25, 172.75),
        WorkBreakdownItem("ub->gm", 1188.75, 119.5),
    ]
    cube_work = sum(item.work for item in cube_items)
    mte_work = sum(item.work for item in mte_items)
    cube_ideal = _ideal_rate(cube_items)
    mte_ideal = _ideal_rate(mte_items)
    elapsed = _elapsed_for_u(mte_work, mte_ideal, utilization=0.82)

    profile = KernelProfileStats(
        kernel_name="mte_bound",
        elapsed_time_us=elapsed,
        components=[
            ProfileComponentStats(
                Component.CUBE,
                cube_work,
                _active_for_e(cube_work, cube_ideal, efficiency=0.76),
                cube_items,
            ),
            ProfileComponentStats(
                Component.MTE_GM,
                mte_work,
                _active_for_e(mte_work, mte_ideal, efficiency=0.94),
                mte_items,
            ),
        ],
    )

    report = analyze_operator_bottleneck(
        profile,
        _bound_for_profile(profile),
        u_threshold=0.80,
    )

    _assert_no_warnings(report)
    assert report.diagnosis == "MTE Bound"
    assert report.bound_kind == "MTE Bound"
    assert report.dominant_component == Component.MTE_GM
    assert report.dominant_item == "gm->ub"
    _assert_close(report.dominant_share, 4333.5 / mte_work)
    _assert_close(report.component_results["mte_gm"].u_utilization, 0.82)


def test_insufficient_parallelism_when_all_active_ratios_are_low():
    """所有有效 component 的 R 都低时，诊断为并行不足。"""
    cube_items = [
        WorkBreakdownItem("fp16", 922.5, 201.5),
        WorkBreakdownItem("bf16", 431.25, 171.25),
        WorkBreakdownItem("int8", 277.75, 384.5),
    ]
    mte_items = [
        WorkBreakdownItem("gm->ub", 2850.5, 226.25),
        WorkBreakdownItem("gm->l1", 1164.25, 166.5),
        WorkBreakdownItem("ub->gm", 708.75, 117.25),
    ]
    cube_work = sum(item.work for item in cube_items)
    mte_work = sum(item.work for item in mte_items)
    cube_ideal = _ideal_rate(cube_items)
    mte_ideal = _ideal_rate(mte_items)
    cube_elapsed = _elapsed_for_u(cube_work, cube_ideal, utilization=0.18)
    mte_elapsed = _elapsed_for_u(mte_work, mte_ideal, utilization=0.22)
    elapsed = max(cube_elapsed, mte_elapsed)

    profile = KernelProfileStats(
        kernel_name="low_parallelism",
        elapsed_time_us=elapsed,
        components=[
            ProfileComponentStats(
                Component.CUBE,
                cube_work,
                elapsed * 0.30,
                cube_items,
            ),
            ProfileComponentStats(
                Component.MTE_GM,
                mte_work,
                elapsed * 0.40,
                mte_items,
            ),
        ],
    )

    report = analyze_operator_bottleneck(
        profile,
        _bound_for_profile(profile),
        u_threshold=0.80,
        r_threshold=0.50,
    )

    _assert_no_warnings(report)
    assert report.diagnosis == "Insufficient Parallelism"
    assert report.dominant_component is None
    assert report.dominant_item is None
    _assert_close(report.dominant_share, 0.0)
    assert report.component_results["cube"].r_residency < 0.50
    assert report.component_results["mte_gm"].r_residency < 0.50


def test_inefficient_compute_when_compute_has_high_r_and_low_e():
    """R 高但 E 低的 Vector component，诊断为 Inefficient Compute。"""
    vector_items = [
        WorkBreakdownItem("fp32", 731.5, 82.75),
        WorkBreakdownItem("fp16", 618.25, 166.5),
        WorkBreakdownItem("int8", 294.75, 332.25),
    ]
    mte_items = [
        WorkBreakdownItem("gm->ub", 1101.5, 221.25),
        WorkBreakdownItem("ub->gm", 553.75, 116.0),
    ]
    vector_work = sum(item.work for item in vector_items)
    mte_work = sum(item.work for item in mte_items)
    vector_ideal = _ideal_rate(vector_items)
    mte_ideal = _ideal_rate(mte_items)
    elapsed = _elapsed_for_u(vector_work, vector_ideal, utilization=0.24)

    profile = KernelProfileStats(
        kernel_name="inefficient_compute",
        elapsed_time_us=elapsed,
        components=[
            ProfileComponentStats(
                Component.VECTOR,
                vector_work,
                elapsed * 0.78,
                vector_items,
            ),
            ProfileComponentStats(
                Component.MTE_GM,
                mte_work,
                _active_for_e(mte_work, mte_ideal, efficiency=0.64),
                mte_items,
            ),
        ],
    )

    report = analyze_operator_bottleneck(
        profile,
        _bound_for_profile(profile),
        u_threshold=0.80,
        r_threshold=0.50,
    )
    vector = report.component_results["vector"]

    _assert_no_warnings(report)
    assert report.diagnosis == "Inefficient Compute"
    assert report.dominant_component == Component.VECTOR
    assert report.dominant_item == "fp32"
    _assert_close(vector.u_utilization, 0.24)
    _assert_close(vector.r_residency, 0.78)
    _assert_close(vector.e_efficiency, 0.24 / 0.78)


def test_inefficient_mte_when_mte_has_high_r_and_low_e():
    """R 高但 E 低的 MTE component，诊断为 Inefficient MTE。"""
    cube_items = [
        WorkBreakdownItem("fp16", 402.5, 188.5),
        WorkBreakdownItem("bf16", 318.25, 154.75),
        WorkBreakdownItem("int8", 144.75, 367.25),
    ]
    mte_items = [
        WorkBreakdownItem("ub->gm", 2524.5, 118.75),
        WorkBreakdownItem("gm->ub", 1046.25, 214.5),
        WorkBreakdownItem("l1->ub", 682.75, 136.25),
    ]
    cube_work = sum(item.work for item in cube_items)
    mte_work = sum(item.work for item in mte_items)
    cube_ideal = _ideal_rate(cube_items)
    mte_ideal = _ideal_rate(mte_items)
    elapsed = _elapsed_for_u(mte_work, mte_ideal, utilization=0.25)

    profile = KernelProfileStats(
        kernel_name="inefficient_mte",
        elapsed_time_us=elapsed,
        components=[
            ProfileComponentStats(
                Component.CUBE,
                cube_work,
                _active_for_e(cube_work, cube_ideal, efficiency=0.58),
                cube_items,
            ),
            ProfileComponentStats(
                Component.MTE_UB,
                mte_work,
                elapsed * 0.80,
                mte_items,
            ),
        ],
    )

    report = analyze_operator_bottleneck(
        profile,
        _bound_for_profile(profile),
        u_threshold=0.80,
        r_threshold=0.50,
    )
    mte = report.component_results["mte_ub"]

    _assert_no_warnings(report)
    assert report.diagnosis == "Inefficient MTE"
    assert report.dominant_component == Component.MTE_UB
    assert report.dominant_item == "ub->gm"
    _assert_close(mte.u_utilization, 0.25)
    _assert_close(mte.r_residency, 0.80)
    _assert_close(mte.e_efficiency, 0.25 / 0.80)


def test_warning_when_work_breakdown_is_missing_and_fallback_is_used():
    """warning 单独测试：没有 breakdown 时会回退到 ComponentBound。"""
    profile = KernelProfileStats(
        kernel_name="missing_breakdown",
        elapsed_time_us=15.0,
        components=[
            ProfileComponentStats(
                component=Component.CUBE,
                work_done=1200.0,
                active_time_us=12.0,
                work_breakdown=[],
            ),
        ],
    )

    report = compute_realized_utilization(
        profile,
        _component_bound(
            total_ops={"cube": 1200.0},
            per_component_us={"cube": 10.0},
        ),
    )

    assert any("未提供 operator-aware work_breakdown" in item for item in report.warnings)
    _assert_close(report.component_results["cube"].ideal_performance, 120.0)


def test_warning_when_active_time_is_larger_than_elapsed_time():
    """warning 单独测试：active_time 不应超过 operator 总时间。"""
    items = [WorkBreakdownItem("fp16", 500.0, 100.0)]
    profile = KernelProfileStats(
        kernel_name="bad_active_time",
        elapsed_time_us=10.0,
        components=[
            ProfileComponentStats(Component.CUBE, 500.0, 11.25, items),
        ],
    )

    report = compute_realized_utilization(profile, _bound_for_profile(profile))

    assert any("active_time_us 大于 elapsed_time_us" in item for item in report.warnings)


def test_warning_when_profile_work_mismatches_component_bound_work():
    """warning 单独测试：profiling work 和理论 work 差距过大。"""
    items = [WorkBreakdownItem("fp16", 1250.0, 100.0)]
    profile = KernelProfileStats(
        kernel_name="work_mismatch",
        elapsed_time_us=20.0,
        components=[
            ProfileComponentStats(Component.CUBE, 1250.0, 15.0, items),
        ],
    )

    report = compute_realized_utilization(
        profile,
        _component_bound(
            total_ops={"cube": 1000.0},
            per_component_us={"cube": 10.0},
        ),
        work_tolerance=0.10,
    )

    assert any("profiling work 和理论 bound work 相差" in item for item in report.warnings)


def test_warning_when_u_or_e_is_unphysical():
    """warning 单独测试：U/E 超过合理范围通常说明单位或峰值来源有问题。"""
    items = [WorkBreakdownItem("fp16", 1000.0, 100.0)]
    profile = KernelProfileStats(
        kernel_name="unphysical_efficiency",
        elapsed_time_us=8.0,
        components=[
            ProfileComponentStats(Component.CUBE, 1000.0, 4.0, items),
        ],
    )

    report = compute_realized_utilization(profile, _bound_for_profile(profile))

    assert any("U > 1.05" in item for item in report.warnings)
    assert any("E > 1.05" in item for item in report.warnings)


def test_hivm_bottleneck_diagnosis_flags_sync_overhead_from_des_ops():
    """HIVM Bottleneck Diagnosis：sync/barrier 占比超过阈值时判定 SyncOverhead。"""
    ops = [
        SimpleNamespace(
            op_id=0,
            op_name="load",
            pipe="PIPE_MTE2_V",
            bytes_transferred=0,
            elements=0,
            flops=0,
            duration_cycles=70,
            loop_multiplier=1,
            start_cycle=0,
            end_cycle=70,
            src_space="gm",
            dst_space="ub",
            precision=Precision.FP16,
        ),
        SimpleNamespace(
            op_id=1,
            op_name="pipe_barrier",
            pipe="PIPE_ALL",
            bytes_transferred=0,
            elements=0,
            flops=0,
            duration_cycles=30,
            loop_multiplier=1,
            start_cycle=70,
            end_cycle=100,
            src_space="",
            dst_space="",
            precision=Precision.FP16,
        ),
    ]
    des_metadata = {
        0: {
            "id": 0,
            "name": "load",
            "pipe": "PIPE_MTE2_V",
            "duration": 70,
            "start_cycle": 0,
            "end_cycle": 70,
            "depends_on": [],
            "is_sync": False,
            "is_barrier": False,
            "bytes": 0,
            "elements": 0,
            "flops": 0,
            "loop_multiplier": 1,
            "src_space": "gm",
            "dst_space": "ub",
            "elem_type": "fp16",
            "repeat": 1,
            "mask": 128,
            "line": 201,
            "core_type": "aiv",
            "event_id": 0,
        },
        1: {
            "id": 1,
            "name": "pipe_barrier",
            "pipe": "PIPE_ALL",
            "duration": 30,
            "start_cycle": 70,
            "end_cycle": 100,
            "depends_on": [0],
            "is_sync": True,
            "is_barrier": True,
            "bytes": 0,
            "elements": 0,
            "flops": 0,
            "loop_multiplier": 1,
            "src_space": "",
            "dst_space": "",
            "elem_type": "fp16",
            "repeat": 1,
            "mask": 0,
            "line": 202,
            "core_type": "aiv",
            "event_id": 1,
        },
    }

    report = diagnose_hivm_bottleneck_from_des_ops(
        ops,
        CalibrationDB(),
        des_metadata=des_metadata,
    )

    assert report.global_root_cause == "SyncOverhead"
    assert report.sync_overhead_ratio == 30.0
    assert report.barrier_overhead_ratio == 30.0
    assert not any("未提供原始 DES metadata" in warning for warning in report.warnings)


def test_hivm_bottleneck_warns_when_raw_des_metadata_is_missing():
    """直接消费完整 OpRecord 时，line/is_sync/is_barrier 来源仍不可见。"""
    ops = [
        SimpleNamespace(
            op_id=0,
            op_name="vadd",
            pipe="PIPE_V",
            bytes_transferred=0,
            elements=128,
            flops=0,
            duration_cycles=40,
            loop_multiplier=1,
            start_cycle=0,
            end_cycle=40,
            src_space="ub",
            dst_space="ub",
            precision=Precision.FP16,
        )
    ]

    report = diagnose_hivm_bottleneck_from_des_ops(ops, CalibrationDB())

    assert any("未提供原始 DES metadata" in warning for warning in report.warnings)


# ── Real-kernel coverage: chunk_kda end-to-end through run_from_files ──────────
#
# Synthetic tests above never exercise run_from_files on a real msprof export.
# These pin the post-merge fixes:
#   - exposed control/sync (high-R Scalar with zero arithmetic work) is routed to
#     the paper's "Insufficient Parallelism" verdict AND localized to scalar
#     (the paper's taxonomy would mis-route it to "Inefficient Component");
#   - the default kernel_name=None path tolerates 'N/A' cells (no crash);
#   - multi-shard kernels reduce to the longest-duration (critical) row.

import pytest

from perfbound.analyze.profile_utilization import (
    emit_perfetto_trace_from_des,
    run_from_files,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_CHUNK_KDA_CSV = _FIXTURES / "chunk_kda_op_summary_910b3.csv"
_KDA_DES = Path(__file__).parents[2] / ".omc" / "research" / "hw_runs" / "kda_des.json"


def test_perfetto_trace_is_emitted_from_des_graph(tmp_path):
    """Python path should emit a Perfetto trace from the same DES graph input."""
    des_path = tmp_path / "des.json"
    trace_path = tmp_path / "perfetto_trace.json"
    des_path.write_text(
        json.dumps(
            {
                "clock_ghz": 1.85,
                "schedule_truncated": False,
                "operations": [
                    {
                        "id": 0,
                        "name": "fake_matmul_compute_bound",
                        "pipe": "Cube",
                        "duration": 200,
                        "start_cycle": 0,
                        "end_cycle": 200,
                        "line": 1,
                        "depends_on": [],
                        "is_sync": False,
                        "is_barrier": False,
                        "event_id": "",
                        "event_generation": 0,
                        "sender_pipe": "Unknown",
                        "receiver_pipe": "Unknown",
                        "core_type": "AIC",
                        "bytes": 0,
                        "elements": 1000000,
                        "flops": 1000000,
                        "loop_multiplier": 1,
                        "read_buffers": [],
                        "write_buffers": [],
                        "read_versions": [],
                        "write_versions": [],
                        "src_space": "l0a",
                        "dst_space": "l0c",
                        "elem_type": "bf16",
                    }
                ],
            }
        )
        + "\n"
    )
    emit_perfetto_trace_from_des(des_path, trace_path)

    trace = json.loads(trace_path.read_text())
    assert trace["displayTimeUnit"] == "us"
    events = trace["traceEvents"]
    assert any(
        event.get("ph") == "M"
        and event.get("name") == "process_name"
        and event.get("args", {}).get("name") == "AIC (Cube Core)"
        for event in events
    )
    op_events = [event for event in events if event.get("ph") == "X"]
    assert len(op_events) == 1
    assert op_events[0]["name"] == "fake_matmul_compute_bound"
    assert op_events[0]["pid"] == 1
    assert op_events[0]["tid"] == 1
    assert op_events[0]["args"]["cycles"] == 200


@pytest.mark.skipif(not _KDA_DES.exists(), reason="real kda_des.json not present")
def test_chunk_kda_exposed_control_is_insufficient_parallelism():
    """Real chunk_kda: exposed scalar control/sync → Insufficient Parallelism @ scalar."""
    report = run_from_files(
        _CHUNK_KDA_CSV, _KDA_DES, None,
        kernel_name="chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2",
    )
    # Paper-faithful verdict, now localized to the exposed control engine.
    assert report.diagnosis == "Insufficient Parallelism"
    assert report.dominant_component == Component.SCALAR
    # Scalar monopolizes the timeline (R≈0.92) but carries zero arithmetic work.
    scalar = report.component_results["scalar"]
    assert scalar.r_residency > 0.8
    assert scalar.work_done == 0.0
    # The exposed-control rationale is surfaced (not a generic parallelism note).
    assert any("exposed control/sync" in w for w in report.warnings)
    # Multi-shard reduction picks the longest-duration row (~104,326 µs), not the
    # first shard (~104,316 µs).
    assert report.elapsed_time_us > 104_320.0
    # Quantified exposed-control/sync deficit (addition beyond the paper):
    # model exposes ~11.9% control on the DES critical path, hardware exposes
    # ~84.6% same-core scalar -> deficit ≈ +72.7 pts. Scale-invariant, no bound.
    assert 0.10 < report.exposed_control_frac_model < 0.15
    assert 0.83 < report.exposed_control_frac_measured < 0.86
    assert 0.70 < report.exposed_control_deficit_pts < 0.75
    assert report.n_sync_ops == 402
    # deficit_us needs a sound (loop-scaled two-tier) bound; None without one.
    assert report.exposed_control_deficit_us is None


@pytest.mark.skipif(not _KDA_DES.exists(), reason="real kda_des.json not present")
def test_chunk_kda_exposed_control_deficit_us_capped_at_headroom():
    """With the sound two-tier bound supplied, deficit_us ≈ 58,216 µs (capped)."""
    report = run_from_files(
        _CHUNK_KDA_CSV, _KDA_DES, None,
        kernel_name="chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2",
        t_bound_us=46_109.91,  # A.8 loop-scaled two-tier bound
    )
    assert report.exposed_control_deficit_us is not None
    author_headroom = report.elapsed_time_us - 46_109.91
    # Capped at author headroom, and matches the A.8 Gap-OVL magnitude.
    assert report.exposed_control_deficit_us <= author_headroom + 1.0
    assert 57_000.0 < report.exposed_control_deficit_us < 59_000.0


@pytest.mark.skipif(not _KDA_DES.exists(), reason="real kda_des.json not present")
def test_chunk_kda_default_kernel_name_does_not_crash_on_na_cells():
    """kernel_name=None must tolerate 'N/A' summary cells (regression for the crash)."""
    report = run_from_files(_CHUNK_KDA_CSV, _KDA_DES, None, kernel_name=None)
    assert report.diagnosis == "Insufficient Parallelism"
    assert report.dominant_component == Component.SCALAR
