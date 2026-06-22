#!/usr/bin/env python3
# msprof_trace_extractor.py — Reconstruct real hardware timeline from msprof CSV.
#
# Reads msprof op_summary.csv and rebuilds per-core timelines, aggregating
# total active time by task type (AIC / AIV) for comparison with the model.
# Also computes pipeline overlap ratios from concurrent task windows.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .scripts.fit_constants import MSProfRow, read_msprof_csv, compute_mean_ci


AICORE_TASK_TYPES = frozenset(
    {"AI_CORE", "AICORE", "MIX_AIC", "MIX_AIV", "AI_VECTOR_CORE", "AIV"}
)

AIC_ONLY_TYPES = frozenset({"AI_CORE", "AICORE"})
AIV_ONLY_TYPES = frozenset({"AI_VECTOR_CORE", "AIV"})


@dataclass
class CoreTimeline:
    """Per-core timeline reconstructed from msprof rows."""
    core_id: int = 0
    task_type: str = ""
    intervals: List[Tuple[float, float]] = field(default_factory=list)
    total_active_us: float = 0.0
    fixpipe_active_us: float = 0.0
    scalar_active_us: float = 0.0
    n_invocations: int = 0


@dataclass
class RealTrace:
    """Real hardware trace extracted from msprof CSV."""
    kernel_name: str = ""
    csv_path: str = ""
    n_cores: int = 0
    block_dim: int = 0

    # Per-task-type aggregates
    aic_total_time_us: float = 0.0
    aiv_total_time_us: float = 0.0
    aic_fixpipe_time_us: float = 0.0
    aiv_scalar_time_us: float = 0.0

    # Per-core timelines
    core_timelines: Dict[int, CoreTimeline] = field(default_factory=dict)

    # Pipeline overlap metrics
    overlap_ratio: float = 0.0
    aic_aiv_concurrent_us: float = 0.0

    # Raw durations
    aic_durations_us: List[float] = field(default_factory=list)
    aiv_durations_us: List[float] = field(default_factory=list)

    @property
    def total_wall_time_us(self) -> float:
        if not self.core_timelines:
            return 0.0
        end_times = []
        for tl in self.core_timelines.values():
            if tl.intervals:
                end_times.append(max(e for _, e in tl.intervals))
        return max(end_times) if end_times else 0.0


def extract_real_trace(csv_path: str | Path) -> RealTrace:
    """Extract real hardware trace from an msprof op_summary.csv.

    Rebuilds per-core timelines using Task Start Time(us) and
    Task Duration(us) columns. Aggregates total active times per
    task type and computes pipeline overlap metrics.

    Args:
        csv_path: Path to msprof op_summary CSV.

    Returns:
        RealTrace with per-core timelines and aggregate metrics.
    """
    csv_path = Path(csv_path)
    rows = read_msprof_csv(csv_path)

    trace = RealTrace(
        kernel_name=csv_path.stem,
        csv_path=str(csv_path),
    )

    core_timelines: Dict[int, List[MSProfRow]] = {}
    for row in rows:
        tid = row.task_id
        core_timelines.setdefault(tid, []).append(row)

    trace.n_cores = len(core_timelines)
    trace.block_dim = rows[0].block_dim if rows else 0

    for core_id, core_rows in core_timelines.items():
        tl = CoreTimeline(core_id=core_id)
        tl.n_invocations = len(core_rows)

        for row in core_rows:
            start = row.start_time_us
            dur = row.duration_us
            if start > 0 and dur > 0:
                tl.intervals.append((start, start + dur))
                tl.total_active_us += dur
            tl.fixpipe_active_us += row.fixpipe_time_us
            tl.scalar_active_us += row.aiv_scalar_time_us

        if core_rows:
            tt = core_rows[0].task_type.strip().upper()
            tl.task_type = tt

            if tt in AIC_ONLY_TYPES or tt in ("MIX_AIC",):
                trace.aic_total_time_us += tl.total_active_us
                trace.aic_fixpipe_time_us += tl.fixpipe_active_us
                trace.aic_durations_us.extend(
                    r.duration_us for r in core_rows if r.duration_us > 0
                )
            elif tt in AIV_ONLY_TYPES or tt in ("MIX_AIV",):
                trace.aiv_total_time_us += tl.total_active_us
                trace.aiv_scalar_time_us += tl.scalar_active_us
                trace.aiv_durations_us.extend(
                    r.duration_us for r in core_rows if r.duration_us > 0
                )

        trace.core_timelines[core_id] = tl

    # Compute pipeline overlap ratio
    aic_core_ids = [cid for cid, tl in trace.core_timelines.items()
                    if tl.task_type in AIC_ONLY_TYPES
                    or tl.task_type in ("MIX_AIC",)]
    aiv_core_ids = [cid for cid, tl in trace.core_timelines.items()
                    if tl.task_type in AIV_ONLY_TYPES
                    or tl.task_type in ("MIX_AIV",)]

    if aic_core_ids and aiv_core_ids:
        trace.aic_aiv_concurrent_us, trace.overlap_ratio = _compute_concurrency(
            trace.core_timelines, aic_core_ids, aiv_core_ids
        )

    return trace


def _compute_concurrency(
    timelines: Dict[int, CoreTimeline],
    aic_ids: List[int],
    aiv_ids: List[int],
) -> Tuple[float, float]:
    """Compute AIC↔AIV concurrency from per-core interval data.

    Returns (concurrent_time_us, overlap_ratio).
    """
    # Collect all intervals from AIC and AIV cores
    aic_intervals = []
    for cid in aic_ids:
        aic_intervals.extend(timelines[cid].intervals)
    aiv_intervals = []
    for cid in aiv_ids:
        aiv_intervals.extend(timelines[cid].intervals)

    if not aic_intervals or not aiv_intervals:
        return 0.0, 0.0

    # Merge intervals from each group
    aic_merged = _merge_intervals(aic_intervals)
    aiv_merged = _merge_intervals(aiv_intervals)

    aic_total = sum(e - s for s, e in aic_merged)
    aiv_total = sum(e - s for s, e in aiv_merged)

    if aic_total <= 0 or aiv_total <= 0:
        return 0.0, 0.0

    # Compute intersection
    concurrent = 0.0
    for a_s, a_e in aic_merged:
        for v_s, v_e in aiv_merged:
            ov_start = max(a_s, v_s)
            ov_end = min(a_e, v_e)
            if ov_start < ov_end:
                concurrent += ov_end - ov_start

    max_possible = min(aic_total, aiv_total)
    overlap_ratio = concurrent / max_possible if max_possible > 0 else 0.0
    return concurrent, overlap_ratio


def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping intervals."""
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_iv[0]]
    for s, e in sorted_iv[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged
