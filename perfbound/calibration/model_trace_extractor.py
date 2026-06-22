#!/usr/bin/env python3
# model_trace_extractor.py — Extract model timeline from Perfetto trace JSON.
#
# Reads tritonsim-opt's Perfetto trace output and extracts per-component
# time allocations for comparison with real hardware profile data.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ModelOpTime:
    """One operation from the model trace."""
    component: str = ""
    duration_us: float = 0.0
    start_us: float = 0.0
    end_us: float = 0.0
    core_id: int = 0


@dataclass
class ModelTrace:
    """Model-generated trace extracted from Perfetto JSON."""
    kernel_name: str = ""
    trace_path: str = ""

    # Per-component aggregates
    aic_total_time_us: float = 0.0
    aiv_total_time_us: float = 0.0
    cube_time_us: float = 0.0
    vector_time_us: float = 0.0
    mte_time_us: float = 0.0

    # Per-component operation lists
    operations: List[ModelOpTime] = field(default_factory=list)

    # Pipeline metrics
    overlap_ratio: float = 1.0
    aic_aiv_concurrent_us: float = 0.0


def extract_model_trace(trace_path: str | Path) -> ModelTrace:
    """Extract model timeline from a Perfetto trace JSON.

    The JSON format follows the standard Perfetto schema produced by
    tritonsim-opt --perfetto-trace-file.  Each slice in the trace
    represents one scheduled operation on a hardware unit.

    Args:
        trace_path: Path to Perfetto trace JSON file.

    Returns:
        ModelTrace with per-component aggregates.
    """
    trace_path = Path(trace_path)

    with open(trace_path) as f:
        data = json.load(f)

    model = ModelTrace(
        kernel_name=trace_path.stem,
        trace_path=str(trace_path),
    )

    # Perfetto format: traceEvents array with ph=X, dur, ts, args
    events = data if isinstance(data, list) else data.get("traceEvents", [])

    aic_ops = []
    aiv_ops = []

    for ev in events:
        if not isinstance(ev, dict):
            continue
        # Only process complete events
        if ev.get("ph") not in ("X", "B"):
            continue

        name = ev.get("name", "")
        dur_us = float(ev.get("dur", 0))
        ts_us = float(ev.get("ts", 0))

        # Perfetto ts is in microseconds by default in tritonsim-opt output
        op = ModelOpTime(
            component=name,
            duration_us=dur_us,
            start_us=ts_us,
            end_us=ts_us + dur_us,
        )
        model.operations.append(op)

        # Classify by component name prefix
        name_lower = name.lower()
        if any(k in name_lower for k in ("cube", "matmul", "aic", "fixpipe", "mte2", "l0a", "l0b", "l1")):
            aic_ops.append(op)
            model.aic_total_time_us += dur_us
            if "cube" in name_lower or "matmul" in name_lower:
                model.cube_time_us += dur_us
            elif "mte" in name_lower or "fixpipe" in name_lower or "l1" in name_lower:
                model.mte_time_us += dur_us
        elif any(k in name_lower for k in ("vector", "aiv", "mte3", "ub", "scalar")):
            aiv_ops.append(op)
            model.aiv_total_time_us += dur_us
            if "vector" in name_lower or "scalar" in name_lower:
                model.vector_time_us += dur_us
            elif "mte" in name_lower:
                model.mte_time_us += dur_us
        else:
            # Default: assign to AIC
            aic_ops.append(op)
            model.aic_total_time_us += dur_us

    # Compute model overlap ratio from interleaved ops
    model.overlap_ratio = _compute_model_overlap(aic_ops, aiv_ops, model)

    return model


def _compute_model_overlap(
    aic_ops: List[ModelOpTime],
    aiv_ops: List[ModelOpTime],
    model: ModelTrace,
) -> float:
    """Compute AIC↔AIV time overlap ratio from model trace."""
    aic_total = sum(op.duration_us for op in aic_ops)
    aiv_total = sum(op.duration_us for op in aiv_ops)

    if aic_total <= 0 or aiv_total <= 0:
        return 1.0

    # Sort by start time
    aic_sorted = sorted(aic_ops, key=lambda o: o.start_us)
    aiv_sorted = sorted(aiv_ops, key=lambda o: o.start_us)

    concurrent = 0.0
    for aic_op in aic_sorted:
        for aiv_op in aiv_sorted:
            ov_s = max(aic_op.start_us, aiv_op.start_us)
            ov_e = min(aic_op.end_us, aiv_op.end_us)
            if ov_s < ov_e:
                concurrent += ov_e - ov_s

    model.aic_aiv_concurrent_us = concurrent
    max_possible = min(aic_total, aiv_total)
    return concurrent / max_possible if max_possible > 0 else 1.0
