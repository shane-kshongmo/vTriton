"""Compare tritonsim Perfetto traces with ``msprof op simulator`` JSON.

Both formats use Perfetto complete events, but model pipes are identified by
thread metadata while simulator pipes are identified by process metadata.
This module normalizes both schemas and compares wall span plus union-busy time
per pipe without double-counting overlapping events.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


_MODEL_PIPE_NAMES = {
    "PIPE_V": "VECTOR",
    "PIPE_M": "CUBE",
    "PIPE_MTE1": "MTE1",
    "PIPE_MTE2_C": "MTE2",
    "PIPE_MTE2_V": "MTE2",
    "PIPE_MTE3": "MTE3",
    "PIPE_FIX": "FIXPIPE",
    "PIPE_ALL": "ALL",
    "Scalar": "SCALAR",
}


@dataclass
class TimelineSummary:
    path: str
    source: str
    span_us: float
    busy_any_us: float
    idle_us: float
    event_count: int
    pipe_busy_us: Dict[str, float] = field(default_factory=dict)
    timing_coverage: str | None = None
    semantic_placement: str | None = None
    unscheduled_semantic_ops: int = 0
    unplaced_semantic_vector_cycles: int = 0


@dataclass
class PipeComparison:
    model_us: float
    simulator_us: float
    ratio: float | None
    error_pct: float | None


@dataclass
class TraceComparison:
    model: TimelineSummary
    simulator: TimelineSummary
    span_ratio: float
    span_error_pct: float
    pipes: Dict[str, PipeComparison]

    def to_dict(self) -> dict:
        return asdict(self)


def _merge_intervals(intervals: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    ordered = sorted(intervals)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _union_duration(intervals: Iterable[Tuple[float, float]]) -> float:
    return sum(end - start for start, end in _merge_intervals(intervals))


def _load_trace(path: Path) -> tuple[list[dict], dict]:
    with path.open() as stream:
        data = json.load(stream)
    events = data if isinstance(data, list) else data.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError(f"trace has no traceEvents array: {path}")
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    return [event for event in events if isinstance(event, dict)], metadata


def summarize_trace(path: str | Path, source: str = "auto") -> TimelineSummary:
    """Normalize one model or simulator Perfetto trace."""
    path = Path(path)
    events, metadata = _load_trace(path)
    process_names = {
        event["pid"]: str(event.get("args", {}).get("name", event["pid"]))
        for event in events
        if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    thread_names = {
        (event["pid"], event["tid"]): str(
            event.get("args", {}).get("name", event["tid"])
        )
        for event in events
        if event.get("ph") == "M" and event.get("name") == "thread_name"
    }
    if source == "auto":
        source = (
            "model"
            if any(name in _MODEL_PIPE_NAMES for name in thread_names.values())
            else "simulator"
        )
    if source not in {"model", "simulator"}:
        raise ValueError(f"unsupported trace source: {source}")

    complete = [
        event
        for event in events
        if event.get("ph") == "X" and float(event.get("dur", 0) or 0) > 0
    ]
    if not complete:
        raise ValueError(f"trace has no complete events: {path}")

    intervals_by_pipe: Dict[str, List[Tuple[float, float]]] = {}
    all_intervals: List[Tuple[float, float]] = []
    for event in complete:
        start = float(event.get("ts", 0) or 0)
        end = start + float(event.get("dur", 0) or 0)
        if source == "model":
            raw_pipe = thread_names.get(
                (event.get("pid"), event.get("tid")),
                process_names.get(event.get("pid"), "UNKNOWN"),
            )
            pipe = _MODEL_PIPE_NAMES.get(raw_pipe, raw_pipe.upper())
        else:
            pipe = process_names.get(event.get("pid"), "UNKNOWN").upper()
        intervals_by_pipe.setdefault(pipe, []).append((start, end))
        all_intervals.append((start, end))

    trace_start = min(start for start, _ in all_intervals)
    trace_end = max(end for _, end in all_intervals)
    span = trace_end - trace_start
    busy_any = _union_duration(all_intervals)
    return TimelineSummary(
        path=str(path),
        source=source,
        span_us=span,
        busy_any_us=busy_any,
        idle_us=max(0.0, span - busy_any),
        event_count=len(complete),
        pipe_busy_us={
            pipe: _union_duration(intervals)
            for pipe, intervals in sorted(intervals_by_pipe.items())
        },
        timing_coverage=metadata.get("timing_coverage"),
        semantic_placement=metadata.get("semantic_placement"),
        unscheduled_semantic_ops=int(
            metadata.get("unscheduled_semantic_ops", 0) or 0
        ),
        unplaced_semantic_vector_cycles=int(
            metadata.get("unplaced_semantic_vector_cycles", 0) or 0
        ),
    )


def compare_traces(
    model_trace: str | Path,
    simulator_trace: str | Path,
) -> TraceComparison:
    """Compare model and simulator traces using normalized pipe timelines."""
    model = summarize_trace(model_trace, source="model")
    simulator = summarize_trace(simulator_trace, source="simulator")
    span_ratio = model.span_us / simulator.span_us if simulator.span_us > 0 else 0.0
    span_error_pct = (
        100.0 * (model.span_us - simulator.span_us) / simulator.span_us
        if simulator.span_us > 0
        else 0.0
    )
    pipes = {}
    for pipe in sorted(set(model.pipe_busy_us) | set(simulator.pipe_busy_us)):
        model_us = model.pipe_busy_us.get(pipe, 0.0)
        simulator_us = simulator.pipe_busy_us.get(pipe, 0.0)
        ratio = model_us / simulator_us if simulator_us > 0 else None
        error_pct = (
            100.0 * (model_us - simulator_us) / simulator_us
            if simulator_us > 0
            else None
        )
        pipes[pipe] = PipeComparison(model_us, simulator_us, ratio, error_pct)
    return TraceComparison(model, simulator, span_ratio, span_error_pct, pipes)


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-trace", required=True)
    parser.add_argument("--simulator-trace", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    comparison = compare_traces(args.model_trace, args.simulator_trace)
    payload = comparison.to_dict()
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
    _cli()
