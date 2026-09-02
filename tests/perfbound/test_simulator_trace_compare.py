import json

import pytest

from perfbound.validate.simulator_trace_compare import compare_traces, summarize_trace


def _write_trace(path, events):
    path.write_text(json.dumps({"traceEvents": events}))


def test_summarize_model_trace_uses_thread_pipe_and_interval_union(tmp_path):
    path = tmp_path / "model.json"
    _write_trace(
        path,
        [
            {"ph": "M", "name": "thread_name", "pid": 2, "tid": 1,
             "args": {"name": "PIPE_V"}},
            {"ph": "X", "name": "vadd", "pid": 2, "tid": 1,
             "ts": 1.0, "dur": 4.0},
            {"ph": "X", "name": "vmul", "pid": 2, "tid": 1,
             "ts": 3.0, "dur": 4.0},
        ],
    )

    summary = summarize_trace(path)

    assert summary.source == "model"
    assert summary.span_us == pytest.approx(6.0)
    assert summary.pipe_busy_us["VECTOR"] == pytest.approx(6.0)
    assert summary.busy_any_us == pytest.approx(6.0)


def test_compare_trace_span_and_normalized_pipes(tmp_path):
    model = tmp_path / "model.json"
    simulator = tmp_path / "simulator.json"
    _write_trace(
        model,
        [
            {"ph": "M", "name": "thread_name", "pid": 2, "tid": 1,
             "args": {"name": "PIPE_V"}},
            {"ph": "X", "name": "vadd", "pid": 2, "tid": 1,
             "ts": 0.0, "dur": 5.0},
        ],
    )
    _write_trace(
        simulator,
        [
            {"ph": "M", "name": "process_name", "pid": 30, "tid": 0,
             "args": {"name": "VECTOR"}},
            {"ph": "X", "name": "VADD", "pid": 30, "tid": 1,
             "ts": 0.0, "dur": 10.0},
        ],
    )

    comparison = compare_traces(model, simulator)

    assert comparison.span_ratio == pytest.approx(0.5)
    assert comparison.span_error_pct == pytest.approx(-50.0)
    assert comparison.pipes["VECTOR"].ratio == pytest.approx(0.5)


def test_model_trace_exposes_timing_coverage_metadata(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps({
        "traceEvents": [
            {"ph": "M", "name": "thread_name", "pid": 2, "tid": 1,
             "args": {"name": "PIPE_V"}},
            {"ph": "X", "name": "vcall", "pid": 2, "tid": 1,
             "ts": 0.0, "dur": 1.0},
        ],
        "metadata": {
            "timing_coverage": "partial",
            "semantic_placement": "weighted_vcall_heuristic",
            "unscheduled_semantic_ops": 3,
            "unplaced_semantic_vector_cycles": 2,
        },
    }))

    summary = summarize_trace(path)

    assert summary.timing_coverage == "partial"
    assert summary.semantic_placement == "weighted_vcall_heuristic"
    assert summary.unscheduled_semantic_ops == 3
    assert summary.unplaced_semantic_vector_cycles == 2
