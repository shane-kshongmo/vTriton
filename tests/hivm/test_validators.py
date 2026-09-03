from scripts.validate_hivm_components import (
    check_buffer_metadata,
    check_component_mapping,
    check_dependency_chain,
    check_loop_multiplier,
    check_op_coverage,
    check_pipe_assignment,
    check_sync_pipe_consistency,
    check_type_inference,
)
from scripts.validate_hivm_sync import (
    check_cross_core_flag_2,
    check_mlir_des_ratio,
    check_pipe_barrier_coverage,
    check_ssa_dynamic_events,
    check_static_event_pairs,
    check_sync_block_pairs,
)


def _flag(op_id, name, sender, receiver):
    return {
        "id": op_id,
        "name": name,
        "event_id": "event_EVENT_ID0",
        "event_generation": 1,
        "core_type": "VECTOR",
        "pipe": sender if name == "set_flag" else receiver,
        "sender_pipe": sender,
        "receiver_pipe": receiver,
        "start_cycle": op_id,
    }


def test_expanded_trace_allows_unit_multipliers_and_dependency_roots():
    ops = [{
        "id": 1,
        "name": "pipe_barrier",
        "start_cycle": 17,
        "loop_multiplier": 1,
        "depends_on": [],
    }]

    assert check_dependency_chain(ops) == []
    assert check_loop_multiplier(ops) == []


def test_dependency_and_loop_checks_reject_invalid_metadata():
    ops = [{
        "id": 1,
        "name": "vadd",
        "loop_multiplier": 0,
        "depends_on": [99],
    }]

    assert "does not exist" in check_dependency_chain(ops)[0]
    assert "loop_multiplier=0" in check_loop_multiplier(ops)[0]


def test_component_metadata_checks_cover_pipe_type_and_buffers(tmp_path):
    ops = [{
        "id": 1,
        "name": "vadd",
        "pipe": "PIPE_V",
        "elements": 128,
        "elem_type": "f16",
        "bytes": 256,
        "read_buffers": ["input"],
        "write_buffers": ["output"],
    }]
    mlir = tmp_path / "kernel.npuir.mlir"
    mlir.write_text('"hivm.hir.vadd"() : () -> ()\n')

    assert check_pipe_assignment(ops) == []
    assert check_component_mapping(ops) == []
    assert check_type_inference(ops) == []
    assert check_buffer_metadata(ops) == []
    assert check_op_coverage(ops, mlir) == []


def test_component_sync_check_matches_routes_without_cartesian_pairs():
    ops = [
        _flag(1, "set_flag", "PIPE_MTE3", "PIPE_V"),
        _flag(2, "set_flag", "PIPE_V", "PIPE_S"),
        _flag(3, "wait_flag", "PIPE_V", "PIPE_S"),
        _flag(4, "wait_flag", "PIPE_MTE3", "PIPE_V"),
    ]

    assert check_sync_pipe_consistency(ops) == []


def test_component_sync_check_rejects_missing_route():
    ops = [
        _flag(1, "set_flag", "PIPE_MTE3", "PIPE_V"),
        _flag(2, "wait_flag", "PIPE_V", "PIPE_S"),
    ]

    assert "has no wait" in check_sync_pipe_consistency(ops)[0]


def test_static_sync_check_rejects_wait_without_set():
    failures = check_static_event_pairs([
        _flag(2, "wait_flag", "PIPE_MTE3", "PIPE_V"),
    ])

    assert len(failures) == 1
    assert "orphan wait_flag" in failures[0]


def test_static_sync_check_accepts_ordered_pair():
    assert check_static_event_pairs([
        _flag(1, "set_flag", "PIPE_MTE3", "PIPE_V"),
        _flag(2, "wait_flag", "PIPE_MTE3", "PIPE_V"),
    ]) == []


def test_block_sync_check_rejects_wait_without_set():
    wait = _flag(2, "sync_block_wait", "PIPE_M", "PIPE_ALL")
    wait["event_id"] = "flag_2"

    failures, cross_core = check_sync_block_pairs([wait])

    assert cross_core == 0
    assert len(failures) == 1
    assert "orphan sync_block_wait" in failures[0]


def test_sync_diagnostics_cover_block_route_barrier_ssa_and_expansion():
    block_set = {
        "id": 1,
        "name": "sync_block_set",
        "event_id": "flag_2",
        "event_generation": 1,
        "core_type": "CUBE",
        "pipe": "PIPE_MTE2_C",
        "sender_pipe": "PIPE_MTE2_C",
        "start_cycle": 1,
    }
    block_wait = {
        "id": 2,
        "name": "sync_block_wait",
        "event_id": "flag_2",
        "event_generation": 1,
        "core_type": "VECTOR",
        "pipe": "PIPE_ALL",
        "sender_pipe": "PIPE_MTE2_C",
        "start_cycle": 2,
    }
    barrier = {"id": 3, "name": "pipe_barrier", "pipe": "PIPE_V"}
    ssa_wait = {
        "id": 4,
        "name": "wait_flag",
        "event_id": "ssa_producer_1",
    }
    ops = [block_set, block_wait, barrier, ssa_wait]

    failures, cross_core = check_sync_block_pairs(ops)
    assert failures == []
    assert cross_core == 1
    assert check_cross_core_flag_2(ops) == []
    assert check_pipe_barrier_coverage(ops) == ([], {"PIPE_V": 1})
    assert check_ssa_dynamic_events(ops)["ssa_total_consumers"] == 1
    assert check_mlir_des_ratio({"pipe_barrier": 1}, ops) == {
        "pipe_barrier": 1.0,
    }
