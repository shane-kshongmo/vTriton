# Tests for M3 — HIVM Extractor.
#
# Covers:
#   - DES graph JSON parsing (operations key, legacy nodes fallback)
#   - MTE byte-vs-element aggregation
#   - Transfer metadata population (sizes, alignments)
#   - Handoff extraction from cross-component dependencies
#   - Memory space normalization
#
# Acceptance: A.3 plan AC-1, AC-2, AC-3, AC-4

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from perfbound.extract.hivm_extractor import (
    HIVMExtract,
    OpRecord,
    HandoffRecord,
    load_hivm_desgraph,
    extract_hivm,
)
from perfbound.extract.op_classifier import Component, Precision


# ── Fixtures: DES graph JSON ──────────────────────────────────────────────


def _make_des_json(operations, schema_version="a3_hivm_des_v1", clock_ghz=1.850):
    """Build a minimal DES graph JSON dict."""
    return {
        "schema_version": schema_version,
        "clock_ghz": clock_ghz,
        "operations": operations,
    }


def _vector_add_ops():
    """Vector add kernel: 2 GM→UB loads, 1 Vector add, 1 UB→GM store."""
    return [
        {
            "id": 1, "name": "data_load_a", "pipe": "VectorMTE2",
            "duration": 100, "bytes": 8192, "elements": 4096,
            "loop_multiplier": 1, "depends_on": [],
            "src_space": "gm", "dst_space": "ub", "elem_type": "f16",
            "start_cycle": 0, "end_cycle": 100,
        },
        {
            "id": 2, "name": "data_load_b", "pipe": "VectorMTE2",
            "duration": 100, "bytes": 8192, "elements": 4096,
            "loop_multiplier": 1, "depends_on": [],
            "src_space": "gm", "dst_space": "ub", "elem_type": "f16",
            "start_cycle": 0, "end_cycle": 100,
        },
        {
            "id": 3, "name": "add", "pipe": "Vector",
            "duration": 50, "bytes": 0, "elements": 4096,
            "loop_multiplier": 1, "depends_on": [1, 2],
            "src_space": "", "dst_space": "", "elem_type": "f16",
            "start_cycle": 100, "end_cycle": 150,
        },
        {
            "id": 4, "name": "data_store", "pipe": "MTE3",
            "duration": 100, "bytes": 8192, "elements": 4096,
            "loop_multiplier": 1, "depends_on": [3],
            "src_space": "ub", "dst_space": "gm", "elem_type": "f16",
            "start_cycle": 150, "end_cycle": 250,
        },
    ]


def _mixed_cv_ops():
    """Mixed Cube+Vector: Cube load, matmul, fixpipe, vector load, add, store."""
    return [
        {
            "id": 1, "name": "cube_load_a", "pipe": "CubeMTE2",
            "duration": 80, "bytes": 16384, "elements": 8192,
            "loop_multiplier": 1, "depends_on": [],
            "src_space": "gm", "dst_space": "l1", "elem_type": "f16",
            "start_cycle": 0, "end_cycle": 80,
        },
        {
            "id": 2, "name": "cube_load_b", "pipe": "CubeMTE2",
            "duration": 80, "bytes": 8192, "elements": 4096,
            "loop_multiplier": 1, "depends_on": [],
            "src_space": "gm", "dst_space": "l1", "elem_type": "f16",
            "start_cycle": 0, "end_cycle": 80,
        },
        {
            "id": 3, "name": "matmul", "pipe": "Cube",
            "duration": 200, "bytes": 0, "elements": 131072,
            "loop_multiplier": 1, "depends_on": [1, 2],
            "src_space": "", "dst_space": "", "elem_type": "f16",
            "start_cycle": 80, "end_cycle": 280,
        },
        {
            "id": 4, "name": "fixpipe", "pipe": "FixPipe",
            "duration": 60, "bytes": 8192, "elements": 4096,
            "loop_multiplier": 1, "depends_on": [3],
            "src_space": "l0c", "dst_space": "ub", "elem_type": "f16",
            "start_cycle": 280, "end_cycle": 340,
        },
        {
            "id": 5, "name": "bias_load", "pipe": "VectorMTE2",
            "duration": 40, "bytes": 8192, "elements": 4096,
            "loop_multiplier": 1, "depends_on": [],
            "src_space": "gm", "dst_space": "ub", "elem_type": "f16",
            "start_cycle": 0, "end_cycle": 40,
        },
        {
            "id": 6, "name": "add_bias", "pipe": "Vector",
            "duration": 30, "bytes": 0, "elements": 4096,
            "loop_multiplier": 1, "depends_on": [4, 5],
            "src_space": "", "dst_space": "", "elem_type": "f16",
            "start_cycle": 340, "end_cycle": 370,
        },
        {
            "id": 7, "name": "store", "pipe": "MTE3",
            "duration": 80, "bytes": 8192, "elements": 4096,
            "loop_multiplier": 1, "depends_on": [6],
            "src_space": "ub", "dst_space": "gm", "elem_type": "f16",
            "start_cycle": 370, "end_cycle": 450,
        },
    ]


# ── Test: DES graph schema (AC-1) ─────────────────────────────────────────


class TestDESGraphSchema:
    """load_hivm_desgraph() must parse C++ emitted JSON with 'operations' key."""

    def test_operations_key_parsed(self, tmp_path):
        """Canonical 'operations' key produces non-empty OpRecord list."""
        des = _make_des_json(_vector_add_ops())
        p = tmp_path / "des.json"
        p.write_text(json.dumps(des))
        ops = load_hivm_desgraph(p)
        assert len(ops) == 4
        assert all(isinstance(o, OpRecord) for o in ops)

    def test_composite_elem_type_preserves_precision(self, tmp_path):
        des = _make_des_json([
            {
                "id": 1, "name": "mmadL1.cube", "pipe": "PIPE_M",
                "duration": 1, "bytes": 0, "elements": 1024,
                "loop_multiplier": 1, "depends_on": [],
                "src_space": "l1", "dst_space": "l1",
                "elem_type": "f32, strided<[128, 128, 8, 1], offset: ?>",
                "start_cycle": 0, "end_cycle": 1,
            },
        ])
        path = tmp_path / "composite-elem-type.json"
        path.write_text(json.dumps(des))

        ops = load_hivm_desgraph(path)

        assert ops[0].component == Component.CUBE
        assert ops[0].precision == Precision.FP32

    def test_operations_fields_populated(self, tmp_path):
        """Each OpRecord has id, name, pipe, component, precision, src/dst_space."""
        des = _make_des_json(_vector_add_ops())
        p = tmp_path / "des.json"
        p.write_text(json.dumps(des))
        ops = load_hivm_desgraph(p)

        load_a = ops[0]
        assert load_a.op_id == 1
        assert load_a.op_name == "data_load_a"
        assert load_a.pipe == "VectorMTE2"
        assert load_a.component == Component.MTE_GM
        assert load_a.precision == Precision.FP16
        assert load_a.bytes_transferred == 8192
        assert load_a.elements == 4096
        assert load_a.src_space == "gm"
        assert load_a.dst_space == "ub"
        assert load_a.loop_multiplier == 1
        assert load_a.depends_on == []
        assert load_a.start_cycle == 0
        assert load_a.end_cycle == 100

    def test_legacy_nodes_key_fallback(self, tmp_path):
        """Legacy 'nodes' key still works as fallback."""
        data = {"clock_ghz": 1.85, "nodes": _vector_add_ops()}
        p = tmp_path / "legacy.json"
        p.write_text(json.dumps(data))
        ops = load_hivm_desgraph(p)
        assert len(ops) == 4

    def test_missing_operations_and_nodes_raises(self, tmp_path):
        """ValueError when JSON has neither 'operations' nor 'nodes'."""
        data = {"clock_ghz": 1.85}
        p = tmp_path / "empty.json"
        p.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="operations"):
            load_hivm_desgraph(p)

    def test_empty_operations_returns_empty_list(self, tmp_path):
        """Empty 'operations' array returns empty list."""
        des = _make_des_json([])
        p = tmp_path / "empty_ops.json"
        p.write_text(json.dumps(des))
        ops = load_hivm_desgraph(p)
        assert ops == []

    def test_missing_required_fields_raises(self, tmp_path):
        """ValueError when operation is missing required fields (id, name, pipe)."""
        data = {"operations": [{"bytes": 100}]}  # missing id, name, pipe
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="missing required fields"):
            load_hivm_desgraph(p)

    def test_partial_required_fields_raises(self, tmp_path):
        """ValueError when operation has some but not all required fields."""
        data = {"operations": [{"id": 1, "name": "load"}]}  # missing pipe
        p = tmp_path / "partial.json"
        p.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="missing required fields.*pipe"):
            load_hivm_desgraph(p)


# ── Test: MTE byte-vs-element aggregation (AC-2) ──────────────────────────


class TestOprecAggregation:
    """O_prec: compute components use elements/flops; MTE uses bytes."""

    def test_mte_uses_bytes_not_elements(self, tmp_path):
        """MTE o_prec must aggregate bytes, not elements."""
        des = _make_des_json(_vector_add_ops())
        p = tmp_path / "des.json"
        p.write_text(json.dumps(des))
        extract = extract_hivm(p)

        # MTE_GM: two loads of 8192 bytes each = 16384 bytes total
        mte_gm_fp16 = extract.o_prec.get("mte_gm", {}).get("fp16", 0)
        assert mte_gm_fp16 == 16384, \
            f"MTE_GM o_prec should be 16384 bytes, got {mte_gm_fp16}"

    def test_compute_uses_elements(self, tmp_path):
        """Compute o_prec must aggregate elements (or flops)."""
        des = _make_des_json(_vector_add_ops())
        p = tmp_path / "des.json"
        p.write_text(json.dumps(des))
        extract = extract_hivm(p)

        # Vector add: 4096 elements
        vec_fp16 = extract.o_prec.get("vector", {}).get("fp16", 0)
        assert vec_fp16 == 4096, \
            f"Vector o_prec should be 4096 elements, got {vec_fp16}"

    def test_mte_ub_uses_bytes(self, tmp_path):
        """MTE_UB (store) o_prec uses bytes."""
        des = _make_des_json(_vector_add_ops())
        p = tmp_path / "des.json"
        p.write_text(json.dumps(des))
        extract = extract_hivm(p)

        mte_ub_fp16 = extract.o_prec.get("mte_ub", {}).get("fp16", 0)
        assert mte_ub_fp16 == 8192, \
            f"MTE_UB o_prec should be 8192 bytes, got {mte_ub_fp16}"

    def test_loop_multiplier_applied(self, tmp_path):
        """Loop multiplier scales work correctly for both compute and MTE."""
        ops = [
            {
                "id": 1, "name": "matmul", "pipe": "Cube",
                "duration": 200, "bytes": 0, "elements": 1000,
                "flops": 2000, "loop_multiplier": 4, "depends_on": [],
                "src_space": "", "dst_space": "", "elem_type": "f16",
            },
            {
                "id": 2, "name": "load", "pipe": "CubeMTE2",
                "duration": 50, "bytes": 4096, "elements": 2048,
                "loop_multiplier": 4, "depends_on": [],
                "src_space": "gm", "dst_space": "l1", "elem_type": "f16",
            },
        ]
        des = _make_des_json(ops)
        p = tmp_path / "des.json"
        p.write_text(json.dumps(des))
        extract = extract_hivm(p)

        # Cube: flops=2000, loop_mult=4 → 8000
        assert extract.o_prec["cube"]["fp16"] == 8000
        # MTE_GM: bytes=4096, loop_mult=4 → 16384
        assert extract.o_prec["mte_gm"]["fp16"] == 16384


# ── Test: Transfer metadata (AC-3) ───────────────────────────────────────


class TestTransferMetadata:
    """transfer_sizes and transfer_alignments populated for MTE ops."""

    def test_transfer_sizes_populated(self, tmp_path):
        """Every MTE op contributes its byte size to transfer_sizes."""
        des = _make_des_json(_vector_add_ops())
        p = tmp_path / "des.json"
        p.write_text(json.dumps(des))
        extract = extract_hivm(p)

        assert "mte_gm" in extract.transfer_sizes
        assert len(extract.transfer_sizes["mte_gm"]) == 2  # two loads
        assert extract.transfer_sizes["mte_gm"] == [8192, 8192]

    def test_transfer_alignments_unknown(self, tmp_path):
        """Transfer alignments are 0 (unknown) until C++ provides address data."""
        des = _make_des_json(_vector_add_ops())
        p = tmp_path / "des.json"
        p.write_text(json.dumps(des))
        extract = extract_hivm(p)

        assert "mte_gm" in extract.transfer_alignments
        # 0 = alignment unknown (C++ emitter does not expose address alignment)
        assert extract.transfer_alignments["mte_gm"] == [0, 0]

    def test_mte_ub_transfer_metadata(self, tmp_path):
        """MTE_UB (store) also has transfer_sizes."""
        des = _make_des_json(_vector_add_ops())
        p = tmp_path / "des.json"
        p.write_text(json.dumps(des))
        extract = extract_hivm(p)

        assert "mte_ub" in extract.transfer_sizes
        assert extract.transfer_sizes["mte_ub"] == [8192]

    def test_no_compute_in_transfer_sizes(self, tmp_path):
        """Compute components (cube, vector, scalar) do NOT appear in transfer_sizes."""
        des = _make_des_json(_vector_add_ops())
        p = tmp_path / "des.json"
        p.write_text(json.dumps(des))
        extract = extract_hivm(p)

        assert "cube" not in extract.transfer_sizes
        assert "vector" not in extract.transfer_sizes
        assert "scalar" not in extract.transfer_sizes


# ── Test: Handoff extraction (AC-4) ──────────────────────────────────────


class TestHandoffExtraction:
    """Cross-component edges become HandoffRecords."""

    def test_canonical_cube_to_vector_handoff(self, tmp_path):
        """Cube→Vector canonical handoff traced through FixPipe(MTE_UB) intermediary.

        The dependency chain: matmul(Cube) → fixpipe(MTE_UB) → add_bias(Vector).
        The extractor must trace through MTE_UB to emit a canonical Cube→Vector
        handoff for serialization.py mandatory/avoidable classification.
        """
        des = _make_des_json(_mixed_cv_ops())
        p = tmp_path / "des.json"
        p.write_text(json.dumps(des))
        extract = extract_hivm(p)

        cv_handoffs = [
            h for h in extract.handoffs
            if h.producer_component == Component.CUBE
            and h.consumer_component == Component.VECTOR
        ]
        assert len(cv_handoffs) >= 1, \
            (f"Expected canonical Cube→Vector handoff, got: "
             f"{[h.producer_component.value + '->' + h.consumer_component.value for h in extract.handoffs]}")

    def test_immediate_mte_ub_to_vector_handoff(self, tmp_path):
        """Immediate MTE_UB→Vector edge also emitted (non-canonical)."""
        des = _make_des_json(_mixed_cv_ops())
        p = tmp_path / "des.json"
        p.write_text(json.dumps(des))
        extract = extract_hivm(p)

        ub_to_vec = [
            h for h in extract.handoffs
            if h.producer_component == Component.MTE_UB
            and h.consumer_component == Component.VECTOR
        ]
        assert len(ub_to_vec) >= 1, \
            f"Expected at least one MTE_UB->Vector handoff, got {len(ub_to_vec)}"

    def test_mte_gm_to_cube_handoff(self, tmp_path):
        """MTE_GM→Cube is cross-component (load feeds matmul)."""
        des = _make_des_json(_mixed_cv_ops())
        p = tmp_path / "des.json"
        p.write_text(json.dumps(des))
        extract = extract_hivm(p)

        gm_to_cube = [
            h for h in extract.handoffs
            if h.producer_component == Component.MTE_GM
            and h.consumer_component == Component.CUBE
        ]
        assert len(gm_to_cube) >= 1, \
            f"Expected at least one MTE_GM→Cube handoff, got {len(gm_to_cube)}"

    def test_no_same_component_handoffs(self, tmp_path):
        """Same-component dependencies are NOT handoffs."""
        # Vector add depends on two loads — both MTE_GM, not same as Vector
        des = _make_des_json(_vector_add_ops())
        p = tmp_path / "des.json"
        p.write_text(json.dumps(des))
        extract = extract_hivm(p)

        same_comp = [
            h for h in extract.handoffs
            if h.producer_component == h.consumer_component
        ]
        assert len(same_comp) == 0, \
            f"Same-component handoffs should not exist, got {len(same_comp)}"


# ── Test: Memory space normalization ─────────────────────────────────────


class TestMemorySpaceNormalization:
    """src_space and dst_space normalized to canonical vocabulary."""

    def test_gm_ub_l1_normalized(self, tmp_path):
        """Standard space names pass through."""
        des = _make_des_json(_vector_add_ops())
        p = tmp_path / "des.json"
        p.write_text(json.dumps(des))
        extract = extract_hivm(p)

        # First load: gm→ub
        assert extract.operations[0].src_space == "gm"
        assert extract.operations[0].dst_space == "ub"

    def test_l0c_space_preserved(self, tmp_path):
        """l0c space (from FixPipe) is preserved."""
        des = _make_des_json(_mixed_cv_ops())
        p = tmp_path / "des.json"
        p.write_text(json.dumps(des))
        extract = extract_hivm(p)

        fixpipe = next(o for o in extract.operations if o.op_name == "fixpipe")
        assert fixpipe.src_space == "l0c"
        assert fixpipe.dst_space == "ub"


# ── Test: Unit assignment ─────────────────────────────────────────────────


class TestUnitAssignment:
    """unit_assignment maps op_id → component name."""

    def test_all_ops_assigned(self, tmp_path):
        """Every op has a unit assignment."""
        des = _make_des_json(_vector_add_ops())
        p = tmp_path / "des.json"
        p.write_text(json.dumps(des))
        extract = extract_hivm(p)

        assert len(extract.unit_assignment) == 4
        assert extract.unit_assignment[1] == "mte_gm"
        assert extract.unit_assignment[3] == "vector"
        assert extract.unit_assignment[4] == "mte_ub"


# ── Test: Total cycles ────────────────────────────────────────────────────


class TestTotalCycles:
    """total_cycles computed from max end_cycle."""

    def test_total_cycles_from_end_cycle(self, tmp_path):
        des = _make_des_json(_vector_add_ops())
        p = tmp_path / "des.json"
        p.write_text(json.dumps(des))
        extract = extract_hivm(p)

        # Last op (store) ends at cycle 250
        assert extract.total_cycles == 250


# ── Test: AC-1 O_prec reconciliation against analytic ground truth ────────


def _gemm_ops(M, N, K, dtype_bytes=2, elem_type="f16"):
    """Build a GEMM DES graph: 2 loads + matmul + FixPipe store."""
    return [
        {
            "id": 1, "name": "cube_load_a", "pipe": "CubeMTE2",
            "duration": 80, "bytes": dtype_bytes * M * K, "elements": M * K,
            "loop_multiplier": 1, "depends_on": [],
            "src_space": "gm", "dst_space": "l1", "elem_type": elem_type,
            "start_cycle": 0, "end_cycle": 80,
        },
        {
            "id": 2, "name": "cube_load_b", "pipe": "CubeMTE2",
            "duration": 80, "bytes": dtype_bytes * K * N, "elements": K * N,
            "loop_multiplier": 1, "depends_on": [],
            "src_space": "gm", "dst_space": "l1", "elem_type": elem_type,
            "start_cycle": 0, "end_cycle": 80,
        },
        {
            "id": 3, "name": "matmul", "pipe": "Cube",
            "duration": 200, "bytes": 0, "elements": M * N,
            "flops": 2 * M * N * K,
            "loop_multiplier": 1, "depends_on": [1, 2],
            "src_space": "", "dst_space": "", "elem_type": elem_type,
            "start_cycle": 80, "end_cycle": 280,
        },
        {
            "id": 4, "name": "fixpipe", "pipe": "FixPipe",
            "duration": 60, "bytes": dtype_bytes * M * N, "elements": M * N,
            "loop_multiplier": 1, "depends_on": [3],
            "src_space": "l0c", "dst_space": "ub", "elem_type": elem_type,
            "start_cycle": 280, "end_cycle": 340,
        },
    ]


class TestAC1Reconciliation:
    """AC-1: Σ O_prec reconciles with analytic flop/byte count within 2%.

    Non-tautological: the DES JSON has per-op values; the analytic ground
    truth is computed independently from M, N, K. The test verifies the
    extractor's aggregation matches.
    """

    def test_cube_flops_reconcile_128(self, tmp_path):
        """GEMM 128³: Σ Cube O_prec = 2·M·N·K within 2%."""
        M = N = K = 128
        des = _make_des_json(_gemm_ops(M, N, K))
        p = tmp_path / "gemm.json"
        p.write_text(json.dumps(des))
        extract = extract_hivm(p)

        analytic_flops = 2 * M * N * K
        cube_flops = extract.o_prec.get("cube", {}).get("fp16", 0)
        rel_err = abs(cube_flops - analytic_flops) / analytic_flops
        assert rel_err < 0.02, \
            f"Cube O_prec {cube_flops} vs analytic {analytic_flops}: {rel_err:.4f}"

    def test_mte_gm_bytes_reconcile_128(self, tmp_path):
        """GEMM 128³: Σ MTE_GM O_prec = bytes(A) + bytes(B) within 2%."""
        M = N = K = 128
        DTYPE = 2  # fp16
        des = _make_des_json(_gemm_ops(M, N, K))
        p = tmp_path / "gemm.json"
        p.write_text(json.dumps(des))
        extract = extract_hivm(p)

        analytic_bytes = DTYPE * (M * K + K * N)
        mte_bytes = extract.o_prec.get("mte_gm", {}).get("fp16", 0)
        rel_err = abs(mte_bytes - analytic_bytes) / analytic_bytes
        assert rel_err < 0.02, \
            f"MTE_GM O_prec {mte_bytes} vs analytic {analytic_bytes}: {rel_err:.4f}"

    def test_cube_flops_reconcile_4096(self, tmp_path):
        """GEMM 4096³: realistic LLM hidden-dim scale."""
        M = N = K = 4096
        des = _make_des_json(_gemm_ops(M, N, K))
        p = tmp_path / "gemm.json"
        p.write_text(json.dumps(des))
        extract = extract_hivm(p)

        analytic_flops = 2 * M * N * K
        cube_flops = extract.o_prec.get("cube", {}).get("fp16", 0)
        assert cube_flops == analytic_flops

    def test_mte_gm_bytes_with_loop_multiplier(self, tmp_path):
        """Loop multiplier scales both flops and bytes correctly."""
        M = N = K = 64
        LOOP = 8
        ops = _gemm_ops(M, N, K)
        for op in ops:
            op["loop_multiplier"] = LOOP
        des = _make_des_json(ops)
        p = tmp_path / "gemm_loop.json"
        p.write_text(json.dumps(des))
        extract = extract_hivm(p)

        analytic_flops = 2 * M * N * K * LOOP
        cube_flops = extract.o_prec.get("cube", {}).get("fp16", 0)
        assert cube_flops == analytic_flops

        analytic_bytes = (2 * M * K + 2 * K * N) * LOOP  # dtype=2 bytes per elem
        mte_bytes = extract.o_prec.get("mte_gm", {}).get("fp16", 0)
        assert mte_bytes == analytic_bytes


# ── Test: Flops inference from input loads ────────────────────────────────


class TestFlopsInference:
    """When C++ emits flops=0 for matmul, Python infers 2·M·N·K from loads."""

    def test_flops_inferred_from_loads(self, tmp_path):
        """Matmul with flops=0 gets flops inferred from input load sizes."""
        M, N, K = 128, 64, 256
        ops = [
            {
                "id": 1, "name": "cube_load_a", "pipe": "CubeMTE2",
                "duration": 80, "bytes": 2 * M * K, "elements": M * K,
                "loop_multiplier": 1, "depends_on": [],
                "src_space": "gm", "dst_space": "l1", "elem_type": "f16",
            },
            {
                "id": 2, "name": "cube_load_b", "pipe": "CubeMTE2",
                "duration": 80, "bytes": 2 * K * N, "elements": K * N,
                "loop_multiplier": 1, "depends_on": [],
                "src_space": "gm", "dst_space": "l1", "elem_type": "f16",
            },
            {
                "id": 3, "name": "matmul", "pipe": "Cube",
                "duration": 200, "bytes": 0, "elements": M * N,
                "flops": 0,  # C++ didn't compute flops
                "loop_multiplier": 1, "depends_on": [1, 2],
                "src_space": "", "dst_space": "", "elem_type": "f16",
            },
        ]
        des = _make_des_json(ops)
        p = tmp_path / "infer.json"
        p.write_text(json.dumps(des))
        extract = extract_hivm(p)

        analytic_flops = 2 * M * N * K
        cube_flops = extract.o_prec.get("cube", {}).get("fp16", 0)
        assert cube_flops == analytic_flops, \
            f"Inferred flops {cube_flops} != analytic {analytic_flops}"

    def test_flops_not_overwritten_when_present(self, tmp_path):
        """When C++ provides flops, inference does not overwrite."""
        M, N, K = 32, 32, 32
        explicit_flops = 99999  # deliberately different from 2*M*N*K
        ops = [
            {
                "id": 1, "name": "cube_load_a", "pipe": "CubeMTE2",
                "duration": 80, "bytes": 2 * M * K, "elements": M * K,
                "loop_multiplier": 1, "depends_on": [],
                "src_space": "gm", "dst_space": "l1", "elem_type": "f16",
            },
            {
                "id": 2, "name": "cube_load_b", "pipe": "CubeMTE2",
                "duration": 80, "bytes": 2 * K * N, "elements": K * N,
                "loop_multiplier": 1, "depends_on": [],
                "src_space": "gm", "dst_space": "l1", "elem_type": "f16",
            },
            {
                "id": 3, "name": "matmul", "pipe": "Cube",
                "duration": 200, "bytes": 0, "elements": M * N,
                "flops": explicit_flops,
                "loop_multiplier": 1, "depends_on": [1, 2],
                "src_space": "", "dst_space": "", "elem_type": "f16",
            },
        ]
        des = _make_des_json(ops)
        p = tmp_path / "explicit.json"
        p.write_text(json.dumps(des))
        extract = extract_hivm(p)

        assert extract.o_prec["cube"]["fp16"] == explicit_flops

    def test_no_inference_without_loads(self, tmp_path):
        """Matmul with no input loads falls back to elements, not inference."""
        ops = [
            {
                "id": 3, "name": "matmul", "pipe": "Cube",
                "duration": 200, "bytes": 0, "elements": 4096,
                "flops": 0,
                "loop_multiplier": 1, "depends_on": [],
                "src_space": "", "dst_space": "", "elem_type": "f16",
            },
        ]
        des = _make_des_json(ops)
        p = tmp_path / "no_loads.json"
        p.write_text(json.dumps(des))
        extract = extract_hivm(p)

        # Falls back to elements since no loads to infer from
        assert extract.o_prec["cube"]["fp16"] == 4096
