# CLI integration tests for M3 — HIVM Extractor.
#
# These tests require the build/bin/tritonsim-hivm binary.
# They are automatically skipped when the binary is not available.
#
# Acceptance: A.3 plan AC-1 (end-to-end verification)

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from perfbound.extract.hivm_extractor import load_hivm_desgraph, extract_hivm


# Binary paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRITONSIM_HIVM = PROJECT_ROOT / "build" / "bin" / "tritonsim-hivm"
TRITONSIM_OPT = PROJECT_ROOT / "build" / "bin" / "tritonsim-opt"

# Test fixtures
FIXTURE_DIR = PROJECT_ROOT / "test"
HIVM_ADD_KERNEL = FIXTURE_DIR / "hivm_add_kernel.npuir.mlir"
HIVM_MIXED_CV_KERNEL = FIXTURE_DIR / "hivm_mixed_cv_kernel.npuir.mlir"

# Hardware config
HW_CONFIG = PROJECT_ROOT / "configs" / "ascend_910b3_v4.json"
CALIBRATED_HW_CONFIG = PROJECT_ROOT / "configs" / "ascend_910b3_v4.json"


# Skip markers
requires_tritonsim_hivm = pytest.mark.skipif(
    not TRITONSIM_HIVM.exists(),
    reason="build/bin/tritonsim-hivm not found — build the project first",
)

requires_tritonsim_opt = pytest.mark.skipif(
    not TRITONSIM_OPT.exists(),
    reason="build/bin/tritonsim-opt not found — build the project first",
)

requires_fixtures = pytest.mark.skipif(
    not HIVM_ADD_KERNEL.exists(),
    reason="test/hivm_add_kernel.npuir.mlir not found",
)


def _run_cli(tool: Path, args: list[str], out_file: Path) -> subprocess.CompletedProcess:
    """Run a CLI tool and return the result. Fails test if command errors."""
    cmd = [str(tool)] + args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return result


@requires_tritonsim_hivm
@requires_fixtures
class TestTritonsimHivmCLI:
    """Tests using tritonsim-hivm --des-graph-file.

    These tests exercise the typed HIVM dialect parser backed by bishengir
    libraries built from AscendNPU-IR's LLVM 19.1.7 tree.
    """

    def test_des_graph_emitted(self, tmp_path):
        """tritonsim-hivm emits valid JSON with 'operations' array."""
        out_file = tmp_path / "hivm_add_des.json"
        cmd = [
            str(TRITONSIM_HIVM),
            "--npuir-file", str(HIVM_ADD_KERNEL),
            "--des-graph-file", str(out_file),
        ]
        if HW_CONFIG.exists():
            cmd.extend(["--hardware-config", str(HW_CONFIG)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (
            f"tritonsim-hivm failed (returncode={result.returncode}): "
            f"{result.stderr[:300]}"
        )
        assert out_file.exists() and out_file.stat().st_size > 0, (
            "DES graph file was not emitted by tritonsim-hivm"
        )

        data = json.loads(out_file.read_text())
        assert "operations" in data or "nodes" in data
        ops = data.get("operations", data.get("nodes", []))
        assert len(ops) > 0, "DES graph must contain at least one operation"

    def test_des_graph_parseable(self, tmp_path):
        """Emitted DES graph is parseable by load_hivm_desgraph()."""
        out_file = tmp_path / "hivm_add_des.json"
        cmd = [
            str(TRITONSIM_HIVM),
            "--npuir-file", str(HIVM_ADD_KERNEL),
            "--des-graph-file", str(out_file),
        ]
        if HW_CONFIG.exists():
            cmd.extend(["--hardware-config", str(HW_CONFIG)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (
            f"tritonsim-hivm failed (returncode={result.returncode}): "
            f"{result.stderr[:300]}"
        )
        assert out_file.exists() and out_file.stat().st_size > 0, (
            "DES graph file was not emitted by tritonsim-hivm"
        )

        ops = load_hivm_desgraph(out_file)
        assert len(ops) > 0, "Parsed operations must be non-empty"

    def test_static_and_resolved_dynamic_sync_flags_share_generations(
        self, tmp_path
    ):
        """A dynamic flag value must match the equivalent static pre-signal."""
        npuir_file = tmp_path / "dynamic_sync_flag.npuir.mlir"
        npuir_file.write_text(
            """
module {
  func.func @dynamic_sync_mix_aic() attributes {
      hacc.entry,
      hivm.func_core_type = #hivm.func_core_type<AIC>
  } {
    %c1 = arith.constant 1 : i64
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = %c1
    return
  }
  func.func @dynamic_sync_mix_aiv() attributes {
      hacc.entry,
      hivm.func_core_type = #hivm.func_core_type<AIV>
  } {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, %c0]
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = %c1
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
    return
  }
}
"""
        )
        out_file = tmp_path / "dynamic_sync_flag.des.json"
        cmd = [
            str(TRITONSIM_HIVM),
            "--npuir-file", str(npuir_file),
            "--scheduler", "des",
            "--des-graph-file", str(out_file),
        ]
        if HW_CONFIG.exists():
            cmd.extend(["--hardware-config", str(HW_CONFIG)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        data = json.loads(out_file.read_text())
        assert data["schedule_truncated"] is False

        operations = data["operations"]
        cube_sets = [
            op for op in operations
            if op["name"] == "sync_block_set"
            and op["core_type"] == "CUBE"
            and op["event_id"] == "flag_1"
        ]
        vector_wait = next(
            op for op in operations
            if op["name"] == "sync_block_wait"
            and op["core_type"] == "VECTOR"
        )
        assert len(cube_sets) == 2
        assert cube_sets[0]["id"] in vector_wait["depends_on"]
        assert cube_sets[1]["id"] not in vector_wait["depends_on"]

        local_set = next(
            op for op in operations
            if op["name"] == "set_flag"
            and op["event_id"] == "event_EVENT_ID0"
        )
        local_wait = next(
            op for op in operations
            if op["name"] == "wait_flag"
            and op["event_id"] == "event_EVENT_ID0"
        )
        assert local_set["id"] in local_wait["depends_on"]

    def test_semantic_sidecar_adds_typed_work_and_completes_coverage(self, tmp_path):
        """Pre-outline tensor work supplements, but does not replace, HIVM topology."""
        semantic_file = tmp_path / "kernel.ttadapter.mlir"
        semantic_file.write_text(
            """
module {
  func.func @semantic_kernel() attributes {global_kernel = "local"} {
    %empty = tensor.empty() : tensor<128xf32>
    %result = math.exp %empty : tensor<128xf32>
    return
  }
}
"""
        )
        out_file = tmp_path / "semantic_des.json"
        cmd = [
            str(TRITONSIM_HIVM),
            "--npuir-file", str(HIVM_ADD_KERNEL),
            "--semantic-ir-file", str(semantic_file),
            "--des-graph-file", str(out_file),
        ]
        if HW_CONFIG.exists():
            cmd.extend(["--hardware-config", str(HW_CONFIG)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr[:500]
        data = json.loads(out_file.read_text())
        coverage = data["model_coverage"]
        assert coverage["status"] == "complete"
        assert coverage["semantic_overlay"]["complete"] is True
        synthetic = [
            op for op in data["operations"]
            if op.get("cost_source") == "ttadapter_semantic_overlay"
        ]
        assert any(op["name"] == "vexp" and op["elements"] == 128
                   for op in synthetic)

    def test_semantic_sidecar_rebuilds_vcall_timeline(self, tmp_path):
        """Recovered vector work must be visible in the rescheduled trace."""
        source = HIVM_ADD_KERNEL.read_text()
        marker = "  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]\n"
        call = (
            "  func.call @missing_outlined_vf(%ub0) "
            "{hivm.vector_function, no_inline} : "
            "(memref<1024xf32, #hivm.address_space<ub>>) -> ()\n"
        )
        assert marker in source
        npuir_file = tmp_path / "semantic_vcall.npuir.mlir"
        npuir_file.write_text(source.replace(marker, call + call + marker, 1))

        semantic_file = tmp_path / "semantic_vcall.ttadapter.mlir"
        semantic_file.write_text(
            """
module {
  func.func @semantic_kernel() attributes {global_kernel = "local"} {
    %empty = tensor.empty() : tensor<128xf32>
    %result = math.rsqrt %empty : tensor<128xf32>
    return
  }
}
"""
        )
        out_file = tmp_path / "semantic_vcall_des.json"
        trace_file = tmp_path / "semantic_vcall_trace.json"
        cmd = [
            str(TRITONSIM_HIVM),
            "--npuir-file", str(npuir_file),
            "--semantic-ir-file", str(semantic_file),
            "--scheduler", "des",
            "--des-graph-file", str(out_file),
            "--perfetto-trace-file", str(trace_file),
        ]
        if HW_CONFIG.exists():
            cmd.extend(["--hardware-config", str(HW_CONFIG)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr[:500]

        data = json.loads(out_file.read_text())
        calls = [op for op in data["operations"] if op["name"] == "vcall"]
        assert len(calls) == 2
        assert all(call["elements"] == 0 for call in calls)
        assert all(call["duration"] > 0 for call in calls)
        if HW_CONFIG.exists():
            assert sum(call["duration"] for call in calls) == 15
        assert all(call["end_cycle"] > call["start_cycle"] for call in calls)
        assert {
            call["cost_source"] for call in calls
        } == {"ttadapter_semantic_projection"}

        trace = json.loads(trace_file.read_text())
        assert trace["metadata"] == {
            "timing_coverage": "complete",
            "semantic_placement": "weighted_vcall_heuristic",
            "unscheduled_semantic_ops": 0,
            "unplaced_semantic_vector_cycles": 0,
        }
        trace_calls = [
            event for event in trace["traceEvents"]
            if event.get("ph") == "X" and event.get("name") == "vcall"
        ]
        assert len(trace_calls) == 2
        assert all(call["dur"] > 0 for call in trace_calls)
        assert sum(call["args"]["cycles"] for call in trace_calls) == sum(
            call["duration"] for call in calls
        )
        assert not any(
            event.get("ph") == "X" and
            event.get("args", {}).get("cost_source") ==
            "ttadapter_semantic_overlay"
            for event in trace["traceEvents"]
        )

    def test_static_semantic_projection_does_not_round_above_calibration(
        self, tmp_path
    ):
        source = HIVM_ADD_KERNEL.read_text()
        marker = "  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]\n"
        looped_call = (
            "  %c1 = arith.constant 1 : index\n"
            "  %c4 = arith.constant 4 : index\n"
            "  scf.for %i = %c0 to %c4 step %c1 {\n"
            "    func.call @missing_outlined_vf(%ub0) "
            "{hivm.vector_function, no_inline} : "
            "(memref<1024xf32, #hivm.address_space<ub>>) -> ()\n"
            "  }\n"
        )
        npuir_file = tmp_path / "static_semantic_vcall.npuir.mlir"
        npuir_file.write_text(source.replace(marker, looped_call + marker, 1))
        semantic_file = tmp_path / "static_semantic_vcall.ttadapter.mlir"
        semantic_file.write_text(
            """
module {
  func.func @semantic_kernel() attributes {global_kernel = "local"} {
    %empty = tensor.empty() : tensor<128xf32>
    %result = math.rsqrt %empty : tensor<128xf32>
    return
  }
}
"""
        )
        out_file = tmp_path / "static_semantic_vcall_des.json"
        cmd = [
            str(TRITONSIM_HIVM),
            "--npuir-file", str(npuir_file),
            "--semantic-ir-file", str(semantic_file),
            "--scheduler", "static",
            "--des-graph-file", str(out_file),
        ]
        if HW_CONFIG.exists():
            cmd.extend(["--hardware-config", str(HW_CONFIG)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr[:500]
        data = json.loads(out_file.read_text())
        calls = [
            op for op in data["operations"]
            if op["name"] == "vcall"
        ]
        assert len(calls) == 1
        assert calls[0]["loop_multiplier"] == 4
        assert 0 < calls[0]["duration"] * 4 <= 15
        assert data["model_coverage"]["trace_timing_status"] == "partial"
        assert data["model_coverage"]["unplaced_semantic_vector_cycles"] == 3

    def test_unresolved_semantic_loop_keeps_coverage_partial(self, tmp_path):
        """A sidecar cannot authorize headroom when its own trip count is unknown."""
        semantic_file = tmp_path / "dynamic.ttadapter.mlir"
        semantic_file.write_text(
            """
module {
  func.func @semantic_kernel(%upper: index) attributes {global_kernel = "local"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %upper step %c1 {
      %empty = tensor.empty() : tensor<128xf32>
      %result = math.exp %empty : tensor<128xf32>
    }
    return
  }
}
"""
        )
        out_file = tmp_path / "dynamic_semantic_des.json"
        result = subprocess.run(
            [
                str(TRITONSIM_HIVM),
                "--npuir-file", str(HIVM_ADD_KERNEL),
                "--semantic-ir-file", str(semantic_file),
                "--des-graph-file", str(out_file),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr[:500]
        coverage = json.loads(out_file.read_text())["model_coverage"]
        assert coverage["status"] == "conservative_partial"
        assert coverage["trace_timing_status"] == "partial"
        assert coverage["semantic_overlay"]["unresolved_loops"] == 1

    def test_unresolved_but_model_equivalent_branch_is_complete(self, tmp_path):
        """A predicate guarding only fusible fill semantics does not lose work."""
        semantic_file = tmp_path / "equivalent_branch.ttadapter.mlir"
        semantic_file.write_text(
            """
module {
  func.func @semantic_kernel(%flag: i1) attributes {global_kernel = "local"} {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<128xf32>
    scf.if %flag {
      %filled = linalg.fill ins(%zero : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
    }
    %result = math.exp %empty : tensor<128xf32>
    return
  }
}
"""
        )
        out_file = tmp_path / "equivalent_branch_des.json"
        result = subprocess.run(
            [
                str(TRITONSIM_HIVM),
                "--npuir-file", str(HIVM_ADD_KERNEL),
                "--semantic-ir-file", str(semantic_file),
                "--des-graph-file", str(out_file),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr[:500]
        coverage = json.loads(out_file.read_text())["model_coverage"]
        assert coverage["status"] == "complete"
        assert coverage["semantic_overlay"]["model_equivalent_branches"] == 1
        assert coverage["semantic_overlay"]["unresolved_branches"] == 0

    def test_non_scheduling_eviction_policy_attr_is_ignored(self, tmp_path):
        """NPUIR dump-only load attrs should not block DES modeling."""
        source = HIVM_ADD_KERNEL.read_text()
        marker = (
            "      outs(%ub0 : memref<1024xf32, #hivm.address_space<ub>>)"
        )
        assert marker in source
        npuir_file = tmp_path / "hivm_add_eviction_policy.npuir.mlir"
        npuir_file.write_text(
            source.replace(
                marker,
                marker + " eviction_policy = <EvictFirst>",
                1,
            )
        )
        out_file = tmp_path / "hivm_add_eviction_policy_des.json"
        cmd = [
            str(TRITONSIM_HIVM),
            "--npuir-file", str(npuir_file),
            "--des-graph-file", str(out_file),
        ]
        if CALIBRATED_HW_CONFIG.exists():
            cmd.extend(["--hardware-config", str(CALIBRATED_HW_CONFIG)])
        elif HW_CONFIG.exists():
            cmd.extend(["--hardware-config", str(HW_CONFIG)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (
            f"tritonsim-hivm failed (returncode={result.returncode}): "
            f"{result.stderr[:300]}"
        )
        data = json.loads(out_file.read_text())
        ops = data.get("operations", data.get("nodes", []))
        assert len(ops) > 0, "DES graph must contain at least one operation"

    def test_remove_pipe_barrier_emits_edited_npuir(self, tmp_path):
        """tritonsim-hivm can erase a pipe_barrier through MLIR parsing."""
        edited_file = tmp_path / "hivm_add_no_barrier.npuir.mlir"
        out_file = tmp_path / "hivm_add_no_barrier_des.json"
        cmd = [
            str(TRITONSIM_HIVM),
            "--npuir-file", str(HIVM_ADD_KERNEL),
            "--remove-pipe-barrier-index", "0",
            "--edited-npuir-file", str(edited_file),
            "--des-graph-file", str(out_file),
        ]
        if HW_CONFIG.exists():
            cmd.extend(["--hardware-config", str(HW_CONFIG)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (
            f"tritonsim-hivm edit failed (returncode={result.returncode}): "
            f"{result.stderr[:300]}"
        )
        assert edited_file.exists() and edited_file.stat().st_size > 0
        assert "hivm.hir.pipe_barrier" not in edited_file.read_text()
        assert out_file.exists() and out_file.stat().st_size > 0

    def test_remove_pipe_barrier_requires_output_path(self):
        """The destructive edit flag must name an edited NPUIR output."""
        cmd = [
            str(TRITONSIM_HIVM),
            "--npuir-file", str(HIVM_ADD_KERNEL),
            "--remove-pipe-barrier-index", "0",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
        assert "must be provided together" in result.stderr

    def test_static_scf_for_emits_loop_multiplier_and_diagnostics(self, tmp_path):
        """Static scf.for trip counts should be resolved and replayed in DES."""
        source = HIVM_ADD_KERNEL.read_text()
        source = source.replace(
            "  %c0 = arith.constant 0 : index\n",
            (
                "  %c0 = arith.constant 0 : index\n"
                "  %c1 = arith.constant 1 : index\n"
                "  %c4 = arith.constant 4 : index\n"
            ),
            1,
        )
        loop_start = (
            "  hivm.hir.vadd ins(%ub0, %ub1 : memref<1024xf32, #hivm.address_space<ub>>,"
        )
        source = source.replace(loop_start, "  scf.for %i = %c0 to %c4 step %c1 {\n" + loop_start, 1)
        loop_end = (
            "      outs(%ub2 : memref<1024xf32, #hivm.address_space<ub>>)\n"
        )
        source = source.replace(loop_end, loop_end + "  }\n", 1)

        npuir_file = tmp_path / "hivm_add_loop.npuir.mlir"
        npuir_file.write_text(source)
        out_file = tmp_path / "hivm_add_loop_des.json"
        cmd = [
            str(TRITONSIM_HIVM),
            "--npuir-file", str(npuir_file),
            "--scheduler", "des",
            "--des-graph-file", str(out_file),
        ]
        if CALIBRATED_HW_CONFIG.exists():
            cmd.extend(["--hardware-config", str(CALIBRATED_HW_CONFIG)])
        elif HW_CONFIG.exists():
            cmd.extend(["--hardware-config", str(HW_CONFIG)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (
            f"tritonsim-hivm failed (returncode={result.returncode}): "
            f"{result.stderr[:300]}"
        )

        data = json.loads(out_file.read_text())
        ops = data.get("operations", data.get("nodes", []))
        vadds = [op for op in ops if op.get("name") == "vadd"]
        assert len(vadds) == 4, "DES should replay the four static loop iterations"
        assert data.get("loop_diagnostics", {}).get("resolved", 0) >= 1
        assert "Loops:" in result.stdout

    def test_unsigned_min_loop_bound_resolves(self, tmp_path):
        """Unsigned min bounds emitted by autoblockify remain concrete."""
        source = HIVM_ADD_KERNEL.read_text()
        source = source.replace(
            "  %c0 = arith.constant 0 : index\n",
            (
                "  %c0 = arith.constant 0 : index\n"
                "  %c1 = arith.constant 1 : index\n"
                "  %c2 = arith.constant 2 : index\n"
                "  %c4 = arith.constant 4 : index\n"
                "  %upper = arith.minui %c4, %c2 : index\n"
            ),
            1,
        )
        loop_start = (
            "  hivm.hir.vadd ins(%ub0, %ub1 : "
            "memref<1024xf32, #hivm.address_space<ub>>,"
        )
        source = source.replace(
            loop_start,
            "  scf.for %i = %c0 to %upper step %c1 {\n" + loop_start,
            1,
        )
        loop_end = (
            "      outs(%ub2 : memref<1024xf32, #hivm.address_space<ub>>)\n"
        )
        source = source.replace(loop_end, loop_end + "  }\n", 1)

        npuir_file = tmp_path / "hivm_add_unsigned_min_loop.npuir.mlir"
        npuir_file.write_text(source)
        out_file = tmp_path / "hivm_add_unsigned_min_loop_des.json"
        result = subprocess.run(
            [
                str(TRITONSIM_HIVM),
                "--npuir-file", str(npuir_file),
                "--scheduler", "des",
                "--des-graph-file", str(out_file),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr[:500]

        diagnostics = json.loads(out_file.read_text())["loop_diagnostics"]
        assert diagnostics["total"] == 1
        assert diagnostics["unresolved"] == 0
        assert diagnostics["loops"][0]["trip_count"] == 2

    def test_captured_bindings_use_user_argument_indices(self, tmp_path):
        """Triton-captured argN values skip hidden HACC launch arguments."""
        source = """\
func.func @binding_kernel(
    %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
    %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
    %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>},
    %arg3: memref<?xf16, #hivm.address_space<gm>>,
    %arg4: memref<?xf16, #hivm.address_space<gm>>,
    %arg5: f32,
    %arg6: memref<?xi64, #hivm.address_space<gm>>,
    %arg7: memref<?xi32, #hivm.address_space<gm>>,
    %arg8: i32,
    %arg9: i32,
    %arg10: i32,
    %arg11: i32) attributes {hacc.entry} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %0 = arith.muli %arg9, %arg10 : i32
  %1 = arith.muli %0, %arg11 : i32
  scf.for %i = %c0 to %1 step %c1 : i32 {
    hivm.hir.pipe_barrier[<PIPE_ALL>]
  }
  return
}
"""
        npuir_file = tmp_path / "captured_binding_indices.npuir.mlir"
        npuir_file.write_text(source)
        out_file = tmp_path / "captured_binding_indices_des.json"
        result = subprocess.run(
            [
                str(TRITONSIM_HIVM),
                "--npuir-file", str(npuir_file),
                "--scheduler", "des",
                "--des-graph-file", str(out_file),
                "--arg-bindings", "arg5=16,arg6=1,arg7=64,arg8=1",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr[:500]

        diagnostics = json.loads(out_file.read_text())["loop_diagnostics"]
        assert diagnostics["total"] == 1
        assert diagnostics["unresolved"] == 0
        assert diagnostics["loops"][0]["trip_count"] == 64

        actual_out_file = tmp_path / "actual_binding_indices_des.json"
        actual_result = subprocess.run(
            [
                str(TRITONSIM_HIVM),
                "--npuir-file", str(npuir_file),
                "--scheduler", "des",
                "--des-graph-file", str(actual_out_file),
                "--arg-bindings", "arg9=1,arg10=64,arg11=1",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert actual_result.returncode == 0, actual_result.stderr[:500]
        actual_diagnostics = json.loads(
            actual_out_file.read_text()
        )["loop_diagnostics"]
        assert actual_diagnostics["unresolved"] == 0
        assert actual_diagnostics["loops"][0]["trip_count"] == 64

    def test_outlined_vector_call_emits_conservative_work_summary(self, tmp_path):
        source = HIVM_ADD_KERNEL.read_text()
        marker = "  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]\n"
        call = (
            "  func.call @missing_outlined_vf(%ub0) "
            "{hivm.vector_function, no_inline} : "
            "(memref<1024xf32, #hivm.address_space<ub>>) -> ()\n"
        )
        assert marker in source
        npuir_file = tmp_path / "outlined_call.npuir.mlir"
        npuir_file.write_text(source.replace(marker, call + marker, 1))
        out_file = tmp_path / "outlined_call_des.json"

        result = subprocess.run(
            [
                str(TRITONSIM_HIVM),
                "--npuir-file", str(npuir_file),
                "--scheduler", "des",
                "--des-graph-file", str(out_file),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr[:500]

        data = json.loads(out_file.read_text())
        calls = [op for op in data["operations"] if op["name"] == "vcall"]
        assert len(calls) == 1
        assert calls[0]["pipe"] == "PIPE_V"
        assert calls[0]["elements"] == 1024
        assert calls[0]["duration"] > 0
        assert data["model_coverage"]["outlined_calls"] == 1
        assert data["model_coverage"]["summarized_outlined_calls"] == 1
        assert data["model_coverage"]["status"] == "conservative_partial"
        assert data["model_coverage"]["trace_timing_status"] == "partial"

    def test_unresolvable_loop_gets_sound_upper_bound_estimate(self, tmp_path):
        """A program-id/data-dependent scf.for bound (unresolvable to an
        exact constant) should still get a diagnostic-only structural upper
        bound when clamped by a resolvable compile-time constant — without
        that estimate ever leaking into the primary (sound) loop_multiplier
        actually applied to ops.
        """
        source = HIVM_ADD_KERNEL.read_text()
        # Add an unbound scalar arg standing in for a program-id-derived
        # value that can never resolve to a compile-time constant.
        source = source.replace(
            "                      %arg2: memref<?xf32, #hivm.address_space<gm>>) {",
            "                      %arg2: memref<?xf32, #hivm.address_space<gm>>,\n"
            "                      %arg3: i32) {",
            1,
        )
        source = source.replace(
            "  %c0 = arith.constant 0 : index\n",
            (
                "  %c0 = arith.constant 0 : index\n"
                "  %c1_i32 = arith.constant 1 : i32\n"
                "  %c0_i32 = arith.constant 0 : i32\n"
                "  %c16_i32 = arith.constant 16 : i32\n"
                "  %clamp_i32 = arith.addi %c0_i32, %c16_i32 : i32\n"
                "  %upper_i32 = arith.minsi %arg3, %clamp_i32 : i32\n"
            ),
            1,
        )
        loop_start = (
            "  hivm.hir.vadd ins(%ub0, %ub1 : memref<1024xf32, #hivm.address_space<ub>>,"
        )
        source = source.replace(
            loop_start,
            "  scf.for %i = %c0_i32 to %upper_i32 step %c1_i32 : i32 {\n" + loop_start,
            1,
        )
        loop_end = (
            "      outs(%ub2 : memref<1024xf32, #hivm.address_space<ub>>)\n"
        )
        source = source.replace(loop_end, loop_end + "  }\n", 1)

        npuir_file = tmp_path / "hivm_add_unresolvable_loop.npuir.mlir"
        npuir_file.write_text(source)
        out_file = tmp_path / "hivm_add_unresolvable_loop_des.json"
        cmd = [
            str(TRITONSIM_HIVM),
            "--npuir-file", str(npuir_file),
            "--scheduler", "des",
            "--des-graph-file", str(out_file),
        ]
        if CALIBRATED_HW_CONFIG.exists():
            cmd.extend(["--hardware-config", str(CALIBRATED_HW_CONFIG)])
        elif HW_CONFIG.exists():
            cmd.extend(["--hardware-config", str(HW_CONFIG)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (
            f"tritonsim-hivm failed (returncode={result.returncode}): "
            f"{result.stderr[:300]}"
        )

        data = json.loads(out_file.read_text())
        loop_diag = data.get("loop_diagnostics", {})
        assert loop_diag.get("unresolved", 0) >= 1, (
            "the arg3-clamped loop must NOT resolve to an exact trip count"
        )
        loops = loop_diag.get("loops", [])
        matching = [l for l in loops if not l.get("resolved", True)]
        assert matching, "expected at least one unresolved loop entry"
        loop = matching[0]
        assert loop["upper_bound_trip_count_estimate"] == 16, (
            "structural min(unresolvable, lower+16) should yield a sound "
            f"upper-bound estimate of 16, got {loop}"
        )
        assert loop["body_first_line"] > 0 and loop["body_last_line"] >= loop["body_first_line"]

        # Soundness check: the estimate must NOT leak into the primary
        # multiplier actually applied to ops inside the loop.
        ops = data.get("operations", data.get("nodes", []))
        vadds = [op for op in ops if op.get("name") == "vadd"]
        assert vadds, "expected the vadd op to be present in the schedule"
        assert all(op.get("loop_multiplier", 1) == 1 for op in vadds), (
            "the diagnostic upper-bound estimate must never feed the "
            "primary (sound) loop_multiplier"
        )

    def test_direct_semantic_scalar_like_ops_are_modeled(self, tmp_path):
        """Text fallback should not drop arith/affine/memref scalar work."""
        source = HIVM_ADD_KERNEL.read_text()
        marker = "  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]\n"
        scalar_block = (
            "  %c1_i32 = arith.constant 1 : i32\n"
            "  %c2_i32 = arith.constant 2 : i32\n"
            "  %s0 = arith.addi %c1_i32, %c2_i32 : i32\n"
            "  %s1 = arith.muli %s0, %c2_i32 : i32\n"
            "  %s2 = arith.cmpi slt, %s0, %s1 : i32\n"
            "  %s3 = arith.select %s2, %s0, %s1 : i32\n"
            "  %s4 = arith.index_cast %s3 : i32 to index\n"
            "  %agu = memref.reinterpret_cast %arg0 to offset: [%s4], sizes: [16], strides: [1]\n"
            "      : memref<?xf32, #hivm.address_space<gm>>\n"
            "      to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>\n"
        )
        assert marker in source
        npuir_file = tmp_path / "hivm_add_scalar_like.npuir.mlir"
        npuir_file.write_text(source.replace(marker, scalar_block + marker, 1))

        out_file = tmp_path / "hivm_add_scalar_like_des.json"
        cmd = [
            str(TRITONSIM_HIVM),
            "--npuir-file", str(npuir_file),
            "--scheduler", "des",
            "--des-graph-file", str(out_file),
        ]
        if CALIBRATED_HW_CONFIG.exists():
            cmd.extend(["--hardware-config", str(CALIBRATED_HW_CONFIG)])
        elif HW_CONFIG.exists():
            cmd.extend(["--hardware-config", str(HW_CONFIG)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (
            f"tritonsim-hivm failed (returncode={result.returncode}): "
            f"{result.stderr[:300]}"
        )

        data = json.loads(out_file.read_text())
        scalar_ops = [
            op for op in data.get("operations", [])
            if op.get("name") in {"addi", "muli", "cmpi", "select", "index_cast", "reinterpret_cast"}
        ]
        assert {op["name"] for op in scalar_ops} >= {
            "addi", "muli", "cmpi", "select", "index_cast", "reinterpret_cast"
        }
        assert all(op["pipe"] == "PIPE_S" for op in scalar_ops)
        assert sum(op["duration"] for op in scalar_ops) > 0
        assert all(op["calibrated_cost"] for op in scalar_ops)

        summary = data["calibration_summary"]
        assert summary["calibrated_ops"] >= len(scalar_ops)
        assert summary["heuristic_ops"] >= 0
        assert summary["by_subpipe"]["scalar_alu"]["ops"] >= 4
        assert summary["by_subpipe"]["agu"]["ops"] >= 2
        assert isinstance(summary["top_unclassified"], list)
        assert "sync_issue_cycles" in summary
        assert "sync_event_wait_cycles" in summary

        critical = data["critical_path_summary"]
        assert critical["cycles"] >= 0
        assert critical["issue_cycles"] >= 0
        assert critical["event_wait_cycles"] >= 0
        assert isinstance(critical["ops"], list)

        sync_ops = [
            op for op in data.get("operations", [])
            if op.get("name") in {"set_flag", "wait_flag"}
        ]
        assert sync_ops
        assert all(op["calibrated_cost"] for op in sync_ops)
        assert {op["cost_subpipe"] for op in sync_ops} == {"sync"}
        sync_durations = {}
        for op in sync_ops:
            sync_durations.setdefault(op["name"], set()).add(op["duration"])
            assert "event_wait_cycles" in op
        assert 1 in {
            op["dependency_latency"]
            for op in sync_ops
            if op["name"] == "set_flag"
        }
        assert all(
            op["issue_duration"] <= 32
            for op in sync_ops
            if op["name"] == "wait_flag"
        )


@requires_tritonsim_opt
@requires_fixtures
class TestTritonsimOptHIVMAnalysis:
    """Tests using tritonsim-opt --analyze-hivm with des-graph-file.

    All tests in this class are xfailed until Gap #1 (C++ HIVM parser fix) is resolved.
    The parser currently fails with "unsupported memory space Attribute" errors.
    """

    @pytest.mark.xfail(
        reason="Gap #1: C++ HIVM parser broken — needs bishengir build or text-parser extension",
        raises=AssertionError,
    )
    def test_des_graph_via_opt(self, tmp_path):
        """tritonsim-opt --analyze-hivm emits DES graph when option set."""
        out_file = tmp_path / "opt_des.json"
        opts_list = [f"des-graph-file={out_file}"]
        if HW_CONFIG.exists():
            opts_list.append(f"hardware-config={HW_CONFIG}")

        cmd = [
            str(TRITONSIM_OPT),
            str(HIVM_ADD_KERNEL),
            "--allow-unregistered-dialect",
            "--analyze-hivm=" + ",".join(opts_list),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (
            f"tritonsim-opt failed (returncode={result.returncode}): "
            f"{result.stderr[:300]}"
        )
        assert out_file.exists() and out_file.stat().st_size > 0, (
            "DES graph file was not emitted by tritonsim-opt"
        )

        data = json.loads(out_file.read_text())
        assert "operations" in data or "nodes" in data
        ops = data.get("operations", data.get("nodes", []))
        assert len(ops) > 0, "DES graph must contain at least one operation"
