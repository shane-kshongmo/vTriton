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
HW_CONFIG = PROJECT_ROOT / "configs" / "ascend_910b.json"
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
        assert 100 in {op["dependency_latency"] for op in sync_ops if op["name"] == "set_flag"}
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
