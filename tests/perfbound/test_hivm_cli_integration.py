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
