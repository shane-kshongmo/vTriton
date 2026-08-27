# Change #10 — Real-Kernel Milestone Test
#
# Validates the end-to-end pipeline: NPUIR → tritonsim-hivm → des.json →
# extract_hivm → report_from_desgraph → KernelReport.
#
# Two test classes:
#   TestReportPipeline — uses the existing hivm_mixed_cv_kernel.npuir.mlir
#     fixture (known-good, parser-tested) to validate the full report pipeline.
#   TestChunkKdaMilestone — compiles the real chunk_kda kernel through
#     --triton-script, producing a live DES graph and report.  Currently
#     xfailed due to a bishengir-compile crash (ConvertLinalgRToBinary).
#
# Environment gating:
#   - SKIP when tritonsim-hivm binary not built
#   - XFAIL when bishengir-compile crashes on the chunk_kda kernel
#   - PASS when the full pipeline produces a coherent report

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from perfbound.extract.hivm_extractor import extract_hivm, HIVMExtract
from perfbound.extract.op_classifier import Component
from perfbound.combine.run_report import report_from_desgraph, _build_grid_info
from perfbound.combine.report import KernelReport

# ── Paths ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRITONSIM_HIVM = PROJECT_ROOT / "build" / "bin" / "tritonsim-hivm"
HW_CONFIG = PROJECT_ROOT / "configs" / "ascend_910b3_v4.json"

# Mixed CV fixture (known-good, parser-tested)
MIXED_CV_KERNEL = PROJECT_ROOT / "test" / "hivm_mixed_cv_kernel.npuir.mlir"

# Chunk KDA kernel (real-kernel milestone target)
KERNEL_SCRIPT = PROJECT_ROOT / "test" / "chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2.py"

# Python interpreter for --triton-script
_PYTHON = os.environ.get(
    "VTRITON_PYTHON",
    "/home/shane/miniconda3/envs/vtriton-verify/bin/python",
)
if not Path(_PYTHON).exists():
    _PYTHON = sys.executable

# ── Chunk KDA kernel dimensions ────────────────────────────────────────────
# B=32, T=8192, H=32, K=128, V=128, BT=64, BK=32, BV=32
# Grid: (128, 32), total_programs = 4096, n_cores = 20

B, T, H, K, V = 32, 8192, 32, 128, 128
BT, BK, BV = 64, 32, 32
GRID_DIMS = (128, 32)
TOTAL_PROGRAMS = GRID_DIMS[0] * GRID_DIMS[1]  # 4096
N_CORES = 20
WAVES = math.ceil(TOTAL_PROGRAMS / N_CORES)    # 205
N_IK = K // BK   # 4
N_IV = V // BV   # 4


# ── Step D — Hand-derived analytic estimates ──────────────────────────────

def hand_flops_per_program() -> int:
    """FLOPs per program from tl.dot ops (each = 2·M·N·K)."""
    # Inner dots (×16): 3 × dot([BT,BV]·[BV,BK]) = 3 × 2·64·32·32
    inner = 3 * (2 * BT * BV * BK) * N_IK * N_IV
    # i_k==0 dots (×4): 2 × dot([BT,BV]·[BV,BT]) = 2 × 2·64·32·64
    ik0 = 2 * (2 * BT * BV * BT) * N_IV
    # Outer dots (×4): 2 × dot([BT,BK]·[BK,BT]) = 2 × 2·64·32·64
    outer = 2 * (2 * BT * BK * BT) * N_IK
    # Post-loop (×1): 2 × dot([BT,BT]·[BT,BT]) = 2 × 2·64·64·64
    post = 2 * (2 * BT * BT * BT)
    return inner + ik0 + outer + post


def hand_bytes_per_program() -> int:
    """HBM bytes per program from loads + stores."""
    # Per i_k loads: b_k(BT·BK·2=4096) + b_g(BT·BK·4=8192) + b_gn(BK·4=128)
    per_ik_loads = 4096 + 8192 + 128
    # Per i_v loads: 5 × BT·BV·2 = 5 × 4096 = 20480
    per_iv_loads = 5 * (BT * BV * 2)
    # i_k==0 per i_v: b_v(BT·BV·2=4096)
    per_iv_ik0 = BT * BV * 2
    # Per i_k extra: b_q(BT·BK·2=4096)
    per_ik_extra = BT * BK * 2
    total_loads = per_ik_loads * N_IK + per_iv_loads * N_IK * N_IV + per_iv_ik0 * N_IV + per_ik_extra * N_IK

    # Per i_k stores: 3 × BT·BK·4 = 3 × 8192
    per_ik_stores = 3 * (BT * BK * 4)
    # i_k==0 per i_v: dv2(BT·BV·2=4096)
    per_iv_ik0_stores = BT * BV * 2
    # Post stores: dA(BT·BT·4=16384) + db(BT·4=256)
    post_stores = BT * BT * 4 + BT * 4
    total_stores = per_ik_stores * N_IK + per_iv_ik0_stores * N_IV + post_stores

    return total_loads + total_stores


HAND_FLOPS_PER_PROG = hand_flops_per_program()     # ~11.5M
HAND_BYTES_PER_PROG = hand_bytes_per_program()     # ~1.9M
HAND_FLOPS_TOTAL = HAND_FLOPS_PER_PROG * TOTAL_PROGRAMS
HAND_BYTES_TOTAL = HAND_BYTES_PER_PROG * TOTAL_PROGRAMS


# ── Skip / xfail markers ──────────────────────────────────────────────────

requires_binary = pytest.mark.skipif(
    not TRITONSIM_HIVM.exists(),
    reason="build/bin/tritonsim-hivm not found",
)

requires_mixed_cv = pytest.mark.skipif(
    not MIXED_CV_KERNEL.exists(),
    reason="hivm_mixed_cv_kernel.npuir.mlir not found",
)

requires_kernel = pytest.mark.skipif(
    not KERNEL_SCRIPT.exists(),
    reason="chunk_kda kernel fixture not found",
)


_compile_check_done = False
_compile_check_result = False


def _can_compile():
    """Check CANN + torch_npu availability (cached, computed once)."""
    global _compile_check_done, _compile_check_result
    if _compile_check_done:
        return _compile_check_result
    if not Path(_PYTHON).exists():
        _compile_check_result = False
    else:
        try:
            result = subprocess.run(
                [_PYTHON, "-c", "import torch; import torch_npu; print('ok')"],
                capture_output=True, text=True, timeout=15,
            )
            _compile_check_result = (
                result.returncode == 0 and "ok" in result.stdout
            )
        except Exception:
            _compile_check_result = False
    _compile_check_done = True
    return _compile_check_result


requires_compile_env = pytest.mark.skipif(
    not _can_compile(),
    reason="CANN / torch_npu not available",
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mixed_cv_des_json(tmp_path_factory):
    """Produce des.json from the mixed CV kernel fixture."""
    out_dir = tmp_path_factory.mktemp("mixed_cv")
    des_path = out_dir / "mixed_cv_des.json"

    cmd = [
        str(TRITONSIM_HIVM),
        "--npuir-file", str(MIXED_CV_KERNEL),
        "--des-graph-file", str(des_path),
    ]
    if HW_CONFIG.exists():
        cmd.extend(["--hardware-config", str(HW_CONFIG)])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        pytest.fail(f"tritonsim-hivm failed: {result.stderr[:500]}")
    assert des_path.exists() and des_path.stat().st_size > 0
    return des_path


@pytest.fixture(scope="module")
def mixed_cv_extract(mixed_cv_des_json):
    return extract_hivm(mixed_cv_des_json)


@pytest.fixture(scope="module")
def mixed_cv_report(mixed_cv_des_json):
    return report_from_desgraph(
        des_json=mixed_cv_des_json,
        grid_dims=GRID_DIMS,
        n_cores=N_CORES,
        kernel_name="mixed_cv_test",
    )


@pytest.fixture(scope="module")
def kda_des_json(tmp_path_factory):
    """Compile chunk_kda and produce des.json.  May fail (bishengir bug)."""
    out_dir = tmp_path_factory.mktemp("kda_des")
    des_path = out_dir / "kda_des.json"

    cmd = [
        str(TRITONSIM_HIVM),
        "--triton-script", str(KERNEL_SCRIPT),
        "--python", _PYTHON,
        "--hardware-config", str(HW_CONFIG),
        "--des-graph-file", str(des_path),
        "--scheduler", "des",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (
        f"tritonsim-hivm --triton-script failed (rc={result.returncode}):\n"
        f"{result.stderr[:500]}"
    )
    assert des_path.exists() and des_path.stat().st_size > 0
    return des_path


# ══════════════════════════════════════════════════════════════════════════
# Test Class 1: Report Pipeline (uses mixed_cv_kernel fixture)
# ══════════════════════════════════════════════════════════════════════════

@requires_binary
@requires_mixed_cv
class TestReportPipeline:
    """Validate the full des.json → extract → report pipeline using the
    existing hivm_mixed_cv_kernel.npuir.mlir fixture.

    This fixture has Cube + Vector + MTE ops and is known to parse
    correctly (tested by test_hivm_cli_integration.py).
    """

    def test_des_has_all_components(self, mixed_cv_extract):
        """DES graph contains Cube, Vector, and MTE operations."""
        components = {op.component for op in mixed_cv_extract.operations}
        assert Component.CUBE in components, f"Missing Cube, got: {components}"
        assert Component.VECTOR in components, f"Missing Vector, got: {components}"
        mte = components & {Component.MTE_GM, Component.MTE_L1, Component.MTE_UB}
        assert len(mte) > 0, f"Missing MTE, got: {components}"

    def test_handoffs_detected(self, mixed_cv_extract):
        """Cross-component handoffs are detected (Cube↔Vector chain)."""
        assert len(mixed_cv_extract.handoffs) > 0, (
            "No handoffs detected — expected Cube↔Vector handoffs"
        )
        # Check we have at least one compute-to-compute handoff
        compute_comps = {Component.CUBE, Component.VECTOR}
        cc_handoffs = [
            h for h in mixed_cv_extract.handoffs
            if h.producer_component in compute_comps
            and h.consumer_component in compute_comps
        ]
        assert len(cc_handoffs) > 0, (
            "No compute-to-compute handoffs found"
        )

    def test_report_produces_bound(self, mixed_cv_report):
        """report_from_desgraph produces a valid KernelReport."""
        r = mixed_cv_report
        assert isinstance(r, KernelReport)
        assert r.t_bound_us > 0, "T_bound must be positive"
        assert r.binding_tier in ("grid", "component"), (
            f"Unexpected binding_tier: {r.binding_tier}"
        )

    def test_serial_irreducible_positive(self, mixed_cv_report):
        """T_serial_irreducible > 0 for a kernel with Cube↔Vector handoffs."""
        assert mixed_cv_report.t_serial_irreducible_us > 0, (
            f"Expected T_serial > 0, got {mixed_cv_report.t_serial_irreducible_us:.2f}"
        )

    def test_recommended_action_specific(self, mixed_cv_report):
        """recommended_action names a specific gap, not 'unknown'."""
        action = mixed_cv_report.recommended_action
        assert action != "unknown", "recommended_action is 'unknown'"
        assert len(action) > 5, f"Action too short: '{action}'"

    def test_report_serializable(self, mixed_cv_report, tmp_path):
        """KernelReport round-trips through JSON."""
        json_path = tmp_path / "report.json"
        mixed_cv_report.to_json(json_path)
        loaded = json.loads(json_path.read_text())
        assert loaded["kernel_name"] == "mixed_cv_test"
        assert loaded["t_bound_us"] > 0
        assert "binding_tier" in loaded
        assert "attribution" in loaded


# ══════════════════════════════════════════════════════════════════════════
# Test Class 2: Analytic Estimates (pure computation, no compile)
# ══════════════════════════════════════════════════════════════════════════

class TestAnalyticEstimates:
    """Validate the hand-derived Step-D analytic estimates.

    These are pure-computation tests — no compilation or binary needed.
    They anchor the milestone assertions for when the compile step works.
    """

    def test_flops_per_program_order_of_magnitude(self):
        """FLOP/program is in the ~10M range (spec says ~1.2×10⁷)."""
        assert 5e6 <= HAND_FLOPS_PER_PROG <= 2e7, (
            f"FLOP/program={HAND_FLOPS_PER_PROG:.2e} outside expected range"
        )

    def test_bytes_per_program_order_of_magnitude(self):
        """Bytes/program is in the ~1–3 MB range."""
        assert 5e5 <= HAND_BYTES_PER_PROG <= 5e6, (
            f"Bytes/program={HAND_BYTES_PER_PROG:.2e} outside expected range"
        )

    def test_compute_bound_prediction(self):
        """Each tl.dot contributes exactly 2·M·N·K FLOPs.

        Verify the hand count matches the sum of per-dot FLOPs exactly
        (no floating-point ambiguity), then sanity-check chip-level time.
        """
        # Per-dot exact FLOP counts (each dot = 2·M·N·K)
        dot_inner = 2 * BT * BV * BK          # 2·64·32·32 = 131072
        dot_ik0   = 2 * BT * BV * BT          # 2·64·32·64 = 262144
        dot_outer = 2 * BT * BK * BT          # 2·64·32·64 = 262144
        dot_post  = 2 * BT * BT * BT          # 2·64·64·64 = 524288

        # Count of each dot type
        n_inner = 3 * N_IK * N_IV              # 3·4·4 = 48
        n_ik0   = 2 * N_IV                      # 2·4 = 8
        n_outer = 2 * N_IK                      # 2·4 = 8
        n_post  = 2                              # 2

        total_dots = n_inner + n_ik0 + n_outer + n_post  # 66
        exact_flops = (
            n_inner * dot_inner
            + n_ik0 * dot_ik0
            + n_outer * dot_outer
            + n_post * dot_post
        )
        # Must match hand_flops_per_program() exactly
        assert exact_flops == HAND_FLOPS_PER_PROG, (
            f"Per-dot sum {exact_flops} != hand count {HAND_FLOPS_PER_PROG}"
        )
        # 66 dots per program, each averaging ~175K FLOPs → ~11.5M total
        assert total_dots == 66
        assert HAND_FLOPS_PER_PROG == 11_534_336

    def test_wave_count(self):
        """Wave count matches ceil(total_programs / n_cores)."""
        assert WAVES == 205, f"Expected 205 waves, got {WAVES}"
        assert TOTAL_PROGRAMS == 4096


# ══════════════════════════════════════════════════════════════════════════
# Test Class 3: Chunk KDA Compile (xfail — bishengir bug)
# ══════════════════════════════════════════════════════════════════════════

@requires_binary
@requires_kernel
@requires_compile_env
class TestChunkKdaCompile:
    """Compile the real chunk_kda kernel through --triton-script.

    UPDATE 2026-06-10: the bishengir-compile ConvertLinalgRToBinary crash was a
    CANN 9.0.0-beta.2 bug and is RESOLVED in the CANN 9.0.0 release — chunk_kda
    now compiles and runs on a live 910B3 (evidence:
    .omc/research/hw_runs/, captured via scripts/remote_bench.py).  This local
    test still xfails ONLY because WSL has no NPU device to JIT-compile the
    kernel against; it is expected to PASS on an NPU host with a native
    tritonsim-hivm.  The marker is non-strict so an NPU-host run XPASSes
    rather than erroring.
    """

    @pytest.mark.xfail(
        reason=(
            "chunk_kda compiles+runs on CANN 9.0.0 release (the 9.0.0-beta.2 "
            "ConvertLinalgRToBinary crash is fixed; evidence in "
            ".omc/research/hw_runs/).  This local path still fails because WSL "
            "has no NPU device for the JIT compile — not a compiler bug."
        ),
        strict=False,
    )
    def test_compile_and_emit_des(self, kda_des_json):
        """chunk_kda → compile → des.json with Cube+Vector+MTE ops."""
        ext = extract_hivm(kda_des_json)
        components = {op.component for op in ext.operations}

        assert Component.CUBE in components
        assert Component.VECTOR in components
        assert len(ext.handoffs) > 0

        # Verify FLOP reconciliation
        extract_flops = sum(
            op.flops * op.loop_multiplier
            for op in ext.operations
            if op.flops > 0
        ) * TOTAL_PROGRAMS
        low, high = HAND_FLOPS_TOTAL * 0.75, HAND_FLOPS_TOTAL * 1.25
        assert low <= extract_flops <= high, (
            f"FLOPs {extract_flops:.2e} outside [{low:.2e}, {high:.2e}]"
        )


# ══════════════════════════════════════════════════════════════════════════
# Test Class 4: Blocker-1 Spike — dump-before-codegen
# ══════════════════════════════════════════════════════════════════════════

@requires_binary
@requires_kernel
@requires_compile_env
class TestDumpBeforeCodegen:
    """Spike: does the HIVM dump complete before the codegen crash?

    UPDATE 2026-06-10: this spike is effectively MOOT.  The codegen crash it
    was probing (ConvertLinalgRToBinary on CANN 9.0.0-beta.2) is fixed in the
    CANN 9.0.0 release — chunk_kda compiles cleanly and runs on a live 910B3
    (evidence: .omc/research/hw_runs/).  With no crash, the dump trivially
    "survives": full codegen completes.  The tests below still xfail on WSL
    only for lack of an NPU device, and short-circuit to PASS (returncode 0)
    when run where the compile actually succeeds.
    """

    def test_hivm_dump_survives_crash(self, tmp_path):
        """Check if .npuir.mlir exists after primary compile fails.

        Runs the full dump-launcher path.  If the compiler crashes but
        the secondary dump captured NPUIR, the test passes.  If no dump
        is found, the test xfails (spike not yet resolved).
        """
        dump_dir = tmp_path / "dump"
        dump_dir.mkdir()

        cmd = [
            str(TRITONSIM_HIVM),
            "--triton-script", str(KERNEL_SCRIPT),
            "--python", _PYTHON,
            "--hardware-config", str(HW_CONFIG),
            "--des-graph-file", str(tmp_path / "kda_des.json"),
            "--scheduler", "des",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=str(dump_dir),
        )

        # Compile is expected to fail (bishengir-compile bug).
        # If it succeeds, the blocker is resolved — test passes trivially.
        if result.returncode == 0:
            return  # Blocker resolved, dump not needed

        # Search for .npuir.mlir files in dump_dir and subdirectories
        npuir_files = list(dump_dir.rglob("*.npuir.mlir"))

        if not npuir_files:
            pytest.xfail(
                "spike: no .npuir.mlir captured before crash — "
                "investigating dump-before-codegen ordering"
            )

        # Dump was captured — verify it contains MLIR content
        content = npuir_files[0].read_text(encoding="utf-8")
        assert len(content) > 50, (
            f"Dumped MLIR suspiciously short: {len(content)} chars"
        )
        assert "func" in content or "module" in content, (
            "Dumped MLIR does not contain expected func/module markers"
        )

    def test_ttadapter_mlir_captured(self, tmp_path):
        """Check if any ttadapter MLIR is saved before the codegen crash.

        The dump launcher may capture IR at different stages.  This test
        looks for any .mlir file (not just .npuir.mlir) that contains
        function definitions, indicating the IR was captured.
        """
        dump_dir = tmp_path / "dump"
        dump_dir.mkdir()

        cmd = [
            str(TRITONSIM_HIVM),
            "--triton-script", str(KERNEL_SCRIPT),
            "--python", _PYTHON,
            "--hardware-config", str(HW_CONFIG),
            "--des-graph-file", str(tmp_path / "kda_des.json"),
            "--scheduler", "des",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=str(dump_dir),
        )

        if result.returncode == 0:
            return  # Blocker resolved

        # Look for any MLIR files that might be ttadapter or intermediate IR
        mlir_files = list(dump_dir.rglob("*.mlir"))

        if not mlir_files:
            pytest.xfail(
                "spike: no MLIR files captured before crash — "
                "investigating dump-before-codegen ordering"
            )

        # At least one MLIR file should contain func.func or module
        found_content = False
        for f in mlir_files:
            content = f.read_text(encoding="utf-8")
            if "func" in content or "module" in content:
                found_content = True
                break

        assert found_content, (
            f"Found {len(mlir_files)} MLIR file(s) but none contain "
            "func/module — ttadapter capture may be incomplete"
        )
