#!/usr/bin/env python3
"""Apply optimisation suggestions to kernel_001.npuir.mlir via hivm-optimize.

This script first performs a text-level analysis of the MLIR to identify
bottleneck patterns, then invokes the hivm-optimize binary (which uses
HivmOpsEditor C++ API) to apply the actual transformations.

Optimisation points (from bottleneck diagnosis of kernel_001):

  1. Redundant pipe_barrier[<PIPE_V>] in AIV inner loop (23 ops)
  2. pipe_barrier[<PIPE_ALL>] at kernel exit (2 ops) — suggestion only
  3. vdiv → vmul replacement (2 ops)
  4. Mask chain fusion (16 vcmp/vnot/vcast/vbrc ops) — suggestion only
  5. L1 double-buffering for nd2nz+mmadL1 — suggestion only
  6. Reduce GM round-trips (4 load/store ops) — suggestion only

Usage:
  python3 scripts/optimize_kernel_001.py [--input PATH] [--output PATH] [--dry-run]
  python3 scripts/optimize_kernel_001.py --no-remove-pipe-v-barriers
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HIVM_OPTIMIZE = REPO_ROOT / "build" / "bin" / "hivm-optimize"
DEFAULT_INPUT = REPO_ROOT / "test" / "kernel_001.npuir.mlir"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "kernel_001_optimized.npuir.mlir"


def read_mlir(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def count_ops(mlir_text: str, op_pattern: str) -> int:
    return len(re.findall(op_pattern, mlir_text))


def find_line_numbers(mlir_text: str, op_pattern: str) -> list[int]:
    lines = mlir_text.split("\n")
    results = []
    for i, line in enumerate(lines, 1):
        if re.search(op_pattern, line):
            results.append(i)
    return results


def analyze_optimizations(mlir_text: str) -> list[dict]:
    optimizations = []

    pipe_v_barriers = find_line_numbers(mlir_text, r"pipe_barrier\[<PIPE_V\>\]")
    if len(pipe_v_barriers) > 5:
        optimizations.append({
            "id": 1,
            "type": "remove_pipe_barrier",
            "description": f"AIV inner loop has {len(pipe_v_barriers)} pipe_barrier[<PIPE_V>] blocking Vector pipeline",
            "lines": pipe_v_barriers,
            "crud_action": "actionable",
            "evidence": f"pipe_barrier[<PIPE_V>] count={len(pipe_v_barriers)}; each stalls PIPE_V, preventing pipelined issue",
            "suggestion": "Remove pipe_barrier[<PIPE_V>] inside scf.for loops; rely on set_flag/wait_flag for cross-pipe sync",
        })

    pipe_all_barriers = find_line_numbers(mlir_text, r"pipe_barrier\[<PIPE_ALL\>\]")
    if pipe_all_barriers:
        optimizations.append({
            "id": 2,
            "type": "replace_pipe_all_barrier",
            "description": f"Found {len(pipe_all_barriers)} pipe_barrier[<PIPE_ALL>] at lines {pipe_all_barriers}",
            "lines": pipe_all_barriers,
            "crud_action": "suggest",
            "evidence": "pipe_barrier[<PIPE_ALL>] blocks all pipes simultaneously",
            "suggestion": "Replace PIPE_ALL barrier with per-pipe set_flag/wait_flag pairs",
        })

    vdiv_lines = find_line_numbers(mlir_text, r"hivm\.hir\.vdiv")
    if vdiv_lines:
        optimizations.append({
            "id": 3,
            "type": "replace_vdiv_with_vmul",
            "description": f"Found {len(vdiv_lines)} vdiv ops at lines {vdiv_lines}",
            "lines": vdiv_lines,
            "crud_action": "actionable",
            "evidence": f"vdiv throughput ~4x lower than vmul on AIV; {len(vdiv_lines)} vdiv ops",
            "suggestion": "Replace vdiv(a,b) with vmul(a, vrec(b))",
        })

    mask_chain_ops = (
        count_ops(mlir_text, r"hivm\.hir\.vcmp")
        + count_ops(mlir_text, r"hivm\.hir\.vnot")
        + count_ops(mlir_text, r"hivm\.hir\.vcast")
        + count_ops(mlir_text, r"hivm\.hir\.vbrc")
    )
    if mask_chain_ops > 10:
        optimizations.append({
            "id": 4,
            "type": "suggest_mask_fusion",
            "description": f"Mask computation uses {mask_chain_ops} ops (vcmp+vnot+vcast+vbrc chain)",
            "lines": [],
            "crud_action": "suggest",
            "evidence": f"mask_chain ops={mask_chain_ops}; total startup overhead={mask_chain_ops * 35} cyc",
            "suggestion": "Fuse mask computation chain into a single macro op",
        })

    nd2nz_lines = find_line_numbers(mlir_text, r"hivm\.hir\.nd2nz")
    mmad_lines = find_line_numbers(mlir_text, r"hivm\.hir\.mmadL1")
    if nd2nz_lines and mmad_lines:
        optimizations.append({
            "id": 5,
            "type": "suggest_double_buffer",
            "description": f"Found {len(nd2nz_lines)} nd2nz + {len(mmad_lines)} mmadL1",
            "lines": nd2nz_lines + mmad_lines[:3],
            "crud_action": "suggest",
            "evidence": "nd2nz(GM→L1) then mmadL1(Cube); sequential with wait_flag sync",
            "suggestion": "Double-buffer L1 buffers so MTE2 prefetches next tile while Cube computes",
        })

    store_lines = find_line_numbers(mlir_text, r"hivm\.hir\.store")
    load_lines = find_line_numbers(mlir_text, r"hivm\.hir\.load\b")
    gm_transfers = len(store_lines) + len(load_lines)
    if gm_transfers > 2:
        optimizations.append({
            "id": 6,
            "type": "suggest_reduce_gm_trips",
            "description": f"Found {len(store_lines)} store + {len(load_lines)} load = {gm_transfers} GM round-trips",
            "lines": store_lines + load_lines,
            "crud_action": "suggest",
            "evidence": f"Total GM transfer ops={gm_transfers}; each pays MTE2/MTE3 startup",
            "suggestion": "Reduce GM round-trips via in-place computation or tiling reuse",
        })

    return optimizations


def run_hivm_optimize(input_path: str, output_path: str,
                      remove_pipe_v: bool = True,
                      replace_vdiv: bool = True,
                      remove_gm_trips: int = 0,
                      fuse_compute: bool = False) -> subprocess.CompletedProcess | None:
    if not HIVM_OPTIMIZE.exists():
        return None

    cmd = [
        str(HIVM_OPTIMIZE),
        "--input", input_path,
        "--output", output_path,
        "--remove-pipe-v-barriers", str(int(remove_pipe_v)),
        "--replace-vdiv-with-vmul", str(int(replace_vdiv)),
        "--remove-gm-trips", str(remove_gm_trips),
        "--fuse-consecutive-compute", str(int(fuse_compute)),
        "--verbose", "1",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30)


def main():
    parser = argparse.ArgumentParser(
        description="Apply optimisation to kernel_001.npuir.mlir via hivm-optimize")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="Input MLIR file path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output MLIR file path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyze only, do not modify files")
    parser.add_argument("--no-remove-pipe-v-barriers", action="store_true",
                        help="Skip removing pipe_barrier[PIPE_V]")
    parser.add_argument("--no-replace-vdiv", action="store_true",
                        help="Skip replacing vdiv with vmul")
    parser.add_argument("--remove-gm-trips", type=int, default=0,
                        help="Remove N redundant GM round-trips")
    parser.add_argument("--fuse-consecutive-compute", action="store_true",
                        help="Fuse consecutive vadd chains")
    args = parser.parse_args()

    mlir_text = read_mlir(args.input)
    optimizations = analyze_optimizations(mlir_text)

    print("=" * 60)
    print("kernel_001.npuir.mlir — Bottleneck Optimization Analysis")
    print("=" * 60)

    actionable = 0
    suggestions = 0
    for opt in optimizations:
        tag = "ACTIONABLE" if opt["crud_action"] != "suggest" else "SUGGESTION"
        print(f"\n[{tag}] #{opt['id']} {opt['type']}")
        print(f"  Description: {opt['description']}")
        print(f"  Evidence: {opt['evidence']}")
        print(f"  Suggestion: {opt['suggestion']}")
        if opt["lines"]:
            print(f"  Lines: {opt['lines'][:10]}{'...' if len(opt['lines']) > 10 else ''}")
        if opt["crud_action"] != "suggest":
            actionable += 1
        else:
            suggestions += 1

    print(f"\n--- Summary: {actionable} actionable, {suggestions} suggestion-only ---")

    if args.dry_run:
        print("\n[dry-run] No modifications applied.")
        return 0

    os.makedirs(os.path.dirname(str(DEFAULT_OUTPUT)), exist_ok=True)

    remove_pipe_v = not args.no_remove_pipe_v_barriers
    replace_vdiv = not args.no_replace_vdiv

    print("\n[Applying transformations via hivm-optimize]")
    result = run_hivm_optimize(
        args.input, args.output,
        remove_pipe_v=remove_pipe_v,
        replace_vdiv=replace_vdiv,
        remove_gm_trips=args.remove_gm_trips,
        fuse_compute=args.fuse_consecutive_compute,
    )

    if result is None:
        print("  hivm-optimize binary not found (BISHENGIR_HIVM not built)")
        print("  Cannot apply C++ transformations — build with BISHENGIR_HIVM enabled")
        print("  or use --dry-run for analysis-only mode")
        return 1

    if result.returncode != 0:
        print(f"  hivm-optimize failed with exit code {result.returncode}")
        print(f"  stderr: {result.stderr}")
        return 1

    print(f"  {result.stdout}")

    # Write diagnosis report
    report_path = Path(args.output).parent / "kernel_001_diagnosis.txt"
    with open(str(report_path), "w") as f:
        f.write("kernel_001.npuir.mlir — Optimization Diagnosis Report\n")
        f.write("=" * 60 + "\n\n")
        for opt in optimizations:
            f.write(f"#{opt['id']} [{opt['crud_action']}] {opt['type']}\n")
            f.write(f"  Evidence:    {opt['evidence']}\n")
            f.write(f"  Suggestion:  {opt['suggestion']}\n")
            f.write(f"  Lines:       {opt['lines'][:20]}\n\n")

        f.write("\nApplied transformations (via hivm-optimize + HivmOpsEditor):\n")
        if remove_pipe_v:
            f.write("  - Removed pipe_barrier[<PIPE_V>] inside scf.for loops\n")
        if replace_vdiv:
            f.write("  - Replaced vdiv with vmul (placeholder: same src/dst)\n")
        if args.remove_gm_trips > 0:
            f.write(f"  - Removed {args.remove_gm_trips} redundant GM round-trips\n")
        if args.fuse_consecutive_compute:
            f.write("  - Fused consecutive vadd chains\n")

    print(f"\nDiagnosis report: {report_path}")
    print(f"Optimized MLIR:   {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
