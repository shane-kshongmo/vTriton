# M5 — End-to-end report driver.
#
# Two explicit entry points:
#   report_from_npuir(npuir_path, grid, calib_db, hardware_config)
#     — runs tritonsim-hivm → des.json → extract → A.5 report
#   report_from_desgraph(des_json, grid, calib_db)
#     — consumes an existing des.json via extract_hivm
#
# The two configs are separate:
#   hardware_config (configs/ascend_910b.json) → C++ tool stage
#   calibration DB (load_default_calib_db)    → Python model stage
#
# CLI:
#   python -m perfbound.combine.run_report --desgraph /tmp/kda_des.json --grid 128,32
#
# Source spec: .omc/plans/a5_bound_combiner.md Change #8

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

from ..extract.hivm_extractor import extract_hivm, HIVMExtract
from ..extract.dsl_extractor import GridInfo
from ..calibration.calib_loader import load_default_calib_db
from ..calibration.constants import CalibrationDB
from ..model.bounds import compute_bounds
from .bound_combiner import combine, BoundResult
from .report import KernelReport
from .two_limit import compute_two_limit, TwoLimitResult


def _parse_grid(grid_str: str) -> tuple[int, ...]:
    """Parse a grid string like '128,32' into a tuple of ints."""
    parts = grid_str.strip().split(",")
    return tuple(int(p.strip()) for p in parts if p.strip())


def _build_grid_info(
    grid_dims: tuple[int, ...],
    n_cores: int = 20,
    occupancy: float = 1.0,
    load_balance: float = 1.0,
) -> GridInfo:
    """Build a minimal GridInfo from launch grid dimensions."""
    total_programs = 1
    for d in grid_dims:
        total_programs *= d
    return GridInfo(
        grid_dims=grid_dims,
        total_programs=total_programs,
        tile_assignment={},
        work={},
        occupancy=occupancy,
        load_balance=load_balance,
        redundancy=1.0,
        busiest_core_id=0,
    )


def report_from_desgraph(
    des_json: str | Path,
    grid_dims: tuple[int, ...],
    calib_db: Optional[CalibrationDB] = None,
    n_cores: int = 20,
    occupancy: float = 1.0,
    load_balance: float = 1.0,
    kernel_name: str = "unknown",
    t_measured_us: float | None = None,
    op_summary_csv: "str | Path | None" = None,
    op_name_filter: "str | None" = None,
) -> KernelReport:
    """Build a full A.5 report from an existing DES graph JSON.

    Args:
        des_json: Path to the DES graph JSON (from tritonsim-hivm --des-graph-file).
        grid_dims: Launch grid dimensions (e.g. (128, 32)).
        calib_db: Calibration DB.  Auto-loaded if None.
        n_cores: Number of cores.
        occupancy: Grid occupancy fraction.
        load_balance: Load balance fraction.
        kernel_name: Kernel label.

    Returns:
        KernelReport with bound, attribution, two-limit, and recommendation.
    """
    if calib_db is None:
        calib_db = load_default_calib_db()

    extract = extract_hivm(des_json)
    grid_info = _build_grid_info(grid_dims, n_cores, occupancy, load_balance)
    total_programs = grid_info.total_programs

    # Compute bound pieces
    pieces = compute_bounds(
        grid_info, extract, calib_db,
        n_cores=n_cores, total_programs=total_programs,
    )
    result = combine(
        pieces.grid, pieces.component, pieces.serial,
        kernel_name=kernel_name, extract=extract,
    )

    # Compute two-limit
    two_limit = compute_two_limit(
        kernel_name=kernel_name,
        grid_info=grid_info,
        extract=extract,
        calib_db=calib_db,
        t_bound_dsl_us=result.t_bound_us,
        t_measured_us=t_measured_us,
        n_cores=n_cores,
        total_programs=total_programs,
    )

    report = KernelReport.from_bound(result, two_limit=two_limit)

    if op_summary_csv is not None:
        _apply_csv_analysis(
            report, result, op_summary_csv,
            op_name_filter or kernel_name,
            des_json, calib_db,
        )

    return report


def report_from_npuir(
    npuir_path: str | Path,
    grid_dims: tuple[int, ...],
    calib_db: Optional[CalibrationDB] = None,
    hardware_config: str | Path | None = None,
    n_cores: int = 20,
    occupancy: float = 1.0,
    load_balance: float = 1.0,
    kernel_name: str = "unknown",
    tritonsim_hivm: str = "tritonsim-hivm",
    python_path: str = sys.executable,
    t_measured_us: float | None = None,
    op_summary_csv: "str | Path | None" = None,
    op_name_filter: "str | None" = None,
) -> KernelReport:
    """Build a full A.5 report by first running tritonsim-hivm on an NPU IR file.

    Args:
        npuir_path: Path to the .npuir.mlir file.
        grid_dims: Launch grid dimensions.
        calib_db: Calibration DB.  Auto-loaded if None.
        hardware_config: Path to hardware config JSON (for C++ tool).
        n_cores: Number of cores.
        occupancy: Grid occupancy fraction.
        load_balance: Load balance fraction.
        kernel_name: Kernel label.
        tritonsim_hivm: Path to tritonsim-hivm binary.
        python_path: Python interpreter for tritonsim-hivm.

    Returns:
        KernelReport.
    """
    import subprocess
    import tempfile

    npuir_path = Path(npuir_path)
    if not npuir_path.exists():
        raise FileNotFoundError(f"NPU IR file not found: {npuir_path}")

    # Run tritonsim-hivm to produce DES graph
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        des_path = f.name

    cmd = [
        tritonsim_hivm,
        "--npuir-file", str(npuir_path),
        "--des-graph-file", des_path,
    ]
    if hardware_config:
        cmd.extend(["--hardware-config", str(hardware_config)])
    # Pass python interpreter if compiling from .py (not from existing .npuir.mlir)
    if python_path:
        cmd.extend(["--python", python_path])

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"tritonsim-hivm failed: {e.stderr}"
        ) from e

    return report_from_desgraph(
        des_json=des_path,
        grid_dims=grid_dims,
        calib_db=calib_db,
        n_cores=n_cores,
        occupancy=occupancy,
        load_balance=load_balance,
        kernel_name=kernel_name,
        t_measured_us=t_measured_us,
        op_summary_csv=op_summary_csv,
        op_name_filter=op_name_filter,
    )


# ── CLI ───────────────────────────────────────────────────────────────────

def _cli():
    """CLI entry point for python -m perfbound.combine.run_report."""
    parser = argparse.ArgumentParser(
        description="Generate A.5 performance bound report from DES graph or NPU IR.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--desgraph", help="Path to existing DES graph JSON")
    group.add_argument("--npuir", help="Path to .npuir.mlir file (runs tritonsim-hivm)")

    parser.add_argument("--grid", required=True,
                        help="Launch grid dimensions (e.g. '128,32')")
    parser.add_argument("--cores", type=int, default=20,
                        help="Number of cores (default: 20)")
    parser.add_argument("--occupancy", type=float, default=1.0,
                        help="Grid occupancy fraction (default: 1.0)")
    parser.add_argument("--load-balance", type=float, default=1.0,
                        help="Load balance fraction (default: 1.0)")
    parser.add_argument("--kernel-name", default="kernel",
                        help="Kernel label")
    parser.add_argument("--hardware-config",
                        help="Path to hardware config JSON (for C++ tool)")
    parser.add_argument("--tritonsim-hivm", default="tritonsim-hivm",
                        help="Path to tritonsim-hivm binary")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter for tritonsim-hivm")
    parser.add_argument("--measured-us", type=float, default=None,
                        help="Measured kernel time in microseconds (from msprof)")
    parser.add_argument("--measured-csv", default=None,
                        help="Path to msprof op_summary CSV (extracts timing + component match)")
    parser.add_argument("--measured-op-name", default=None,
                        help="Op name filter for measured CSV (default: --kernel-name)")
    parser.add_argument("--output-json",
                        help="Write report JSON to this path")

    args = parser.parse_args()
    grid_dims = _parse_grid(args.grid)

    calib_db = load_default_calib_db()

    if args.desgraph:
        report = report_from_desgraph(
            des_json=args.desgraph,
            grid_dims=grid_dims,
            calib_db=calib_db,
            n_cores=args.cores,
            occupancy=args.occupancy,
            load_balance=args.load_balance,
            kernel_name=args.kernel_name,
            t_measured_us=args.measured_us,
            op_summary_csv=args.measured_csv,
            op_name_filter=args.measured_op_name,
        )
    else:
        report = report_from_npuir(
            npuir_path=args.npuir,
            grid_dims=grid_dims,
            calib_db=calib_db,
            hardware_config=args.hardware_config,
            n_cores=args.cores,
            occupancy=args.occupancy,
            load_balance=args.load_balance,
            kernel_name=args.kernel_name,
            tritonsim_hivm=args.tritonsim_hivm,
            python_path=args.python,
            t_measured_us=args.measured_us,
        )

    print(report.to_text())
    if args.output_json:
        report.to_json(args.output_json)
        print(f"\nJSON written to {args.output_json}")


def _apply_csv_analysis(
    report: KernelReport,
    bound_result: BoundResult,
    csv_path: "str | Path",
    op_name: str,
    des_json: "str | Path",
    calib_db: Optional[CalibrationDB],
    n_warmup: int = 1,
) -> None:
    """Run validation + profile utilization from op_summary CSV; merge into report."""
    import sys as _sys
    from ..validate.harness import ValidationCase, validate_from_csv, ValidationStatus
    from ..analyze.profile_utilization import run_from_files as _profile

    case = ValidationCase(
        kernel_name=report.kernel_name,
        profiler_op_name=op_name,
        bound_result=bound_result,
        csv_path=Path(csv_path),
        n_warmup=n_warmup,
    )
    vr = validate_from_csv(case)
    if vr.status not in (ValidationStatus.PASS, ValidationStatus.BOUND_VIOLATION):
        print(f"Warning: validation failed ({vr.notes})", file=_sys.stderr)
        return

    report.merge_validation(
        t_measured_us=vr.t_measured_us,
        msprof_source=vr.msprof_source,
        n_invocations=vr.n_invocations,
        component_match=vr.component_match,
    )

    try:
        profile_report = _profile(
            op_summary_path=csv_path,
            desgraph_path=des_json,
            calibration_path=None,
            kernel_name=op_name if op_name != report.kernel_name else None,
            t_bound_us=report.t_bound_dsl_us,
        )
        report.merge_profile(profile_report)
    except Exception as exc:
        print(f"Warning: profile utilization skipped: {exc}", file=_sys.stderr)


if __name__ == "__main__":
    _cli()
