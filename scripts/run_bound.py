#!/usr/bin/env python3
"""Stage-B bound pipeline.

Pipeline:
  kernel.py -> Triton NPUIR dump -> cleaned NPUIR -> DES graph JSON ->
  ``perfbound.combine.report_from_desgraph`` JSON report.

The pure command-building helpers are used by CI; the CLI performs real local
execution only when not run with ``--dry-run``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perfbound.calibration.calib_loader import load_calibration
from perfbound.combine.run_report import report_from_npuir
from perfbound.experiments.artifacts import STAGEB_ROOT
from perfbound.experiments.registry import get_kernel


@dataclass(frozen=True)
class BoundPipelinePlan:
    kernel_name: str
    kernel_script: Path
    output_dir: Path
    dump_dir: Path
    raw_npuir: Path
    clean_npuir: Path
    des_json: Path
    report_json: Path
    grid_dims: tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "kernel_name": self.kernel_name,
            "kernel_script": str(self.kernel_script),
            "output_dir": str(self.output_dir),
            "dump_dir": str(self.dump_dir),
            "raw_npuir": str(self.raw_npuir),
            "clean_npuir": str(self.clean_npuir),
            "des_json": str(self.des_json),
            "report_json": str(self.report_json),
            "grid_dims": list(self.grid_dims),
        }


def parse_grid(grid: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(grid, str):
        dims = tuple(int(part.strip()) for part in grid.split(",") if part.strip())
    else:
        dims = tuple(int(v) for v in grid)
    if not dims or any(v <= 0 for v in dims):
        raise ValueError(f"invalid grid dimensions: {grid!r}")
    return dims


def build_dump_env(dump_dir: str | Path, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Environment used to force a Triton NPUIR dump."""
    env = dict(base_env or os.environ)
    env.update(
        {
            "TRITON_DEBUG": "1",
            "TRITON_KERNEL_DUMP": "1",
            "TRITON_DUMP_DIR": str(dump_dir),
            "TRITON_ALWAYS_COMPILE": "1",
        }
    )
    return env


def newest_npuir(paths: Iterable[str | Path]) -> Path:
    """Return the most recently modified ``*.npuir.mlir`` under the given paths."""
    candidates: list[Path] = []
    for root in paths:
        p = Path(root)
        if p.is_file() and p.name.endswith(".npuir.mlir"):
            candidates.append(p)
        elif p.exists():
            candidates.extend(p.rglob("*.npuir.mlir"))
    if not candidates:
        raise FileNotFoundError("no .npuir.mlir file found")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def dumped_npuir(
    dump_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Select a kernel dump, preferring the run-local dump directory."""
    try:
        return newest_npuir((dump_dir,))
    except FileNotFoundError:
        if cache_dir is None:
            raise
        return newest_npuir((cache_dir,))


def build_tritonsim_command(
    *,
    tritonsim_hivm: str | Path,
    npuir_path: str | Path,
    des_json: str | Path,
    hardware_config: str | Path | None = None,
    python_path: str | Path | None = None,
) -> list[str]:
    cmd = [
        str(tritonsim_hivm),
        "--npuir-file",
        str(npuir_path),
        "--des-graph-file",
        str(des_json),
    ]
    if hardware_config:
        cmd.extend(["--hardware-config", str(hardware_config)])
    if python_path:
        cmd.extend(["--python", str(python_path)])
    return cmd


def build_plan(
    *,
    kernel_name: str,
    kernel_script: str | Path,
    output_dir: str | Path,
    grid_dims: tuple[int, ...],
    raw_npuir: str | Path | None = None,
) -> BoundPipelinePlan:
    out = Path(output_dir)
    dump_dir = out / "ttdump"
    raw = Path(raw_npuir) if raw_npuir else out / f"{kernel_name}.raw.npuir.mlir"
    return BoundPipelinePlan(
        kernel_name=kernel_name,
        kernel_script=Path(kernel_script),
        output_dir=out,
        dump_dir=dump_dir,
        raw_npuir=raw,
        clean_npuir=out / f"{kernel_name}.clean.npuir.mlir",
        des_json=out / f"{kernel_name}.des.json",
        report_json=out / f"{kernel_name}.report.json",
        grid_dims=grid_dims,
    )


def run_bound_pipeline(
    plan: BoundPipelinePlan,
    *,
    python: str = sys.executable,
    clean_npuir_script: str | Path = PROJECT_ROOT / "scripts" / "clean_npuir.py",
    tritonsim_hivm: str | Path = PROJECT_ROOT / "build" / "bin" / "tritonsim-hivm",
    hardware_config: str | Path | None = None,
    calibration: str | Path | None = None,
    measured_csv: str | Path | None = None,
    measured_op_name: str | None = None,
    measured_us: float | None = None,
    dry_run: bool = False,
) -> dict:
    """Execute the bound pipeline, or return the plan when ``dry_run`` is true."""
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    plan.dump_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        return {
            "dry_run": True,
            "plan": plan.to_dict(),
            "dump_env": {
                key: build_dump_env(plan.dump_dir, {})[key]
                for key in (
                    "TRITON_DEBUG",
                    "TRITON_KERNEL_DUMP",
                    "TRITON_DUMP_DIR",
                    "TRITON_ALWAYS_COMPILE",
                )
            },
            "tritonsim_command": build_tritonsim_command(
                tritonsim_hivm=tritonsim_hivm,
                npuir_path=plan.clean_npuir,
                des_json=plan.des_json,
                hardware_config=hardware_config,
                python_path=python,
            ),
        }

    if not plan.raw_npuir.exists():
        subprocess.run(
            [python, str(plan.kernel_script)],
            cwd=PROJECT_ROOT,
            env=build_dump_env(plan.dump_dir),
            check=True,
        )
        dumped = dumped_npuir(
            plan.dump_dir,
            cache_dir=Path.home() / ".triton" / "cache",
        )
        plan.raw_npuir.write_text(dumped.read_text())

    subprocess.run(
        [python, str(clean_npuir_script), str(plan.raw_npuir), str(plan.clean_npuir)],
        cwd=PROJECT_ROOT,
        check=True,
    )
    report = report_from_npuir(
        npuir_path=plan.clean_npuir,
        grid_dims=plan.grid_dims,
        calib_db=load_calibration(calibration) if calibration else None,
        hardware_config=hardware_config,
        des_json_path=plan.des_json,
        kernel_name=plan.kernel_name,
        tritonsim_hivm=str(tritonsim_hivm),
        python_path=python,
        t_measured_us=measured_us,
        op_summary_csv=measured_csv,
        op_name_filter=measured_op_name,
        calibration_source=calibration,
    )
    report.to_json(plan.report_json)
    return {"dry_run": False, "plan": plan.to_dict(), "report_json": str(plan.report_json)}


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Run the Stage-B bound pipeline")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--kernel", help="Registry kernel name")
    src.add_argument("--script", help="Kernel script path")
    parser.add_argument("--kernel-name", help="Kernel label when using --script")
    parser.add_argument("--grid", required=True, help="Launch grid, e.g. 128,32")
    parser.add_argument("--output-dir", default=STAGEB_ROOT / "bounds")
    parser.add_argument("--raw-npuir", help="Existing NPUIR file; skips kernel dump")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--tritonsim-hivm", default=str(PROJECT_ROOT / "build" / "bin" / "tritonsim-hivm"))
    parser.add_argument("--hardware-config")
    parser.add_argument("--calibration")
    parser.add_argument("--measured-csv")
    parser.add_argument("--measured-op-name")
    parser.add_argument("--measured-us", type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.kernel:
        spec = get_kernel(args.kernel)
        kernel_name = spec.name
        kernel_script = spec.path
    else:
        kernel_script = Path(args.script)
        kernel_name = args.kernel_name or kernel_script.stem

    plan = build_plan(
        kernel_name=kernel_name,
        kernel_script=kernel_script,
        output_dir=args.output_dir,
        grid_dims=parse_grid(args.grid),
        raw_npuir=args.raw_npuir,
    )
    result = run_bound_pipeline(
        plan,
        python=args.python,
        tritonsim_hivm=args.tritonsim_hivm,
        hardware_config=args.hardware_config,
        calibration=args.calibration,
        measured_csv=args.measured_csv,
        measured_op_name=args.measured_op_name,
        measured_us=args.measured_us,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
