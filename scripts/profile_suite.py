#!/usr/bin/env python3
"""Stage-B multi-card profiling scheduler.

``remote_bench.py`` profiles one host at a time.  This script builds the
kernel x shape job matrix and assigns jobs round-robin across a host list.
The default path writes a schedule artifact; ``--execute`` fans assignments
out with a thread pool sized to the number of target cards.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perfbound.experiments.artifacts import (
    STAGEB_ROOT,
    ExperimentResult,
    write_experiment_result,
)
from perfbound.experiments.registry import KernelGroup, KernelSpec, iter_kernel_specs
from scripts.remote_bench import run_remote_bench


@dataclass(frozen=True)
class Card:
    host: str
    remote_path: str = "~/vTriton"


@dataclass(frozen=True)
class ProfileJob:
    kernel_name: str
    kernel_script: str
    group: str
    optimized: bool
    shape_id: str
    shape: dict


@dataclass(frozen=True)
class ProfileAssignment:
    job: ProfileJob
    card: Card
    output_csv: Path
    output_npy: Path

    def to_dict(self) -> dict:
        return {
            "host": self.card.host,
            "remote_path": self.card.remote_path,
            "kernel_name": self.job.kernel_name,
            "kernel_script": self.job.kernel_script,
            "group": self.job.group,
            "optimized": self.job.optimized,
            "shape_id": self.job.shape_id,
            "shape": self.job.shape,
            "output_csv": str(self.output_csv),
            "output_npy": str(self.output_npy),
        }


def parse_cards(hosts: str | None, remote_path: str = "~/vTriton") -> tuple[Card, ...]:
    raw = hosts or os.environ.get("VTRITON_REMOTE_HOSTS") or os.environ.get("VTRITON_REMOTE_HOST")
    if not raw:
        raise ValueError("no hosts configured; use --hosts or VTRITON_REMOTE_HOSTS")
    cards = tuple(Card(host=h.strip(), remote_path=remote_path) for h in raw.split(",") if h.strip())
    if not cards:
        raise ValueError("host list is empty")
    return cards


def build_jobs(specs: Iterable[KernelSpec]) -> tuple[ProfileJob, ...]:
    jobs: list[ProfileJob] = []
    for spec in specs:
        for shape in spec.shapes:
            shape_dict = dict(shape)
            shape_id = str(shape_dict.get("id", "default"))
            jobs.append(
                ProfileJob(
                    kernel_name=spec.name,
                    kernel_script=spec.path_for_cli,
                    group=spec.group.value,
                    optimized=spec.optimized,
                    shape_id=shape_id,
                    shape=shape_dict,
                )
            )
    return tuple(jobs)


def schedule_round_robin(
    jobs: Iterable[ProfileJob],
    cards: Iterable[Card],
    output_root: str | Path,
) -> tuple[ProfileAssignment, ...]:
    card_list = tuple(cards)
    if not card_list:
        raise ValueError("cannot schedule without cards")

    out = Path(output_root)
    assignments: list[ProfileAssignment] = []
    for idx, job in enumerate(jobs):
        card = card_list[idx % len(card_list)]
        stem = f"{idx:04d}_{job.kernel_name}_{job.shape_id}"
        assignments.append(
            ProfileAssignment(
                job=job,
                card=card,
                output_csv=out / "profiles" / f"{stem}_op_summary.csv",
                output_npy=out / "profiles" / f"{stem}_output.npy",
            )
        )
    return tuple(assignments)


def write_schedule(
    assignments: Iterable[ProfileAssignment],
    *,
    output_root: str | Path = STAGEB_ROOT,
    status: str = "planned",
) -> Path:
    rows = tuple(a.to_dict() for a in assignments)
    result = ExperimentResult(
        experiment="profile_suite_schedule",
        status=status,
        rows=rows,
        metrics={
            "job_count": len(rows),
            "card_count": len({row["host"] for row in rows}),
        },
        provenance={"scheduler": "round_robin"},
    )
    return write_experiment_result(result, root=output_root)


def shape_env(shape: Mapping[str, object]) -> dict[str, str]:
    """Convert env-named shape metadata into kernel environment overrides."""
    return {
        str(key): str(value)
        for key, value in shape.items()
        if key != "id" and value != "module_default"
    }


def _profile_assignment(assignment: ProfileAssignment) -> dict:
    row = assignment.to_dict()
    env = shape_env(assignment.job.shape)
    row["kernel_env"] = env
    try:
        csv_path, npy_path = run_remote_bench(
            remote_host=assignment.card.host,
            remote_path=assignment.card.remote_path,
            kernel_name=assignment.job.kernel_name,
            kernel_script=Path(assignment.job.kernel_script),
            output_csv=assignment.output_csv,
            output_npy=assignment.output_npy,
            kernel_env=env or None,
        )
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
    else:
        row["status"] = "complete"
        row["actual_csv"] = str(csv_path)
        row["actual_npy"] = str(npy_path) if npy_path else None
    return row


def execute_assignments(
    assignments: Iterable[ProfileAssignment],
    *,
    max_workers: int | None = None,
) -> list[dict]:
    assignment_list = tuple(assignments)
    if not assignment_list:
        return []
    workers = max_workers or len({a.card.host for a in assignment_list}) or 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_profile_assignment, assignment_list))


def execute_assignments_sequential(assignments: Iterable[ProfileAssignment]) -> list[dict]:
    """Debug helper for single-step repros."""
    return [_profile_assignment(assignment) for assignment in assignments]


def _split_csv(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    return parts or None


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Profile the Stage-B kernel suite")
    parser.add_argument("--hosts", help="Comma-separated SSH hosts; defaults to VTRITON_REMOTE_HOSTS")
    parser.add_argument("--remote-path", default="~/vTriton")
    parser.add_argument("--groups", help="Comma-separated groups I,V,...")
    parser.add_argument("--kernels", help="Comma-separated registry kernel names")
    parser.add_argument("--user-manifest")
    parser.add_argument("--output-root", default=STAGEB_ROOT)
    parser.add_argument("--execute", action="store_true", help="Run remote profiling instead of only writing a schedule")
    parser.add_argument("--max-workers", type=int, default=None, help="Concurrent remote profiling workers")
    args = parser.parse_args()

    groups = tuple(KernelGroup(g) for g in _split_csv(args.groups) or ()) or None
    names = _split_csv(args.kernels)
    specs = iter_kernel_specs(user_manifest=args.user_manifest, groups=groups, names=names)
    cards = parse_cards(args.hosts, args.remote_path)
    assignments = schedule_round_robin(build_jobs(specs), cards, args.output_root)

    if args.execute:
        rows = execute_assignments(assignments, max_workers=args.max_workers)
        failed_count = sum(1 for row in rows if row["status"] == "failed")
        result = ExperimentResult(
            experiment="profile_suite",
            status="failed" if failed_count else "complete",
            rows=tuple(rows),
            metrics={
                "job_count": len(rows),
                "card_count": len(cards),
                "failed_count": failed_count,
            },
            provenance={"scheduler": "round_robin"},
        )
        path = write_experiment_result(result, root=args.output_root)
    else:
        path = write_schedule(assignments, output_root=args.output_root)

    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
