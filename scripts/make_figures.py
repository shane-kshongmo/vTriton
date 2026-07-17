#!/usr/bin/env python3
"""Generate Stage-B figure/table inputs from experiment JSON artifacts.

This foundation script intentionally avoids new dependencies.  It validates all
Stage-B experiment JSON files and writes CSV tables that paper figure scripts
can consume later.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perfbound.experiments.artifacts import STAGEB_ROOT, load_experiment_result


def collect_results(root: str | Path = STAGEB_ROOT) -> tuple[dict[str, Any], ...]:
    root_path = Path(root)
    if not root_path.exists():
        return ()
    results = []
    for path in sorted(root_path.glob("*.json")):
        data = load_experiment_result(path)
        data["_source"] = str(path)
        results.append(data)
    return tuple(results)


def metric_summary_rows(results: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for result in results:
        metrics = result.get("metrics", {})
        if not metrics:
            rows.append(
                {
                    "experiment": result["experiment"],
                    "metric": "",
                    "value": "",
                    "status": result["status"],
                    "source": result.get("_source", ""),
                }
            )
            continue
        for metric, value in sorted(metrics.items()):
            rows.append(
                {
                    "experiment": result["experiment"],
                    "metric": metric,
                    "value": value,
                    "status": result["status"],
                    "source": result.get("_source", ""),
                }
            )
    return tuple(rows)


def write_metric_summary(
    results: Iterable[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    rows = metric_summary_rows(results)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("experiment", "metric", "value", "status", "source"),
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Build Stage-B figure/table CSVs")
    parser.add_argument("--input-root", default=STAGEB_ROOT)
    parser.add_argument(
        "--output",
        default=STAGEB_ROOT / "figure_metrics.csv",
    )
    args = parser.parse_args()

    results = collect_results(args.input_root)
    path = write_metric_summary(results, args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
