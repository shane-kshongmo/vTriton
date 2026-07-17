"""Experiment result artifacts for Stage B.

Each experiment writes one JSON file under ``.omc/research/hw_runs/stageB``.
The schema is deliberately small so CI fixtures can validate campaign outputs
without requiring hardware access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


STAGEB_ROOT = Path(".omc") / "research" / "hw_runs" / "stageB"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExperimentResult:
    """Serializable Stage-B experiment artifact."""

    experiment: str
    rows: tuple[Mapping[str, Any], ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    status: str = "complete"
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment": self.experiment,
            "status": self.status,
            "generated_at": self.generated_at,
            "metrics": dict(self.metrics),
            "rows": [dict(row) for row in self.rows],
            "provenance": dict(self.provenance),
        }


def validate_experiment_result(data: Mapping[str, Any]) -> None:
    """Validate the shared Stage-B result schema.

    Raises:
        ValueError: if the artifact cannot be consumed by fixture tests or the
            figure generator.
    """
    required = {
        "schema_version",
        "experiment",
        "status",
        "generated_at",
        "metrics",
        "rows",
        "provenance",
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"missing Stage-B result fields: {sorted(missing)}")
    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported Stage-B schema_version={data['schema_version']}"
        )
    if not isinstance(data["experiment"], str) or not data["experiment"]:
        raise ValueError("experiment must be a non-empty string")
    if data["status"] not in {"planned", "running", "complete", "failed"}:
        raise ValueError(f"invalid experiment status: {data['status']!r}")
    if not isinstance(data["metrics"], dict):
        raise ValueError("metrics must be an object")
    if not isinstance(data["rows"], list):
        raise ValueError("rows must be a list")
    if not all(isinstance(row, dict) for row in data["rows"]):
        raise ValueError("each row must be an object")
    if not isinstance(data["provenance"], dict):
        raise ValueError("provenance must be an object")


def write_experiment_result(
    result: ExperimentResult,
    output_path: str | Path | None = None,
    root: str | Path = STAGEB_ROOT,
) -> Path:
    """Write a Stage-B experiment JSON artifact and return its path."""
    path = Path(output_path) if output_path else Path(root) / f"{result.experiment}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = result.to_dict()
    validate_experiment_result(data)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return path


def load_experiment_result(path: str | Path) -> dict[str, Any]:
    """Load and validate a Stage-B experiment JSON artifact."""
    data = json.loads(Path(path).read_text())
    validate_experiment_result(data)
    return data
