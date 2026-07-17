"""Shared Stage-A experiment fixtures and lightweight loaders."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HW_RUNS_ROOT = PROJECT_ROOT / ".omc" / "research" / "hw_runs"

CHUNK_KDA_DES_JSON = HW_RUNS_ROOT / "kda_des.json"
CHUNK_KDA_MSPROF_CSV = HW_RUNS_ROOT / "chunk_kda_op_summary.csv"
CHUNK_KDA_KERNEL_OP_NAME = "chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2"
CHUNK_KDA_KERNEL_SUBSTR = "chunk_kda_bwd"
CHUNK_KDA_GRID_DIMS = (128, 32)
CHUNK_KDA_TOTAL_PROGRAMS = 4096
CHUNK_KDA_N_CORES = 20
CHUNK_KDA_MEASURED_US = 104326.0


def load_des_graph(path: str | Path = CHUNK_KDA_DES_JSON) -> dict[str, Any]:
    """Load a DES graph JSON artifact."""
    with Path(path).open() as f:
        return json.load(f)


def matching_msprof_rows(
    csv_path: str | Path = CHUNK_KDA_MSPROF_CSV,
    kernel_substr: str = CHUNK_KDA_KERNEL_SUBSTR,
) -> list[dict[str, str]]:
    """Return msprof rows whose ``Op Name`` contains ``kernel_substr``."""
    with Path(csv_path).open(newline="") as f:
        return [
            row
            for row in csv.DictReader(f)
            if kernel_substr in row.get("Op Name", "")
        ]
