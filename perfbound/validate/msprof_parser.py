# M6 — msprof CSV Parser for Validation
#
# Parses msprof op_summary CSVs to extract kernel timing measurements.
# Used by the validation harness to compare T_measured vs T_bound.
#
# Source spec: .omc/plans/a6_validation_harness.md §2

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path
from typing import NamedTuple, Optional

# Import from fit_constants (do not copy)
sys.path.insert(0, str(Path(__file__).parent.parent / "calibration" / "scripts"))
from fit_constants import MSProfRow, read_msprof_csv


# Task-type values that real msprof emits for AI compute-core execution.
# A Triton/HIVM kernel shows up as one of these depending on whether it is
# cube-only (AI_CORE/AICORE), mixed (MIX_AIC cube-dominant / MIX_AIV
# vector-dominant), or pure vector (AI_VECTOR_CORE / AIV).  The chunk_kda
# kernel profiles as MIX_AIC — recognising only "AI_CORE" silently drops it.
#
# Exact (set) membership, not substring: msprof task_type is a discrete enum,
# and substring matching on the short "AIV" token would misclassify any future
# composite type that happens to contain it (e.g. a hypothetical
# "HCCL_AIV_TASK").
_AICORE_TASK_TYPES = frozenset(
    {"AI_CORE", "AICORE", "MIX_AIC", "MIX_AIV", "AI_VECTOR_CORE", "AIV"}
)


def _is_aicore_task(task_type: str) -> bool:
    """True if an (already upper-cased, stripped) task_type is AI compute-core.

    Shared by the timing parser and the component-duration aggregator so the
    two cannot drift — a divergence first surfaced on real hardware, where the
    timing filter dropped MIX_AIC rows the component mapper kept.
    """
    return task_type in _AICORE_TASK_TYPES


class TimingResult(NamedTuple):
    """Parsed kernel timing from msprof CSV."""
    t_us: float                      # median duration in microseconds
    n_invocations: int               # valid invocations used in median
    n_warmup_discarded: int          # warmup invocations discarded


def parse_kernel_time_us(
    csv_path: Path,
    op_name_filter: str | None = None,
    n_warmup: int = 1,
) -> TimingResult:
    """Parse kernel timing from msprof op_summary CSV.

    Algorithm:
    1. Load rows via read_msprof_csv(csv_path).
    2. Filter to AiCore rows: task_type contains "AI_CORE" (case-insensitive);
       fall back to op_type if task_type is empty (old CANN CSV).
    3. If op_name_filter given: keep rows where op_name contains the filter
       (exact normalized match, not unrestricted substring).
    4. Raise ValueError if no rows remain.
    5. Sort by start_time_us. Group sequential rows into invocations:
       each invocation = one or more AiCore rows that started within a tight
       time window (gap threshold: 10× median row duration). Wall-clock latency
       per invocation = max duration_us across concurrent rows for that invocation
       (parallel device tasks should not be summed).
    6. Discard the first n_warmup invocations explicitly.
    7. Raise ValueError if fewer than 1 valid invocation remains.
    8. Return statistics.median(per_invocation_durations).

    Args:
        csv_path: Path to op_summary CSV.
        op_name_filter: Optional op name substring to filter (case-insensitive).
        n_warmup: Number of initial invocations to discard (default: 1).

    Returns:
        TimingResult with median duration, invocation count, and warmup count.

    Raises:
        ValueError: No matching rows, or insufficient invocations after warmup.
        OSError: CSV file not found or unreadable.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise OSError(f"CSV file not found: {csv_path}")

    rows = read_msprof_csv(csv_path)

    # Filter to AiCore rows
    aicore_rows = []
    for row in rows:
        # Skip malformed rows (NaN, zero duration)
        if not row.duration_us or math.isnan(row.duration_us) or row.duration_us <= 0:
            print(f"Warning: Skipping row with invalid duration: {row.duration_us}", file=sys.stderr)
            continue

        # Filter by task_type (prefer) or op_type (fallback)
        task_type = row.task_type.strip().upper() if row.task_type else ""
        if not task_type:
            # Old CANN CSV: fall back to op_type
            task_type = row.op_type.strip().upper() if row.op_type else ""

        if _is_aicore_task(task_type):
            # Apply op_name filter if specified
            if op_name_filter:
                if op_name_filter.lower() in row.op_name.lower():
                    aicore_rows.append(row)
            else:
                aicore_rows.append(row)

    if not aicore_rows:
        raise ValueError(f"No AiCore rows found in {csv_path} (filter={op_name_filter})")

    # Sort by start_time_us
    aicore_rows.sort(key=lambda r: r.start_time_us)

    # Group into invocations using gap detection.
    # NOTE: gap_threshold = 10× median row duration is a heuristic.  Known
    # failure mode: tightly-pipelined real traces where the inter-invocation
    # gap is comparable to per-row duration will merge adjacent invocations
    # into one, undercounting n_invocations and inflating per-invocation
    # latency (max across merged rows).  Acceptable for A.6.1; a more robust
    # detector (e.g. adaptive threshold from bimodal gap distribution) is
    # future work.
    durations = [r.duration_us for r in aicore_rows]
    median_duration = statistics.median(durations)
    gap_threshold = 10.0 * median_duration

    invocations = []
    current_invocation = [aicore_rows[0]]

    for i in range(1, len(aicore_rows)):
        row = aicore_rows[i]
        prev_row = aicore_rows[i - 1]
        time_gap = row.start_time_us - prev_row.start_time_us

        if time_gap > gap_threshold:
            # New invocation
            invocations.append(current_invocation)
            current_invocation = [row]
        else:
            # Same invocation
            current_invocation.append(row)

    invocations.append(current_invocation)

    # Discard warmup invocations
    if n_warmup > 0 and len(invocations) > n_warmup:
        invocations = invocations[n_warmup:]
        n_warmup_discarded = n_warmup
    else:
        n_warmup_discarded = 0

    if not invocations:
        raise ValueError(f"No valid invocations remaining after warmup={n_warmup} in {csv_path}")

    # Compute wall-clock latency per invocation (max duration across concurrent rows)
    per_invocation_durations = []
    for inv in invocations:
        max_duration = max(r.duration_us for r in inv)
        per_invocation_durations.append(max_duration)

    # Return median
    t_us = statistics.median(per_invocation_durations)

    return TimingResult(
        t_us=t_us,
        n_invocations=len(per_invocation_durations),
        n_warmup_discarded=n_warmup_discarded,
    )


def parse_component_durations(
    csv_path: Path,
    op_name_filter: str | None = None,
) -> dict[str, float]:
    """Return total duration per task-type category from all rows.

    Categories: 'aicore', 'mte', 'aicpu', 'other'.

    Map CSV task_type values (via _is_aicore_task):
    - AI_CORE/AICORE/MIX_AIC/MIX_AIV/AI_VECTOR_CORE/AIV → 'aicore'
    - MTE* → 'mte'
    - AI_CPU/AiCPU → 'aicpu'
    - else → 'other'

    Args:
        csv_path: Path to op_summary CSV.
        op_name_filter: Optional op name substring to filter (case-insensitive).
            When provided, only rows whose op_name contains the filter are
            included in the totals.  This is essential for multi-kernel CSVs
            where the dominant component must be computed per-kernel.

    Returns:
        Dict mapping category → total duration in microseconds.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise OSError(f"CSV file not found: {csv_path}")

    rows = read_msprof_csv(csv_path)

    totals = {"aicore": 0.0, "mte": 0.0, "aicpu": 0.0, "other": 0.0}

    for row in rows:
        if not row.duration_us or math.isnan(row.duration_us) or row.duration_us <= 0:
            continue

        # Apply op_name filter if specified
        if op_name_filter:
            if op_name_filter.lower() not in row.op_name.lower():
                continue

        task_type = row.task_type.strip().upper() if row.task_type else ""
        if not task_type:
            task_type = row.op_type.strip().upper() if row.op_type else ""

        if _is_aicore_task(task_type):
            totals["aicore"] += row.duration_us
        elif task_type.startswith("MTE"):
            totals["mte"] += row.duration_us
        elif "AI_CPU" in task_type or "AICPU" in task_type:
            totals["aicpu"] += row.duration_us
        else:
            totals["other"] += row.duration_us

    return totals
