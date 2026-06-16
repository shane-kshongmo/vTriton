# M6 — HIVM Edit Primitives for Counterfactual Validation
#
# Structured source-to-source edits on HIVM JSON (NOT regex — per project rule).
# Each edit targets a specific gap and mutates the HIVM JSON accordingly.
# Edits write to a temp file, never mutating the input in place.
#
# Source spec: .omc/plans/a6_2_counterfactual.md §1

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class HivmEdit:
    """A gap-targeted HIVM edit."""
    gap_name: str                 # e.g. "gap3_avoidable_serial"
    description: str
    apply: Callable[[Path], Path] # edited HIVM written to a temp path


def _load_hivm_data(hivm_path: Path) -> dict:
    """Load an HIVM DES-graph JSON file and return the full data dict.

    The returned dict is guaranteed to contain an ``operations`` array
    (matches the schema consumed by ``hivm_extractor.load_hivm_desgraph``).

    Raises:
        ValueError: If JSON has no 'operations' key.
    """
    with open(hivm_path) as f:
        data = json.load(f)

    ops = data.get("operations")
    if ops is None:
        raise ValueError(
            f"HIVM JSON must contain an 'operations' array. "
            f"Found keys: {list(data.keys())}"
        )
    return data


def _write_edited(data: dict, suffix: str = "_edited") -> Path:
    """Write edited HIVM JSON to a temp file and return the path."""
    tmpdir = tempfile.mkdtemp()
    tmp = Path(tmpdir) / f"hivm{suffix}.json"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    return tmp


def raise_repeat(hivm_path: Path, factor: int = 2) -> Path:
    """Raise the repeat count on all compute ops by the given factor.

    Targets Gap 4 (intra-unit execution efficiency): increasing repeat
    amortizes loop overhead and improves SIMD utilization.

    Args:
        hivm_path: Path to original HIVM JSON.
        factor: Multiplier for repeat field (default: 2).

    Returns:
        Path to edited HIVM JSON (temp file).

    Raises:
        ValueError: If input has no 'operations' key.
    """
    if factor < 1:
        raise ValueError(f"repeat factor must be >= 1, got {factor}")

    data = _load_hivm_data(hivm_path)
    ops = data["operations"]

    # Accept both the synthetic test pipe names ("Cube"/"Vector"/"Scalar") and
    # the raw HIVM pipe tokens emitted into real des.json by the C++ emitter
    # ("PIPE_M" = cube/matrix, "PIPE_V" = vector, "PIPE_S" = scalar).  Matching
    # only the synthetic names silently no-op'd on every real kernel.
    compute_pipes = {"Cube", "Vector", "Scalar", "PIPE_M", "PIPE_V", "PIPE_S"}

    matched = 0
    for op in ops:
        pipe = op.get("pipe", "")
        if pipe in compute_pipes:
            matched += 1
            current_repeat = op.get("repeat", 1)
            op["repeat"] = current_repeat * factor

    # No-op guard: an edit that targets nothing is almost always a mistake
    # (wrong kernel / wrong pipe names). Fail loudly rather than emit an
    # identical "edited" file that silently measures a zero counterfactual.
    if matched == 0:
        raise ValueError(
            f"raise_repeat is a no-op: no compute ops (Cube/Vector/Scalar) "
            f"found in {hivm_path}. The edit does not apply to this kernel."
        )

    return _write_edited(data, suffix=f"_repeat_{factor}")


def insert_pingpong(hivm_path: Path) -> Path:
    """Insert a duplicate buffer op after each MTE_UB transfer.

    Targets Gap 3 (avoidable serialization): adding a second buffer
    enables double-buffering / ping-pong, hiding the serialization
    latency between MTE_UB and downstream consumers.

    Args:
        hivm_path: Path to original HIVM JSON.

    Returns:
        Path to edited HIVM JSON (temp file).

    Raises:
        ValueError: If input has no 'operations' key.
    """
    data = _load_hivm_data(hivm_path)
    ops = data["operations"]

    max_id = max((op.get("id", 0) for op in ops), default=0)
    new_ops = []
    next_id = max_id + 1

    added = 0
    for op in ops:
        new_ops.append(op)
        pipe = op.get("pipe", "")
        # MTE_UB (UB→OUT or UB→L1) transfers benefit from ping-pong
        if "MTE_UB" in pipe or pipe == "MTE_UB" or pipe == "UnifiedBuffer":
            dup = dict(op)
            dup["id"] = next_id
            dup["name"] = op.get("name", "") + "_pingpong"
            next_id += 1
            new_ops.append(dup)
            added += 1

    # No-op guard: fail loudly if there are no MTE_UB transfers to double-buffer.
    if added == 0:
        raise ValueError(
            f"insert_pingpong is a no-op: no MTE_UB transfers found in "
            f"{hivm_path}. The edit does not apply to this kernel."
        )

    data = dict(data)
    data["operations"] = new_ops

    return _write_edited(data, suffix="_pingpong")


def merge_transfers(hivm_path: Path) -> Path:
    """Merge consecutive same-space MTE_GM transfers.

    Targets Gap 2 (coalescing / transfer efficiency): merging small
    adjacent transfers into one large transfer improves bandwidth
    utilization by avoiding small-packet amortization overhead.

    Consecutive MTE_GM ops with the same src_space and dst_space are
    merged: bytes are summed, the first op is kept, and the rest removed.

    Args:
        hivm_path: Path to original HIVM JSON.

    Returns:
        Path to edited HIVM JSON (temp file).

    Raises:
        ValueError: If input has no 'operations' key.
    """
    data = _load_hivm_data(hivm_path)
    ops = data["operations"]

    mte_gm_pipes = {"MTE2", "MTE_GM", "MTE-GM"}

    mte_seen = 0
    merged = []
    skip_next = False

    for i, op in enumerate(ops):
        if skip_next:
            skip_next = False
            continue

        pipe = op.get("pipe", "")
        if pipe in mte_gm_pipes:
            mte_seen += 1

        if pipe in mte_gm_pipes and i + 1 < len(ops):
            next_op = ops[i + 1]
            next_pipe = next_op.get("pipe", "")

            if (
                next_pipe in mte_gm_pipes
                and op.get("src_space") == next_op.get("src_space")
                and op.get("dst_space") == next_op.get("dst_space")
            ):
                # Merge: sum bytes, keep first op
                merged_op = dict(op)
                merged_op["bytes"] = op.get("bytes", 0) + next_op.get("bytes", 0)
                merged.append(merged_op)
                skip_next = True
                continue

        merged.append(op)

    # No-op guard: fail loudly if there are no MTE_GM transfers at all.
    # (Having MTE_GM transfers that are simply non-adjacent is a legitimate
    # no-merge outcome and must NOT raise.)
    if mte_seen == 0:
        raise ValueError(
            f"merge_transfers is a no-op: no MTE_GM transfers found in "
            f"{hivm_path}. The edit does not apply to this kernel."
        )

    data = dict(data)
    data["operations"] = merged

    return _write_edited(data, suffix="_merged")


def verify_edit_via_extract(original_path: Path, edited_path: Path) -> bool:
    """Confirm an edit is visible through the real HIVM extractor.

    Re-extracts both the original and edited HIVM via
    ``hivm_extractor.load_hivm_desgraph`` (the same consumer the model uses)
    and reports whether any structural property the model reads actually
    changed: operation count, total repeat, or total bytes transferred.

    This is the reversibility check from the A.6.2 plan §1 — it guarantees an
    edit is not merely textual but alters a field the bound model consumes.
    A genuine no-op (e.g. ``raise_repeat(factor=1)``) correctly returns False.

    Args:
        original_path: Path to the original HIVM DES-graph JSON.
        edited_path: Path to the edited HIVM DES-graph JSON.

    Returns:
        True if a model-visible structural field differs, False otherwise.
    """
    from ..extract.hivm_extractor import load_hivm_desgraph

    orig = load_hivm_desgraph(original_path)
    edited = load_hivm_desgraph(edited_path)

    if len(orig) != len(edited):
        return True
    if sum(op.repeat for op in orig) != sum(op.repeat for op in edited):
        return True
    if sum(op.bytes_transferred for op in orig) != sum(op.bytes_transferred for op in edited):
        return True
    return False
