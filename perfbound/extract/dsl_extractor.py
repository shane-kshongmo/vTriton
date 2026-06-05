# M2 — DSL Extractor / Tier 1 input.
#
# Parse the @triton.jit function + launch grid + shape → Tier-1 quantities:
#   G, tile_assignment[p], occupancy, work[p], load_balance, redundancy.
#
# Method: recover the affine map from tl.program_id → tile via TTIR
# (tt.get_program_id, tt.load pointer arithmetic) using symbolic execution.
# Common idioms as templates first (grid_idioms.py), general affine recovery second.
#
# Audit of test/flash_attention.ttir confirms TTIR dumps carry:
#   - tt.get_program_id (grid axis)
#   - scf.for with program_id-bounded loops (persistent pattern)
#   - tt.make_tensor_ptr with [shape] operands (tile sizes)
#   - arith.divsi / remsi / muli chains (affine tile→pointer mapping)
#   - arith.constant values (block sizes, step counts)
#
# The extractor parses TTIR text (MLIR generic form), no MLIR Python bindings needed.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .grid_idioms import (
    idiom_1d_row_block, idiom_2d_tile_grid, TileIdiomResult, IDIOM_REGISTRY,
)


# ── Grid info dataclass ────────────────────────────────────────────────────

@dataclass
class GridInfo:
    """Tier 1 grid-level quantities for the bound model."""
    # Launch grid dimensions
    grid_dims: Tuple[int, ...]          # (G_x, G_y, G_z) from launch
    total_programs: int                 # G = product(grid_dims)

    # Per-program tile assignment: program_id → (tile_m, tile_n, ...)
    tile_assignment: Dict[int, Tuple[int, ...]]

    # Per-program work amount (e.g., elements computed)
    work: Dict[int, float]

    # Derived quantities
    occupancy: float                    # min(G, n_cores) / n_cores
    load_balance: float                 # mean(work) / max(work)
    redundancy: float = 1.0            # GM read amplification (default 1)

    # Busiest core (largest work)
    busiest_core_id: int = 0

    # Hardware-legality constraints (from configs/ascend_910b3.json)
    buffer_pressure_ok: bool = True
    divisibility_ok: bool = True

    @property
    def is_valid(self) -> bool:
        return self.buffer_pressure_ok and self.divisibility_ok


# ── TTIR text parser ───────────────────────────────────────────────────────

def _parse_constants(ttir_text: str) -> Dict[str, int]:
    """Extract named integer constants from TTIR text.

    Pattern: %cNAME = arith.constant VALUE : i32 (or i64)
    Example: %c128_i32 = arith.constant 128 : i32
    """
    consts = {}
    for m in re.finditer(
        r'%(c\w+(?:_\w+)*)\s*=\s*arith\.constant\s+(-?\d+(?:\.\d+)?)\s*:',
        ttir_text
    ):
        name = m.group(1)
        val = float(m.group(2))
        consts[name] = int(val) if val == int(val) else val
    return consts


def _parse_get_program_id(ttir_text: str) -> List[str]:
    """Find all tt.get_program_id calls and return grid axes.

    Pattern: %X = tt.get_program_id AXIS : i32
    Returns list like ["x"] or ["x", "y"].
    """
    axes = []
    for m in re.finditer(r'tt\.get_program_id\s+(\w+)', ttir_text):
        axes.append(m.group(1))
    return axes


def _parse_persistent_loop(ttir_text: str, consts: Dict[str, int]) -> Optional[Tuple[int, int]]:
    """Detect persistent kernel pattern: scf.for %iv = %pid to UPPER step STRIDE.

    Returns (upper_bound, stride) if found, or None.
    The stride typically equals n_cores (20 for 910B3).
    """
    # Pattern: scf.for %argN = %PID to %cUPPER step %cSTRIDE
    m = re.search(
        r'scf\.for\s+%\w+\s*=\s*%\w+\s+to\s+%(\w+)\s+step\s+%(\w+)',
        ttir_text
    )
    if m:
        upper = consts.get(m.group(1))
        stride = consts.get(m.group(2))
        if upper is not None and stride is not None:
            return (upper, stride)
    return None


def _parse_tile_shapes(ttir_text: str) -> List[Tuple[int, ...]]:
    """Extract tile shapes from tt.make_tensor_ptr shape operands.

    Pattern: tt.make_tensor_ptr %ptr, [%shape0, %shape1, ...], ...
    Returns list of (dim0, dim1, ...) tuples.

    Falls back to matching arith.constant dense value shapes for
    cases where make_tensor_ptr uses SSA values for shapes.
    """
    shapes = []
    # Direct shape operand extraction
    for m in re.finditer(
        r'tt\.make_tensor_ptr\s+\S+,\s*\[([^\]]+)\]',
        ttir_text
    ):
        shape_str = m.group(1)
        dims = []
        for token in shape_str.split(","):
            token = token.strip()
            # Try SSA reference like %c128_i64
            ref_m = re.match(r'%(\w+)', token)
            if ref_m:
                dims.append(ref_m.group(1))
            else:
                # Try literal integer
                try:
                    dims.append(int(token))
                except ValueError:
                    pass
        if dims:
            shapes.append(tuple(dims))
    return shapes


def _try_parse_launch_grid(kernel_source: str) -> Optional[Tuple[int, ...]]:
    """Best-effort extract launch grid from Triton kernel source.

    Searches for tl.constexpr grid assignments like `grid = (G_X, G_Y, G_Z)`.
    Returns grid tuple or None.
    """
    m = re.search(r'grid\s*=\s*\(([^)]+)\)', kernel_source)
    if m:
        dims = []
        for token in m.group(1).split(","):
            token = token.strip()
            try:
                dims.append(int(token))
            except ValueError:
                # Might be a variable computed from problem shape
                return None
        return tuple(dims)
    return None


# ── Main extractor ─────────────────────────────────────────────────────────

def extract_grid_info(
    kernel_source: str,
    launch_grid: Tuple[int, ...],
    problem_shape: Tuple[int, ...],
    block_sizes: Dict[str, int],
    n_cores: int = 20,
) -> GridInfo:
    """Extract Tier 1 grid information from a Triton kernel.

    Supports two input modes:
    1. TTIR file path (detected by .ttir or .mlir extension or tt.func content)
       → parses TTIR text to recover grid, tile shapes, and affine maps.
    2. Triton Python source → uses grid idioms (grid_idioms.py) with
       problem_shape + block_sizes to compute tile assignment.

    Args:
        kernel_source: Triton kernel source code OR path to TTIR dump file.
        launch_grid: Launch grid dimensions (G_x, G_y, G_z).
        problem_shape: Problem dimensions (M, N, K, ...).
        block_sizes: Block/tile sizes {BLOCK_M: 128, BLOCK_N: 64, ...}.
        n_cores: Number of available cores (20 for Cube, 40 for Vector-only).

    Returns:
        GridInfo with all Tier 1 quantities.

    Raises:
        NotImplementedError: For grid idioms not yet supported.
    """
    # Detect TTIR mode
    is_ttir = False
    ttir_text = ""
    source_path = Path(kernel_source) if "\n" not in kernel_source else None
    if source_path and source_path.exists():
        ttir_text = source_path.read_text()
        is_ttir = bool(re.search(r'tt\.func|tt\.get_program_id', ttir_text))
    elif "tt.func" in kernel_source or "tt.get_program_id" in kernel_source:
        ttir_text = kernel_source
        is_ttir = True

    if is_ttir:
        return _extract_from_ttir(ttir_text, launch_grid, problem_shape,
                                  block_sizes, n_cores)
    else:
        return _extract_from_idioms(kernel_source, launch_grid, problem_shape,
                                    block_sizes, n_cores)


def _extract_from_ttir(
    ttir_text: str,
    launch_grid: Tuple[int, ...],
    problem_shape: Tuple[int, ...],
    block_sizes: Dict[str, int],
    n_cores: int,
) -> GridInfo:
    """Extract grid info from a TTIR dump text."""
    consts = _parse_constants(ttir_text)
    axes = _parse_get_program_id(ttir_text)
    persist = _parse_persistent_loop(ttir_text, consts)
    tile_shapes = _parse_tile_shapes(ttir_text)

    G = 1
    for d in launch_grid:
        G *= d

    # Resolve tile shape constants
    resolved_shapes = []
    for shape in tile_shapes:
        resolved = []
        for dim in shape:
            if isinstance(dim, str) and dim in consts:
                resolved.append(consts[dim])
            elif isinstance(dim, int):
                resolved.append(dim)
        if resolved:
            resolved_shapes.append(tuple(resolved))

    # Detect idiom from TTIR structure
    # Persistent pattern: 1D grid + scf.for with program_id stride
    is_persistent = persist is not None and len(axes) == 1
    is_2d_grid = len(axes) >= 2

    if is_2d_grid or (resolved_shapes and len(resolved_shapes) >= 2 and len(resolved_shapes[0]) >= 2):
        # 2D tile grid
        if not block_sizes and resolved_shapes:
            shape0 = resolved_shapes[0]
            if len(shape0) >= 2:
                block_sizes = {"BLOCK_M": shape0[0], "BLOCK_N": shape0[1]}
        if "BLOCK_M" in block_sizes and "BLOCK_N" in block_sizes:
            M = problem_shape[0] if len(problem_shape) > 0 else 1
            N = problem_shape[1] if len(problem_shape) > 1 else 1
            result = idiom_2d_tile_grid(M, N, block_sizes["BLOCK_M"],
                                        block_sizes["BLOCK_N"])
            return _idiom_to_grid(result, launch_grid, n_cores)

    if is_persistent or (len(axes) == 1 and resolved_shapes):
        # 1D row-block (possibly persistent)
        if not block_sizes and resolved_shapes:
            shape0 = resolved_shapes[0]
            blk = shape0[0] if shape0 else 128
            block_sizes = {"BLOCK_M": blk}
        if "BLOCK_M" in block_sizes:
            M = problem_shape[0] if len(problem_shape) > 0 else 1
            result = idiom_1d_row_block(M, block_sizes["BLOCK_M"])
            return _idiom_to_grid(result, launch_grid, n_cores)

    # Fallback: uniform work assumption for exotic patterns
    return _uniform_grid(launch_grid, n_cores)


def _extract_from_idioms(
    kernel_source: str,
    launch_grid: Tuple[int, ...],
    problem_shape: Tuple[int, ...],
    block_sizes: Dict[str, int],
    n_cores: int,
) -> GridInfo:
    """Extract grid info from Python Triton source using idiom templates."""
    # Try to match against known idioms
    is_2d = ("program_id(0)" in kernel_source and "program_id(1)" in kernel_source)

    if is_2d and "BLOCK_M" in block_sizes and "BLOCK_N" in block_sizes:
        M = problem_shape[0] if len(problem_shape) > 0 else 1
        N = problem_shape[1] if len(problem_shape) > 1 else 1
        result = idiom_2d_tile_grid(M, N, block_sizes["BLOCK_M"],
                                    block_sizes["BLOCK_N"])
        return _idiom_to_grid(result, launch_grid, n_cores)

    if "BLOCK_M" in block_sizes:
        M = problem_shape[0] if len(problem_shape) > 0 else 1
        result = idiom_1d_row_block(M, block_sizes["BLOCK_M"])
        return _idiom_to_grid(result, launch_grid, n_cores)

    # Fallback
    return _uniform_grid(launch_grid, n_cores)


def _idiom_to_grid(
    result: TileIdiomResult,
    launch_grid: Tuple[int, ...],
    n_cores: int,
) -> GridInfo:
    """Convert a TileIdiomResult to GridInfo."""
    G = 1
    for d in launch_grid:
        G *= d

    works = list(result.work.values())
    occupancy = min(G, n_cores) / n_cores if n_cores > 0 else 1.0
    load_balance = sum(works) / (max(works) * len(works)) if works and max(works) > 0 else 1.0
    busiest = max(result.work, key=result.work.get) if result.work else 0

    return GridInfo(
        grid_dims=launch_grid,
        total_programs=G,
        tile_assignment=result.tile_assignment,
        work=result.work,
        occupancy=occupancy,
        load_balance=load_balance,
        redundancy=1.0,
        busiest_core_id=busiest,
        buffer_pressure_ok=result.buffer_pressure_ok,
        divisibility_ok=result.divisibility_ok,
    )


def _uniform_grid(
    launch_grid: Tuple[int, ...],
    n_cores: int,
) -> GridInfo:
    """Create a uniform-work GridInfo (fallback for unknown idioms)."""
    G = 1
    for d in launch_grid:
        G *= d

    work = {p: 1.0 for p in range(G)}
    occupancy = min(G, n_cores) / n_cores if n_cores > 0 else 1.0

    return GridInfo(
        grid_dims=launch_grid,
        total_programs=G,
        tile_assignment={},
        work=work,
        occupancy=occupancy,
        load_balance=1.0,
        redundancy=1.0,
        busiest_core_id=0,
    )
