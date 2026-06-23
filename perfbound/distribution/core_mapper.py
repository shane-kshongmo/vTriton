#!/usr/bin/env python3
"""
Core distribution mapper — bridge from per-block DES simulation to
end-to-end kernel launch wall-clock.

Models the NPU firmware's block-to-core round-robin distribution
that happens inside ``rtKernelLaunch``.  One DES trace covers
**one program block** (one ``(pid_x, pid_y)`` tuple).  The mapper
replicates this across the physical core array to produce the
full-launch estimate.

Principles (verified against triton-ascend driver.py / rtKernelLaunch):
  * total_blocks = grid_x × grid_y × grid_z
  * Each block burns 1 AIC core + 1 AIV core simultaneously (MIX modes)
    or only one core type (AIC-only / AIV-only).
  * The firmware does ceil(total_blocks / effective_cores) waves.
  * E2E wall = waves × per_block_span  (on the bottleneck core type).

Usage::

    from perfbound.distribution.core_mapper import CoreMapper

    mapper = CoreMapper(aic_cores=20, aiv_cores=40)
    result = mapper.map(
        grid=(128, 32),
        per_block_span_aic_us=14.0,   # from DES trace
        per_block_span_aiv_us=272.0,  # from DES trace
        task_type="MIX_AIC",
    )
    print(f"E2E wall = {result.e2e_wall_us:.0f} us")
    print(f"Waves AIC = {result.waves_aic}, AIV = {result.waves_aiv}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


# ── Task-type classification (mirrors msprof Task Type column) ──────────────

_TASK_AIC_ONLY  = frozenset({"AI_CORE", "AICORE"})
_TASK_AIV_ONLY  = frozenset({"AI_VECTOR_CORE", "AIV"})
_TASK_MIX       = frozenset({"MIX_AIC", "MIX_AIV"})


# ── Preset hardware profiles ────────────────────────────────────────────────

@dataclass(frozen=True)
class _HWProfile:
    """Immutable hardware description used inside CoreMapper."""
    aic_cores: int
    aiv_cores: int
    label: str = ""

    def __repr__(self) -> str:
        return (f"HWProfile({self.label!r}, aic={self.aic_cores}, "
                f"aiv={self.aiv_cores})")


_PRESETS: dict[str, _HWProfile] = {
    "910B":  _HWProfile(aic_cores=20, aiv_cores=40, label="Ascend 910B"),
    "910B3": _HWProfile(aic_cores=20, aiv_cores=40, label="Ascend 910B3"),
    "910_9362": _HWProfile(aic_cores=20, aiv_cores=40,
                           label="Ascend 910_9362 (CANN 9.0)"),
}


# ── Result dataclass ────────────────────────────────────────────────────────

@dataclass
class DistributionResult:
    """Output of CoreMapper.map() — end-to-end estimates."""

    total_blocks: int = 0
    grid: Tuple[int, ...] = ()

    # waves = ceil(total_blocks / effective_cores)
    waves_aic: int = 0
    waves_aiv: int = 0

    # per-block spans fed in from DES
    per_block_span_aic_us: float = 0.0
    per_block_span_aiv_us: float = 0.0

    # scaled E2E per core type
    e2e_aic_us: float = 0.0
    e2e_aiv_us: float = 0.0

    # overall launch wall clock (bottleneck core type)
    e2e_wall_us: float = 0.0

    # bottleneck info
    bottleneck: str = ""          # "AIC" or "AIV"
    effective_cores: int = 0
    occupancy_pct: float = 100.0

    task_type: str = ""

    @property
    def e2e_wall_ms(self) -> float:
        return self.e2e_wall_us * 1e-3


# ── Mapper ──────────────────────────────────────────────────────────────────

class CoreMapper:
    """Maps a grid of program blocks across an Ascend NPU core array.

    Parameters
    ----------
    aic_cores: Number of AIC (Cube) cores.  Default: 20 (910B / 910B3).
    aiv_cores: Number of AIV (Vector) cores.  Default: 40 (910B / 910B3).
    """

    def __init__(self, *, aic_cores: int = 20, aiv_cores: int = 40):
        self._aic = aic_cores
        self._aiv = aiv_cores

    # ── factory ─────────────────────────────────────────────────────────

    @classmethod
    def from_preset(cls, name: str) -> "CoreMapper":
        """Create a mapper from a preset name (e.g. ``"910B"``, ``"910_9362"``).

        Presets match those used by ``triton-ascend`` 's
        ``AscendBackend`` / ``Ascend910_9362`` targets.
        """
        profile = _PRESETS.get(name)
        if profile is None:
            raise KeyError(
                f"Unknown preset {name!r}; available: "
                f"{list(_PRESETS)}")
        return cls(aic_cores=profile.aic_cores, aiv_cores=profile.aiv_cores)

    @classmethod
    def auto(cls) -> "CoreMapper":
        """Try to detect the platform at runtime via the NPU driver.

        Falls back to 910B3 defaults if the driver is unavailable.
        """
        try:
            import triton.backends.ascend.driver as _drv
            utils = _drv.NPUUtils()
            aic = utils.get_aicore_num()
            return cls(aic_cores=aic, aiv_cores=aic * 2)
        except Exception:
            return cls(aic_cores=20, aiv_cores=40)

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _flatten_grid(grid: Union[int, Tuple[int, ...], List[int]]) -> Tuple[int, ...]:
        if isinstance(grid, int):
            grid = (grid,)
        return tuple(int(g) for g in grid)

    @staticmethod
    def _total_blocks(grid: Tuple[int, ...]) -> int:
        total = 1
        for dim in grid:
            total *= dim
        return total

    @staticmethod
    def _waves(blocks: int, cores: int) -> int:
        """ceil(blocks / cores) with safety for zero cores."""
        if cores <= 0:
            return blocks
        return (blocks + cores - 1) // cores

    @staticmethod
    def _effective_cores(task_type: str, aic: int, aiv: int) -> Tuple[int, int]:
        """Return the number of cores used for AIC and AIV sides.

        For MIX task types every block uses one AIC core AND one AIV core
        simultaneously, so both core types are fully utilised for the
        launch duration (bottleneck is the fewer-core type).

        For AIC- / AIV-only, only the corresponding type is active.
        """
        tt = task_type.strip().upper()
        if tt in _TASK_AIC_ONLY:
            return (aic, 0)
        if tt in _TASK_AIV_ONLY:
            return (0, aiv)
        # MIX: both participate
        return (aic, aiv)

    # ── public API ──────────────────────────────────────────────────────

    def map(
        self,
        grid: Union[int, Tuple[int, ...], List[int]],
        per_block_span_aic_us: float = 0.0,
        per_block_span_aiv_us: float = 0.0,
        task_type: str = "MIX_AIC",
    ) -> DistributionResult:
        """Map a grid onto the core array and compute E2E wall clock.

        Parameters
        ----------
        grid:
            Launch grid dimensions, e.g. ``(128, 32)`` or ``4096``.
        per_block_span_aic_us:
            DES per-block span for the AIC (Cube) core side, in **μs**.
        per_block_span_aiv_us:
            DES per-block span for the AIV (Vector) core side, in **μs**.
        task_type:
            msprof Task Type string (``"MIX_AIC"``, ``"AI_CORE"``, etc.).

        Returns
        -------
        DistributionResult
            End-to-end estimates; ``result.e2e_wall_us`` is the number
            to compare with msprof ``Task Duration(us)``.
        """
        grid_tuple = self._flatten_grid(grid)
        total_blocks = self._total_blocks(grid_tuple)
        eff_aic, eff_aiv = self._effective_cores(task_type,
                                                  self._aic, self._aiv)

        waves_aic = self._waves(total_blocks, eff_aic) if eff_aic > 0 else 0
        waves_aiv = self._waves(total_blocks, eff_aiv) if eff_aiv > 0 else 0

        e2e_aic = waves_aic * per_block_span_aic_us
        e2e_aiv = waves_aiv * per_block_span_aiv_us

        e2e_wall = max(e2e_aic, e2e_aiv)

        # Determine bottleneck core type
        if e2e_aic >= e2e_aiv:
            bottleneck = "AIC"
            eff_cores = eff_aic
        else:
            bottleneck = "AIV"
            eff_cores = eff_aiv

        # Occupancy: what fraction of wall clock is spent on bottleneck-core work
        bottleneck_busy = e2e_aic if bottleneck == "AIC" else e2e_aiv
        occupancy = (bottleneck_busy / e2e_wall * 100.0) if e2e_wall > 0 else 100.0

        return DistributionResult(
            total_blocks=total_blocks,
            grid=grid_tuple,
            waves_aic=waves_aic,
            waves_aiv=waves_aiv,
            per_block_span_aic_us=per_block_span_aic_us,
            per_block_span_aiv_us=per_block_span_aiv_us,
            e2e_aic_us=e2e_aic,
            e2e_aiv_us=e2e_aiv,
            e2e_wall_us=e2e_wall,
            bottleneck=bottleneck,
            effective_cores=eff_cores,
            occupancy_pct=occupancy,
            task_type=task_type.strip().upper(),
        )

    def __repr__(self) -> str:
        return f"CoreMapper(aic={self._aic}, aiv={self._aiv})"
