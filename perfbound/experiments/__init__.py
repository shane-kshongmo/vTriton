"""Stage-B experiment foundation.

This package is intentionally hardware-light: it defines kernel metadata,
experiment artifact schemas, and helpers used by the Stage-B scripts.  Remote
execution stays in ``scripts/remote_bench.py`` and bound computation stays in
``perfbound.combine``.
"""

from .artifacts import (
    STAGEB_ROOT,
    ExperimentResult,
    load_experiment_result,
    validate_experiment_result,
    write_experiment_result,
)
from .registry import (
    KernelGroup,
    KernelSpec,
    get_kernel,
    iter_kernel_specs,
    load_user_registry,
)
from .stage_a import (
    CHUNK_KDA_DES_JSON,
    CHUNK_KDA_GRID_DIMS,
    CHUNK_KDA_KERNEL_OP_NAME,
    CHUNK_KDA_MEASURED_US,
    CHUNK_KDA_MSPROF_CSV,
    CHUNK_KDA_N_CORES,
    CHUNK_KDA_TOTAL_PROGRAMS,
    HW_RUNS_ROOT,
)

__all__ = [
    "STAGEB_ROOT",
    "ExperimentResult",
    "CHUNK_KDA_DES_JSON",
    "CHUNK_KDA_GRID_DIMS",
    "CHUNK_KDA_KERNEL_OP_NAME",
    "CHUNK_KDA_MEASURED_US",
    "CHUNK_KDA_MSPROF_CSV",
    "CHUNK_KDA_N_CORES",
    "CHUNK_KDA_TOTAL_PROGRAMS",
    "HW_RUNS_ROOT",
    "KernelGroup",
    "KernelSpec",
    "get_kernel",
    "iter_kernel_specs",
    "load_experiment_result",
    "load_user_registry",
    "validate_experiment_result",
    "write_experiment_result",
]
