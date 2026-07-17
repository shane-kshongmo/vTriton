"""Kernel registry and adapter layer for Stage-B experiments.

The registry records the small interface needed by the experiment harness:
``build_inputs``, ``Model``, optional ``reference``, group I-V, optimization
status, and shape metadata.  Existing ``test/*_bench.py`` kernels are adapted
by path, and external user kernels can be added through a JSON manifest.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from hashlib import sha1
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class KernelGroup(str, Enum):
    """Stage-B kernel-suite groups from the implementation plan."""

    COMPUTE_REGULAR = "I"
    MEMORY_REGULAR = "II"
    MIXED_CUBE_VECTOR = "III"
    RAGGED_GRID = "IV"
    GAP_SEEDED = "V"


@dataclass(frozen=True)
class KernelSpec:
    """Metadata and adapter for one experiment kernel."""

    name: str
    path: Path
    group: KernelGroup
    optimized: bool
    shapes: tuple[Mapping[str, Any], ...] = field(
        default_factory=lambda: ({"id": "default"},)
    )
    description: str = ""
    tags: tuple[str, ...] = ()
    build_inputs_name: str = "build_inputs"
    model_name: str = "Model"
    reference_name: str = "reference"

    @property
    def path_for_cli(self) -> str:
        try:
            return str(self.path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(self.path)

    def load_module(self) -> ModuleType:
        return _load_module(self.path)

    def build_inputs(self) -> Any:
        module = self.load_module()
        fn = getattr(module, self.build_inputs_name, None)
        if fn is None:
            raise AttributeError(
                f"{self.path} does not expose {self.build_inputs_name}()"
            )
        return fn()

    def model_factory(self) -> Any:
        module = self.load_module()
        cls = getattr(module, self.model_name, None)
        if cls is None:
            raise AttributeError(f"{self.path} does not expose {self.model_name}")
        return cls()

    def reference(self) -> Callable[..., Any] | None:
        module = self.load_module()
        return getattr(module, self.reference_name, None)

    def validate_interface(self) -> None:
        module = self.load_module()
        if not hasattr(module, self.build_inputs_name):
            raise ValueError(f"{self.name}: missing {self.build_inputs_name}()")
        if not hasattr(module, self.model_name):
            raise ValueError(f"{self.name}: missing {self.model_name}")

    def to_manifest_row(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path_for_cli,
            "group": self.group.value,
            "optimized": self.optimized,
            "shapes": [dict(shape) for shape in self.shapes],
            "description": self.description,
            "tags": list(self.tags),
        }


def _load_module(path: str | Path) -> ModuleType:
    kernel_path = Path(path)
    if not kernel_path.is_absolute():
        kernel_path = PROJECT_ROOT / kernel_path
    kernel_path = kernel_path.resolve()
    if not kernel_path.exists():
        raise FileNotFoundError(f"kernel script not found: {kernel_path}")

    digest = sha1(str(kernel_path).encode("utf-8")).hexdigest()[:12]
    module_name = f"_stageb_kernel_{kernel_path.stem}_{digest}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, kernel_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _shape(shape_id: str = "default", **values: Any) -> dict[str, Any]:
    return {"id": shape_id, **values}


def _builtin(
    name: str,
    filename: str,
    *,
    group: KernelGroup = KernelGroup.MEMORY_REGULAR,
    optimized: bool = True,
    shapes: tuple[Mapping[str, Any], ...] = (_shape(),),
    description: str = "",
    tags: tuple[str, ...] = (),
) -> KernelSpec:
    return KernelSpec(
        name=name,
        path=PROJECT_ROOT / "test" / filename,
        group=group,
        optimized=optimized,
        shapes=shapes,
        description=description,
        tags=tags,
    )


_BUILTIN_KERNELS: tuple[KernelSpec, ...] = (
    _builtin(
        "vector_add",
        "vector_add_bench.py",
        shapes=(_shape("default", N="module_default"),),
        description="Regular memory-bound elementwise add.",
    ),
    _builtin(
        "vector_add_2x",
        "vector_add_2x_bench.py",
        shapes=(_shape("default", N="module_default"),),
        description="Regular memory-bound two-output add fixture.",
    ),
    _builtin(
        "softmax",
        "softmax_bench.py",
        description="Regular reduction-style softmax fixture.",
    ),
    _builtin(
        "layernorm",
        "layernorm_bench.py",
        description="Regular normalization fixture.",
    ),
    _builtin(
        "rmsnorm",
        "rmsnorm_bench.py",
        description="Regular RMSNorm fixture.",
    ),
    _builtin(
        "seeded_gap1",
        "seeded_gap1_bench.py",
        group=KernelGroup.GAP_SEEDED,
        optimized=False,
        description="Group-V seeded wrong-unit placement fixture.",
    ),
    _builtin(
        "seeded_gap2",
        "seeded_gap2_bench.py",
        group=KernelGroup.GAP_SEEDED,
        optimized=False,
        description="Group-V seeded small-transfer coalescing fixture.",
    ),
    _builtin(
        "seeded_serial",
        "seeded_serial_bench.py",
        group=KernelGroup.GAP_SEEDED,
        optimized=False,
        shapes=(
            _shape("default", N_A="module_default", N_B="module_default"),
            _shape("small", SEED_N_A=262144, SEED_N_B=8192, SEED_NITER=64),
        ),
        description="Group-V seeded avoidable serialization fixture.",
        tags=("gap3", "counterfactual"),
    ),
)


def load_user_registry(path: str | Path) -> tuple[KernelSpec, ...]:
    """Load user-supplied kernels from a JSON manifest.

    Manifest format:

    ``{"kernels": [{"name": "...", "path": "...", "group": "III",
    "optimized": true, "shapes": [{"id": "default"}]}]}``
    """
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text())
    rows = data.get("kernels", data if isinstance(data, list) else [])
    if not isinstance(rows, list):
        raise ValueError("user kernel manifest must contain a kernels list")

    specs: list[KernelSpec] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each user kernel entry must be an object")
        raw_path = Path(row["path"])
        kernel_path = raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path
        shape_rows = row.get("shapes") or [{"id": "default"}]
        specs.append(
            KernelSpec(
                name=row["name"],
                path=kernel_path,
                group=KernelGroup(row["group"]),
                optimized=bool(row.get("optimized", False)),
                shapes=tuple(dict(shape) for shape in shape_rows),
                description=row.get("description", ""),
                tags=tuple(row.get("tags", ())),
                build_inputs_name=row.get("build_inputs", "build_inputs"),
                model_name=row.get("model", "Model"),
                reference_name=row.get("reference", "reference"),
            )
        )
    return tuple(specs)


def iter_kernel_specs(
    *,
    user_manifest: str | Path | None = None,
    include_builtin: bool = True,
    groups: Iterable[KernelGroup | str] | None = None,
    names: Iterable[str] | None = None,
) -> tuple[KernelSpec, ...]:
    """Return registry entries filtered by group/name."""
    specs: list[KernelSpec] = []
    if include_builtin:
        specs.extend(_BUILTIN_KERNELS)
    if user_manifest is not None:
        specs.extend(load_user_registry(user_manifest))

    wanted_groups = {KernelGroup(g) for g in groups} if groups is not None else None
    wanted_names = set(names) if names is not None else None

    filtered = []
    for spec in specs:
        if wanted_groups is not None and spec.group not in wanted_groups:
            continue
        if wanted_names is not None and spec.name not in wanted_names:
            continue
        filtered.append(spec)
    return tuple(filtered)


def get_kernel(
    name: str,
    *,
    user_manifest: str | Path | None = None,
    include_builtin: bool = True,
) -> KernelSpec:
    """Fetch one kernel by registry name."""
    for spec in iter_kernel_specs(
        user_manifest=user_manifest,
        include_builtin=include_builtin,
        names=(name,),
    ):
        return spec
    raise KeyError(f"unknown Stage-B kernel: {name}")
