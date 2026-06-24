#!/usr/bin/env python3
"""
kernel_launcher.py — Remote Triton kernel launcher for msprof profiling.

Loads a kernel module by path, builds inputs, runs the kernel under
torch_npu, and dumps the output tensor(s) to .npy for correctness
verification.

Usage (on remote 910B3):
    python scripts/kernel_launcher.py \\
        --kernel test/chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2.py \\
        --output-dir kernel_outputs \\
        --iters 10

Expected kernel module interface:
    - build_inputs() → dict of kwargs (tensors on 'npu')
    - Model class with forward(data) → tuple of output tensors
      OR a main() function

The launcher writes:
    <output-dir>/kernel_output_0.npy  (first output tensor)
    <output-dir>/kernel_output_1.npy  (second output tensor, if any)
    ...
    <output-dir>/kernel_output.npy    (alias for first output, for
                                       backward compat with fetch_output)

Source spec: .omc/plans/a6_2_blockers_scope.md Blocker 2 gap #2
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path


def load_kernel_module(kernel_path: str):
    """Dynamically import a Python kernel module from a file path.

    Args:
        kernel_path: Path to the .py file containing the kernel.

    Returns:
        The imported module object.

    Raises:
        FileNotFoundError: If kernel_path does not exist.
        ImportError: If the module cannot be loaded.
    """
    path = Path(kernel_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Kernel script not found: {path}")

    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def enable_ttadapter_override_patch() -> None:
    """Allow Triton-Ascend TTAdapter IR overrides in validation runs.

    Some Triton-Ascend builds find `TRITON_KERNEL_OVERRIDE` files for the
    `ttadapter` stage but their generic compiler parser only handles module
    objects and final binaries. The Ascend backend expects TTAdapter input as a
    string, so an opt-in parser patch is enough to let the normal
    TTAdapter-to-npubin pipeline compile and launch an overridden compiler IR.
    """
    if os.environ.get("TRITON_ACCEPT_TTADAPTER_OVERRIDE") not in {"1", "true", "TRUE"}:
        return

    try:
        import triton.compiler.compiler as triton_compiler
    except Exception as exc:
        raise RuntimeError(
            "TRITON_ACCEPT_TTADAPTER_OVERRIDE=1 requires triton.compiler.compiler"
        ) from exc

    current_parse = triton_compiler.parse
    if getattr(current_parse, "_accepts_ttadapter_override", False):
        return

    def parse_with_ttadapter(full_name, ext, context):
        if ext == "ttadapter":
            return Path(full_name).read_text()
        return current_parse(full_name, ext, context)

    parse_with_ttadapter._accepts_ttadapter_override = True
    triton_compiler.parse = parse_with_ttadapter


def run_kernel(module, iters: int = 10, warmup: int = 1) -> list:
    """Run the kernel from a loaded module and return output tensors.

    Probes the module for standard entry points in preference order:
    1. module.main() — if present, call it directly (returns outputs)
    2. module.build_inputs() + module.Model().forward(data) — standard
       Triton kernel pattern
    3. module.Model().forward(module.build_inputs()) — fallback

    Args:
        module: The imported kernel module.
        iters: Number of measured iterations to run.
        warmup: Number of warmup iterations to run before measured iterations.

    Returns:
        List of output tensors (torch.Tensor on CPU).
    """
    import torch

    # Strategy 1: module has a main() function
    if hasattr(module, "main"):
        return module.main()

    # Strategy 2: build_inputs + Model
    if not hasattr(module, "build_inputs"):
        raise RuntimeError(
            f"Kernel module {module.__name__} has no build_inputs() or main(). "
            f"The launcher requires one of these entry points."
        )

    data = module.build_inputs()

    if hasattr(module, "Model"):
        model = module.Model()
        # Warmup also catches compile errors early.
        for _ in range(warmup):
            _ = model.forward({k: v.clone() if hasattr(v, 'clone') else v
                               for k, v in data.items()})

        # Timed iterations
        outputs = None
        for _ in range(iters):
            # Rebuild data each iteration to avoid in-place mutation artifacts
            run_data = module.build_inputs()
            outputs = model.forward(run_data)

        if outputs is None:
            raise RuntimeError("Model.forward() returned None")

        # Normalize to list
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        elif isinstance(outputs, (tuple, list)):
            outputs = list(outputs)
        else:
            outputs = [outputs]

        return [o.cpu() if hasattr(o, 'cpu') else o for o in outputs]

    raise RuntimeError(
        f"Kernel module {module.__name__} has build_inputs() but no Model class "
        f"or main() function."
    )


def save_outputs(outputs: list, output_dir: str) -> list[Path]:
    """Save output tensors to .npy files.

    Args:
        outputs: List of output tensors (on CPU).
        output_dir: Directory to write .npy files.

    Returns:
        List of paths to written .npy files.
    """
    import numpy as np

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved = []
    for i, tensor in enumerate(outputs):
        arr = tensor.numpy() if hasattr(tensor, 'numpy') else np.asarray(tensor)
        npy_path = out_path / f"kernel_output_{i}.npy"
        np.save(npy_path, arr)
        saved.append(npy_path)

    # Backward-compat alias: kernel_output.npy → first output
    if saved:
        import shutil
        compat_path = out_path / "kernel_output.npy"
        shutil.copy2(saved[0], compat_path)

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Launch a Triton kernel for msprof profiling.",
    )
    parser.add_argument(
        "--kernel", required=True,
        help="Path to kernel .py file (e.g. test/chunk_kda_...py)",
    )
    parser.add_argument(
        "--output-dir", default="kernel_outputs",
        help="Directory to write output .npy files",
    )
    parser.add_argument(
        "--iters", type=int, default=10,
        help="Number of kernel iterations to run (default: 10)",
    )
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="Number of warmup iterations before measured iterations (default: 1)",
    )
    args = parser.parse_args()

    print(f"[kernel_launcher] Loading kernel from {args.kernel}", file=sys.stderr)
    enable_ttadapter_override_patch()
    module = load_kernel_module(args.kernel)

    print(
        f"[kernel_launcher] Running {args.warmup} warmup + "
        f"{args.iters} measured iterations...",
        file=sys.stderr,
    )
    t0 = time.time()
    outputs = run_kernel(module, iters=args.iters, warmup=args.warmup)
    elapsed = time.time() - t0
    print(f"[kernel_launcher] Done in {elapsed:.2f}s, "
          f"{len(outputs)} output(s)", file=sys.stderr)

    saved = save_outputs(outputs, args.output_dir)
    for p in saved:
        print(f"[kernel_launcher] Saved {p}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
