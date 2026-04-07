#!/usr/bin/env python3

import argparse
import ast
import json
import numbers
import runpy
import sys
import types
from pathlib import Path


_METADATA_FILENAME = "tritonsim_hivm_bindings.jsonl"


def _is_scalar_value(value):
    if isinstance(value, bool):
        return True
    return isinstance(value, numbers.Real) and not isinstance(value, complex)


def _format_scalar(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    return repr(float(value))


def _append_bindings(dump_dir, kernel_name, entries):
    output = Path(dump_dir) / _METADATA_FILENAME
    record = {"kernel_name": kernel_name, "entries": entries}
    with output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True))
        f.write("\n")


def _compute_constant_param_nums(params, bound_args, backend):
    """Determine which params become compile-time constants.

    This emulates the logic in ``jit.py`` (``JITFunction.run``) where
    ``AttrsDescriptor.get_constants()`` detects params whose runtime value
    equals 1 (stride specialisation) and marks them as constants.  Those
    params are baked into the compiled binary and **excluded** from the
    struct passed to the HIVM kernel at launch time, so our positional
    ``argN`` indices must skip them too.
    """
    try:
        bound_vals = tuple(bound_args.values())
        attrs = backend.get_attrs_descriptor(params, bound_vals)
        constant_params = attrs.get_constants()  # {param_num: value}
        # Also include params whose value is None (nullptr — treated as
        # constants in jit.py:616).
        none_nums = {
            p.num
            for p, v in zip(params, bound_vals)
            if v is None and not getattr(p, "is_constexpr", False)
        }
        return set(constant_params.keys()) | none_nums
    except Exception:
        # If anything goes wrong (e.g. backend not fully initialised in
        # compile-only mode), fall back to an empty set so we degrade
        # gracefully to the old behaviour.
        return set()


def _capture_entries_from_params(params, bound_args, grid,
                                 constant_param_nums=None):
    if constant_param_nums is None:
        constant_param_nums = set()
    entries = []
    named_entries = []
    runtime_index = 0
    for param in params:
        if getattr(param, "is_constexpr", False):
            continue
        value = bound_args.get(param.name)
        # Params that the compiler bakes as constants are excluded from
        # the HIVM function signature.  Skip them when counting the
        # positional index so that ``argN`` aligns with the Nth user arg
        # the HIVM kernel actually receives.
        if param.num in constant_param_nums:
            # Still record the named binding for reference.
            if _is_scalar_value(value):
                named_entries.append(f"{param.name}={_format_scalar(value)}")
            continue
        current_index = runtime_index
        runtime_index += 1
        if not _is_scalar_value(value):
            continue
        formatted = _format_scalar(value)
        entries.append(f"arg{current_index}={formatted}")
        named_entries.append(f"{param.name}={formatted}")

    if grid is not None:
        # Grid dimensions are appended to the struct after user args and
        # appear as the last 3 user-visible args in the HIVM function.
        for i, name in enumerate(("pid_x", "pid_y", "pid_z")):
            entries.append(f"arg{runtime_index + i}=0")
            named_entries.append(f"{name}=0")
    return entries, named_entries


def _install_capture_hook(dump_dir):
    from triton.runtime.jit import JITFunction

    original_run = JITFunction.run

    def wrapped_run(self, *args, **kwargs):
        from triton.compiler import make_backend
        from triton.runtime.driver import driver

        grid = kwargs.get("grid")
        target = driver.active.get_current_target()
        backend = make_backend(target)
        if self.binder is None:
            self.create_binder(backend)

        bound_args, _, _, _, _ = self.binder(*args, **kwargs)
        constant_param_nums = _compute_constant_param_nums(
            self.params, bound_args, backend)
        entries, named_entries = _capture_entries_from_params(
            self.params, bound_args, grid, constant_param_nums)
        if entries:
            kernel_name = (
                getattr(getattr(self, "fn", None), "__name__", None)
                or getattr(self, "__name__", None)
                or getattr(self, "name", None)
                or ""
            )
            _append_bindings(dump_dir, kernel_name, entries + named_entries)

        return original_run(self, *args, **kwargs)

    JITFunction.run = wrapped_run


def _install_torch_npu_compat():
    import torch

    if hasattr(torch, "npu"):
        return

    class _MockStream:
        npu_stream = 0

        def synchronize(self):
            return None

    torch.npu = types.SimpleNamespace(
        current_device=lambda: 0,
        device_count=lambda: 1,
        set_device=lambda *a, **kw: None,
        synchronize=lambda *a, **kw: None,
        is_available=lambda: True,
        mem_get_info=lambda *a, **kw: (8 * 1024**3, 16 * 1024**3),
        empty_cache=lambda: None,
        manual_seed=lambda *a, **kw: None,
        manual_seed_all=lambda *a, **kw: None,
        current_stream=lambda *a, **kw: _MockStream(),
    )

    if not hasattr(torch.Tensor, "npu"):
        torch.Tensor.npu = lambda self, *a, **kw: self


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Launch a Triton DSL script and capture inferred HIVM bindings."
    )
    parser.add_argument("--script", required=True, help="Path to the Triton Python script")
    parser.add_argument("--dump-dir", required=True, help="Directory where Triton dumps IR")
    parser.add_argument(
        "--entry-func",
        help="Optional function inside the script to call explicitly instead of running __main__",
    )
    parser.add_argument(
        "--entry-arg",
        action="append",
        default=[],
        help="Argument expression passed to --entry-func; may be repeated",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the target script after `--`",
    )
    return parser.parse_args()


def _resolve_entry_arg(expr):
    try:
        return ast.literal_eval(expr)
    except Exception:
        pass

    import torch

    allowed_globals = {
        "torch": torch,
        "True": True,
        "False": False,
        "None": None,
    }
    return eval(expr, {"__builtins__": {}}, allowed_globals)


def main():
    args = _parse_args()
    script_path = Path(args.script).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Triton script does not exist: {script_path}")

    forwarded_args = list(args.script_args)
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    _install_capture_hook(args.dump_dir)
    _install_torch_npu_compat()

    sys.path.insert(0, str(script_path.parent))
    sys.argv = [str(script_path), *forwarded_args]
    if not args.entry_func:
        runpy.run_path(str(script_path), run_name="__main__")
        return

    module_globals = runpy.run_path(str(script_path), run_name="__tritonsim__")
    target = module_globals.get(args.entry_func)
    if target is None:
        raise AttributeError(
            f"Entry function `{args.entry_func}` not found in {script_path}"
        )
    if not callable(target):
        raise TypeError(
            f"Entry target `{args.entry_func}` is not callable in {script_path}"
        )

    resolved_args = [_resolve_entry_arg(expr) for expr in args.entry_arg]
    target(*resolved_args)


if __name__ == "__main__":
    main()
