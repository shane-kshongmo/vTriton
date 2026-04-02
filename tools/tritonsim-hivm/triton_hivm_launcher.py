#!/usr/bin/env python3

import argparse
import numbers
import runpy
import sys
from pathlib import Path


_METADATA_FILENAME = "tritonsim_hivm_bindings.txt"


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


def _write_bindings(dump_dir, entries):
    output = Path(dump_dir) / _METADATA_FILENAME
    output.write_text(",".join(entries), encoding="utf-8")


def _capture_entries_from_params(params, bound_args, grid):
    entries = []
    for index, param in enumerate(params):
        if getattr(param, "is_constexpr", False):
            continue
        value = bound_args.get(param.name)
        if not _is_scalar_value(value):
            continue
        entries.append(f"arg{index}={_format_scalar(value)}")

    if grid is not None:
        entries.extend(["pid_x=0", "pid_y=0", "pid_z=0"])
    return entries


def _install_capture_hook(dump_dir):
    from triton.runtime.jit import JITFunction

    original_run = JITFunction.run

    def wrapped_run(self, *args, **kwargs):
        from triton.compiler import make_backend
        from triton.runtime.driver import driver

        grid = kwargs.get("grid")
        if self.binder is None:
            target = driver.active.get_current_target()
            backend = make_backend(target)
            self.create_binder(backend)

        bound_args, _, _, _, _ = self.binder(*args, **kwargs)
        entries = _capture_entries_from_params(self.params, bound_args, grid)
        if entries:
            _write_bindings(dump_dir, entries)

        return original_run(self, *args, **kwargs)

    JITFunction.run = wrapped_run


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Launch a Triton DSL script and capture inferred HIVM bindings."
    )
    parser.add_argument("--script", required=True, help="Path to the Triton Python script")
    parser.add_argument("--dump-dir", required=True, help="Directory where Triton dumps IR")
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the target script after `--`",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    script_path = Path(args.script).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Triton script does not exist: {script_path}")

    forwarded_args = list(args.script_args)
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    _install_capture_hook(args.dump_dir)

    sys.path.insert(0, str(script_path.parent))
    sys.argv = [str(script_path), *forwarded_args]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
