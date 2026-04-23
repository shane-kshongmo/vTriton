#!/usr/bin/env python3

import argparse
import ast
import json
import numbers
import os
import runpy
import shutil
import sys
import types
from pathlib import Path


_METADATA_FILENAME = "tritonsim_hivm_bindings.jsonl"
_COMPILE_COMMANDS_FILENAME = "tritonsim_hivm_compile_commands.jsonl"


def _pick_existing_dir(candidates):
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return str(path)
    return ""


def _normalize_ascend_env():
    """Fill Ascend runtime paths so compile-only lowering works in mock mode.

    Some CANN layouts expose headers/libs under `<prefix>/x86_64-linux` but keep
    runtime/opp under the parent toolkit directory. Resolve both shapes.
    """
    roots = []
    for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        value = os.environ.get(key, "")
        if value:
            roots.append(Path(value))

    # torch_npu validates runtime/compiler under ASCEND_HOME_PATH directly.
    # If callers provide `<toolkit>/x86_64-linux`, rewrite to toolkit root.
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    if ascend_home:
        ascend_home_path = Path(ascend_home)
        if not (ascend_home_path / "runtime").exists() and ascend_home_path.name == "x86_64-linux":
            toolkit_root = ascend_home_path.parent
            if (toolkit_root / "runtime").exists() and (toolkit_root / "compiler").exists():
                os.environ["ASCEND_HOME_PATH"] = str(toolkit_root)
                os.environ["ASCEND_TOOLKIT_HOME"] = str(toolkit_root)
                os.environ["ASCEND_AICPU_PATH"] = str(toolkit_root)
                roots.append(toolkit_root)

    runtime = os.environ.get("ASCEND_RUNTIME_PATH", "")
    if not runtime or not Path(runtime).exists():
        runtime_candidates = []
        for root in roots:
            runtime_candidates.append(root / "runtime")
            if root.name == "x86_64-linux":
                runtime_candidates.append(root.parent / "runtime")
        resolved = _pick_existing_dir(runtime_candidates)
        if resolved:
            os.environ["ASCEND_RUNTIME_PATH"] = resolved

    opp = os.environ.get("ASCEND_OPP_PATH", "")
    if not opp or not Path(opp).exists():
        opp_candidates = []
        for root in roots:
            opp_candidates.append(root / "opp")
            if root.name == "x86_64-linux":
                opp_candidates.append(root.parent / "opp")
        resolved = _pick_existing_dir(opp_candidates)
        if resolved:
            os.environ["ASCEND_OPP_PATH"] = resolved


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


def _install_compile_command_capture(dump_dir):
    import subprocess

    original_run = subprocess.run
    original_check_call = subprocess.check_call
    commands_output = Path(dump_dir) / _COMPILE_COMMANDS_FILENAME
    _npuir_index = [0]
    # Asks bishengir-compile to print HIVM IR after the sync-injection pass.
    _NPUIR_FLAG = "--bishengir-print-ir-after=hivm-inject-sync"
    # Marker for the start/end of the IR dump in bishengir output.
    _NPUIR_START = "// -----// IR Dump After"
    _NPUIR_END = "// -----// IR Dump Before"

    def _is_npu_compiler(cmd):
        if not isinstance(cmd, (list, tuple)) or not cmd:
            return False
        return Path(str(cmd[0])).name in ("bishengir-compile", "npu-compile")

    def _extract_mlir_module(text):
        """Extract just the MLIR module from bishengir IR dump output."""
        lines = text.splitlines()
        start_idx = None
        end_idx = None

        for i, line in enumerate(lines):
            if _NPUIR_START in line:
                # Module starts after the next line (skip the header line itself).
                start_idx = i + 1
            elif start_idx is not None and _NPUIR_END in line:
                end_idx = i
                break

        if start_idx is None:
            # No header marker found; look for the first 'func.func' as fallback.
            for i, line in enumerate(lines):
                if line.strip().startswith("func.func"):
                    start_idx = i
                    break

        if start_idx is not None:
            module_lines = lines[start_idx:end_idx]
            if end_idx is None:
                balanced = []
                depth = 0
                saw_body = False
                for line in module_lines:
                    balanced.append(line)
                    depth += line.count("{")
                    depth -= line.count("}")
                    saw_body = saw_body or "{" in line
                    if saw_body and depth <= 0:
                        break
                module_lines = balanced
            # Strip leading/trailing blank lines.
            while module_lines and not module_lines[0].strip():
                module_lines.pop(0)
            while module_lines and not module_lines[-1].strip():
                module_lines.pop()
            return "\n".join(module_lines)
        return None

    def _find_bishengir_opt():
        found = shutil.which("bishengir-opt")
        if found:
            return found
        roots = [
            os.environ.get("ASCEND_HOME_PATH"),
            os.environ.get("ASCEND_TOOLKIT_HOME"),
            "/home/shane/Ascend/ascend-toolkit/latest",
            "/home/shane/Ascend/cann-8.5.0",
        ]
        for root in roots:
            if not root:
                continue
            candidate = Path(root) / "tools" / "bishengir" / "bin" / "bishengir-opt"
            if candidate.exists():
                return str(candidate)
        return None

    def _normalize_npuir(module_text):
        """Print through bishengir-opt so vTriton can parse generic MLIR."""
        import tempfile

        opt = _find_bishengir_opt()
        if not opt:
            return module_text
        with tempfile.NamedTemporaryFile("w", suffix=".npuir.mlir", delete=False) as src:
            src.write(module_text)
            src_path = src.name
        out_path = src_path + ".generic"
        try:
            ret = original_run(
                [
                    opt,
                    "--allow-unregistered-dialect",
                    "--mlir-print-op-generic",
                    src_path,
                    "-o",
                    out_path,
                ],
                capture_output=True,
                timeout=30,
            )
            if ret.returncode == 0 and Path(out_path).exists():
                return Path(out_path).read_text(encoding="utf-8")
        except Exception:
            pass
        finally:
            Path(src_path).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)
        return module_text

    def _dump_npuir_secondary(cmd):
        """Run bishengir-compile a second time with NPUIR dump flag only."""
        import tempfile

        cmd_with_flag = list(cmd) + [_NPUIR_FLAG]
        # Keep the -o flag (required by bishengir-compile) but redirect to a temp file.
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_bin:
                cmd_with_output = []
                skip_next = False
                for i, arg in enumerate(cmd_with_flag):
                    if skip_next:
                        skip_next = False
                        continue
                    if arg == "-o":
                        cmd_with_output.extend(["-o", tmp_bin.name])
                        skip_next = True
                    elif arg.startswith("-o"):
                        cmd_with_output.append("-o=" + tmp_bin.name)
                    else:
                        cmd_with_output.append(arg)

            ret = original_run(cmd_with_output, capture_output=True, timeout=30)
            Path(tmp_bin.name).unlink(missing_ok=True)
            combined = (ret.stdout or b"") + b"\n" + (ret.stderr or b"")
            text = combined.decode("utf-8", errors="replace")
            module_text = _extract_mlir_module(text)
            if module_text and "func.func" in module_text:
                _npuir_index[0] += 1
                fname = f"kernel_{_npuir_index[0]:03d}.npuir.mlir"
                module_text = _normalize_npuir(module_text)
                (Path(dump_dir) / fname).write_text(module_text, encoding="utf-8")
                return True
        except Exception:
            # Dump failed; primary compilation already succeeded, so ignore.
            pass
        return False

    def record(cmd):
        if not isinstance(cmd, (list, tuple)) or not cmd:
            return
        tool = Path(str(cmd[0])).name
        if tool not in ("bishengir-compile", "npu-compile"):
            return
        if "--help" in cmd:
            return
        with commands_output.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"tool": tool, "command": [str(part) for part in cmd]}))
            f.write("\n")

    def wrapped_run(*args, **kwargs):
        if not args:
            return original_run(*args, **kwargs)

        cmd = args[0]
        record(cmd)

        if not _is_npu_compiler(cmd):
            return original_run(*args, **kwargs)

        # Run the original compile to completion.
        ret = original_run(*args, **kwargs)

        # If successful, run a secondary dump-only invocation.
        if ret.returncode == 0:
            if _dump_npuir_secondary(cmd):
                _exit_success()

        return ret

    def wrapped_check_call(*args, **kwargs):
        if args:
            record(args[0])
        return original_check_call(*args, **kwargs)

    subprocess.run = wrapped_run
    subprocess.check_call = wrapped_check_call


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

    try:
        import torch_npu  # noqa: F401
        return
    except Exception:
        pass

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


def _install_compile_only_mock():
    """Activate compile-only mock so torch factory functions redirect to CPU.

    Tries the installed triton.backends.ascend package first, then falls back
    to the patched submodule source tree. This allows the launcher to work even
    when triton-ascend is installed as an unpatched pip wheel.
    """
    try:
        from triton.backends.ascend.compile_only_mock import install
        install()
        return
    except Exception:
        pass

    mock_path = (
        Path(__file__).resolve().parent.parent.parent
        / "thirdparty" / "triton-ascend"
        / "third_party" / "ascend" / "backend"
        / "compile_only_mock.py"
    )
    if mock_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "triton_ascend_compile_only_mock", mock_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "install"):
            mod.install()
        return

    print(
        "Warning: compile_only_mock not found — HIVM IR dump may fail without NPU hardware.",
        file=sys.stderr,
    )


def _install_arch_override():
    import os

    arch = os.environ.get("TRITON_ASCEND_ARCH")
    if not arch:
        return

    try:
        from triton.backends.ascend.driver import NPUDriver, NPUUtils
    except Exception:
        return

    NPUUtils.get_arch = lambda self: arch
    NPUUtils.get_aicore_num = lambda self: 20
    NPUUtils.get_aivector_core_num = lambda self: 40
    NPUUtils.get_device_properties = lambda self, device: {
        "max_shared_mem": 1,
        "num_aicore": 20,
        "num_vectorcore": 40,
    }
    NPUDriver.get_current_device = lambda self: 0
    NPUDriver.set_current_device = lambda self, device: None
    NPUDriver.get_current_stream = lambda self, device=None: 0
    NPUDriver.get_device_properties = lambda self, device: {
        "max_shared_mem": 1,
        "num_aicore": 20,
        "num_vectorcore": 40,
    }
    # Skip npu_utils.cpp JIT-compilation in compile-only mode; mirrors the
    # driver.py source patch for installed (unpatched) wheels.
    NPUUtils.__init__ = lambda self: setattr(self, "npu_utils_mod", None)
    try:
        from triton.runtime.driver import driver
        active = driver.active
        active._initialize_obj()
        active._obj.get_current_device = lambda: 0
        active._obj.set_current_device = lambda device: None
        active._obj.get_current_stream = lambda device=None: 0
        active._obj.utils.get_arch = lambda: arch
        active._obj.utils.get_aicore_num = lambda: 20
        active._obj.utils.get_aivector_core_num = lambda: 40
        active._obj.get_device_properties = lambda device: {
            "max_shared_mem": 1,
            "num_aicore": 20,
            "num_vectorcore": 40,
        }
    except Exception:
        pass


def _install_header_compat():
    """Patch triton-ascend header template for torch_npu/CANN API skew.

    Some torch_npu headers reference `aclOpExecutor`, declared in
    `aclnn/acl_meta.h`, but the generated precompiled header only includes
    `acl/acl.h`. Inject the missing include in compile-only launcher mode.
    """
    try:
        from triton.backends.ascend import driver as ascend_driver
    except Exception:
        return

    original_generate = getattr(ascend_driver, "generate_npu_header_src", None)
    original_make_stub = getattr(ascend_driver, "make_npu_launcher_stub", None)
    if original_generate is None or original_make_stub is None:
        return

    def _inject_acl_meta(header):
        marker = "#include <acl/acl.h>"
        inject = "#include <aclnn/acl_meta.h>"
        stamp = "// tritonsim aclOpExecutor compatibility"
        if marker in header and inject not in header:
            header = header.replace(marker, f"{marker}\n{inject}")
        if stamp not in header:
            header = header.replace("#endif", f"{stamp}\n#endif")
        return header

    def wrapped_generate_npu_header_src():
        return _inject_acl_meta(original_generate())

    def wrapped_make_npu_launcher_stub(header_src, wrapper_src, debug=False):
        return original_make_stub(_inject_acl_meta(header_src), wrapper_src, debug)

    ascend_driver.generate_npu_header_src = wrapped_generate_npu_header_src
    ascend_driver.make_npu_launcher_stub = wrapped_make_npu_launcher_stub
    launcher_init = getattr(getattr(ascend_driver, "NPULauncher", None), "__init__", None)
    if launcher_init is not None:
        launcher_init.__globals__["generate_npu_header_src"] = wrapped_generate_npu_header_src
        launcher_init.__globals__["make_npu_launcher_stub"] = wrapped_make_npu_launcher_stub


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Launch a Triton DSL script in compile-only dump mode."
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


def _exit_success():
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main():
    args = _parse_args()
    script_path = Path(args.script).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Triton script does not exist: {script_path}")

    forwarded_args = list(args.script_args)
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    os.environ.setdefault("TRITON_ENABLE_TASKQUEUE", "0")
    _normalize_ascend_env()
    _install_compile_only_mock()
    _install_header_compat()
    _install_compile_command_capture(args.dump_dir)
    _install_capture_hook(args.dump_dir)
    _install_torch_npu_compat()
    _install_arch_override()

    sys.path.insert(0, str(script_path.parent))
    sys.argv = [str(script_path), *forwarded_args]
    if not args.entry_func:
        runpy.run_path(str(script_path), run_name="__main__")
        _exit_success()

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
    _exit_success()


if __name__ == "__main__":
    main()
