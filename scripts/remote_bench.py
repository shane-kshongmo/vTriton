#!/usr/bin/env python3
"""
remote_bench.py — Remote 910B3 validation runner for A.6.1 / A.6.2

Syncs local repo → remote 910B3, runs msprof profiling, fetches CSV back.
Supports both prebuilt ELF binaries and Triton/Python kernels via
kernel_launcher.py.

Usage:
    python scripts/remote_bench.py --host <remote> --kernel <name> --output <csv_path>
    python scripts/remote_bench.py --kernel <name> --output <csv_path>  # uses env vars

Responsibilities:
- SSH sync local repo → remote 910B3 under the triton_hxl conda env
- Source CANN env (/usr/local/Ascend/ascend-toolkit/set_env.sh by default;
  override via VTRITON_REMOTE_CANN_SETENV / VTRITON_REMOTE_CONDA_ENV)
- Export PYTHONPATH for triton-ascend
- Run msprof --application=<exe> --output=<msprof_dir> on remote
  (or msprof --application="python kernel_launcher.py ..." for Triton kernels)
- Locate op_summary_*.csv via find (with rglob fallback)
- Sync CSV + optional kernel_output.npy back to a local temp path
- Return the local CSV path to the caller

Not in scope:
- SSH key management (assumes ssh-agent or key already configured)
- Remote CANN install (assumes CANN already installed)

Host configuration (preference order):
1. CLI args (--host, --remote-path)
2. Environment variables (VTRITON_REMOTE_HOST, VTRITON_REMOTE_PATH)
3. Config file (~/.vtriton_remote, INI format: [remote]\nhost=...\npath=...)

Source spec: .omc/plans/a6_validation_harness.md §5, .omc/plans/a6_2_blockers_scope.md
"""

from __future__ import annotations

import argparse
import configparser
import os
import subprocess
import sys
import tempfile
from pathlib import Path


# ── Remote environment preamble (shared across all ssh scripts) ────────
# Sources CANN, activates the conda env, and exports PYTHONPATH for
# triton-ascend. Every ssh script fragment starts with this preamble.
#
# Defaults match the real 910B3 box: CANN 9.0.0 under ascend-toolkit and the
# 'triton_hxl' conda env. Both are overridable via env vars so the runner is
# not brittle across machines.
_DEFAULT_CANN_SETENV = "/usr/local/Ascend/ascend-toolkit/set_env.sh"
_DEFAULT_CONDA_ENV = "triton_hxl"


def _remote_cann_setenv() -> str:
    """CANN set_env.sh path on the remote (overridable)."""
    return os.environ.get("VTRITON_REMOTE_CANN_SETENV", _DEFAULT_CANN_SETENV)


def _remote_conda_env() -> str:
    """Conda env to activate on the remote (overridable)."""
    return os.environ.get("VTRITON_REMOTE_CONDA_ENV", _DEFAULT_CONDA_ENV)


def _remote_preamble() -> str:
    """Source CANN + conda + activate the env (single-brace shell groups).

    Note: inside this f-string, ``{{`` / ``}}`` emit literal ``{`` / ``}`` so
    the generated shell uses valid ``{ cmd; }`` group syntax for the ``||``
    fallbacks.
    """
    cann_setenv = _remote_cann_setenv()
    conda_env = _remote_conda_env()
    return (
        f"source {cann_setenv} || {{ echo 'CANN env not found' >&2; exit 1; }}"
        f" && source $(conda info --base)/etc/profile.d/conda.sh || {{ echo 'conda not found' >&2; exit 1; }}"
        f" && conda activate {conda_env} || {{ echo 'conda activate {conda_env} failed' >&2; exit 1; }}"
    )


def _remote_env_preamble(remote_path: str) -> str:
    """Build the full remote preamble with cd + CANN + conda + PYTHONPATH.

    ``remote_path`` is interpolated unquoted into the generated shell so a
    leading ``~`` still tilde-expands on the remote (the default is
    ``~/vTriton``); it must therefore be a shell-safe path with no spaces or
    metacharacters.  Configured machine paths satisfy this.
    """
    return (
        f"cd {remote_path}"
        f" && {_remote_preamble()}"
        f" && export PYTHONPATH={remote_path}/thirdparty/triton-ascend/python:$PYTHONPATH"
    )


# ── Host configuration ──────────────────────────────────────────────────

def resolve_remote_config(
    cli_host: str | None = None,
    cli_remote_path: str | None = None,
) -> tuple[str | None, str]:
    """Resolve remote host/path from CLI args → env vars → config file.

    Preference order:
    1. CLI args (--host, --remote-path)
    2. Environment variables (VTRITON_REMOTE_HOST, VTRITON_REMOTE_PATH)
    3. Config file (~/.vtriton_remote, INI format)

    Returns:
        (remote_host, remote_path). remote_host may be None if unresolved.
    """
    host = cli_host
    path = cli_remote_path or "~/vTriton"

    # Level 2: environment variables
    if not host:
        host = os.environ.get("VTRITON_REMOTE_HOST")
    if cli_remote_path is None:
        path = os.environ.get("VTRITON_REMOTE_PATH", path)

    # Level 3: config file
    if not host:
        config_path = Path.home() / ".vtriton_remote"
        if config_path.exists():
            cfg = configparser.ConfigParser()
            cfg.read(config_path)
            if cfg.has_section("remote"):
                host = host or cfg.get("remote", "host", fallback=None)
                if cli_remote_path is None:
                    path = cfg.get("remote", "path", fallback=path)

    return host, path


# Heavy/derived paths never needed on the remote (triton is installed in the
# conda env, so the in-repo thirdparty checkout is redundant).  Without these
# excludes the sync pulls tens of GB (.git ~8G, thirdparty ~16G).
_SYNC_EXCLUDES = (
    ".git", "build", "thirdparty", ".claude", ".omc",
    "__pycache__", "*.pyc",
)


def sync_to_remote(
    local_path: Path,
    remote_host: str,
    remote_path: str,
) -> None:
    """Sync local directory to remote via rsync (excludes heavy/derived paths).

    Args:
        local_path: Local directory to sync.
        remote_host: SSH host (user@hostname).
        remote_path: Remote destination path.
    """
    cmd = ["rsync", "-az", "--delete"]
    for pat in _SYNC_EXCLUDES:
        cmd.extend(["--exclude", pat])
    cmd.extend([
        f"{local_path}/",
        f"{remote_host}:{remote_path}/",
    ])
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def run_msprof_remote(
    remote_host: str,
    remote_path: str,
    kernel_exe: str,
    msprof_dir: str = "msprof_output",
) -> None:
    """Run msprof profiling on remote for a prebuilt ELF binary.

    Args:
        remote_host: SSH host.
        remote_path: Remote repo path.
        kernel_exe: Path to kernel executable on remote.
        msprof_dir: Output directory for msprof (relative to remote_path).
    """
    script = (
        f"{_remote_env_preamble(remote_path)}"
        f" && command -v msprof >/dev/null 2>&1 || {{ echo 'msprof not found' >&2; exit 1; }}"
        f" && msprof --application={kernel_exe} --output={msprof_dir}"
    )
    cmd = ["ssh", remote_host, script]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def run_python_kernel_remote(
    remote_host: str,
    remote_path: str,
    kernel_script: str,
    msprof_dir: str = "msprof_output",
    output_dir: str = "kernel_outputs",
    iters: int = 10,
) -> None:
    """Run msprof profiling on remote for a Triton/Python kernel.

    Deploys kernel_launcher.py and wraps it with msprof.

    Args:
        remote_host: SSH host.
        remote_path: Remote repo path.
        kernel_script: Relative path to kernel .py file on remote.
        msprof_dir: Output directory for msprof (relative to remote_path).
        output_dir: Directory for kernel output .npy files.
        iters: Number of kernel iterations to run.
    """
    launcher = f"{remote_path}/scripts/kernel_launcher.py"
    script = (
        f"{_remote_env_preamble(remote_path)}"
        f" && command -v msprof >/dev/null 2>&1 || {{ echo 'msprof not found' >&2; exit 1; }}"
        f" && mkdir -p {output_dir}"
        f" && msprof --application=\"python {launcher}"
        f" --kernel {kernel_script}"
        f" --output-dir {output_dir}"
        f" --iters {iters}\""
        f" --output={msprof_dir}"
    )
    cmd = ["ssh", remote_host, script]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def fetch_csv_from_remote(
    remote_host: str,
    remote_path: str,
    msprof_dir: str,
    local_output: Path,
) -> Path:
    """Fetch op_summary CSV from remote.

    Args:
        remote_host: SSH host.
        remote_path: Remote repo path.
        msprof_dir: msprof output directory (relative to remote_path).
        local_output: Local path to write CSV.

    Returns:
        Path to local CSV file.

    Raises:
        FileNotFoundError: No op_summary CSV found on remote.
    """
    # Find CSV on remote (try find first, fall back to rglob pattern)
    find_cmd = (
        f"find {remote_path}/{msprof_dir} -name 'op_summary_*.csv' 2>/dev/null | head -n 1"
        f" || python3 -c \"import pathlib; cs=list(pathlib.Path('{remote_path}/{msprof_dir}').rglob('op_summary_*.csv')); print(cs[0] if cs else '')\""
    )
    result = subprocess.run(
        ["ssh", remote_host, find_cmd],
        check=True, capture_output=True, text=True,
    )
    remote_csv = result.stdout.strip()
    if not remote_csv:
        raise FileNotFoundError(f"No op_summary CSV found in {remote_path}/{msprof_dir}")

    # Fetch CSV
    local_output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["scp", f"{remote_host}:{remote_csv}", str(local_output)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    return local_output


def fetch_output_from_remote(
    remote_host: str,
    remote_path: str,
    output_dir: str,
    local_output: Path,
) -> Path | None:
    """Fetch kernel output .npy from remote.

    Args:
        remote_host: SSH host.
        remote_path: Remote repo path.
        output_dir: Remote directory containing kernel outputs.
        local_output: Local path to write the .npy file.

    Returns:
        Path to local .npy file, or None if not found.
    """
    remote_npy = f"{remote_path}/{output_dir}/kernel_output.npy"
    # Check if the file exists on remote
    check_cmd = f"test -f {remote_npy} && echo exists"
    result = subprocess.run(
        ["ssh", remote_host, check_cmd],
        capture_output=True, text=True,
    )
    if "exists" not in result.stdout:
        return None

    local_output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["scp", f"{remote_host}:{remote_npy}", str(local_output)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return local_output


def _validate_bishengir_input(hivm_path: Path) -> None:
    """Reject analytical DES JSON before invoking bishengir-compile.

    The counterfactual model can mutate DES JSON for analysis, but
    bishengir-compile consumes compiler IR.  Failing here prevents a remote run
    from pretending an analytical JSON edit is hardware-reachable.
    """
    name = hivm_path.name
    if name.endswith(".json"):
        raise ValueError(
            "bishengir-compile accepts MLIR/NPUIR, not DES JSON; "
            f"got {hivm_path}"
        )
    if not name.endswith(".mlir"):
        raise ValueError(
            "edited compiler input must be an MLIR/NPUIR file ending in .mlir; "
            f"got {hivm_path}"
        )


def recompile_remote(
    remote_host: str,
    remote_path: str,
    hivm_path: Path,
    kernel_name: str,
    bishengir_path: str | None = None,
) -> str:
    """Recompile an edited HIVM on the remote 910B3.

    Uploads the edited HIVM file, runs bishengir to compile it, and
    returns the path to the compiled binary on the remote.

    Args:
        remote_host: SSH host (user@hostname).
        remote_path: Remote repo path.
        hivm_path: Local path to edited HIVM file.
        kernel_name: Kernel identifier (used for output binary naming).
        bishengir_path: Path to bishengir on remote (default: auto-detect).

    Returns:
        Remote path to compiled binary.

    Raises:
        RuntimeError: If compilation fails on remote.
    """
    _validate_bishengir_input(hivm_path)

    remote_hivm = f"{remote_path}/tmp_edits/{hivm_path.name}"
    remote_bin = f"{remote_path}/build/bin/{kernel_name}"

    if bishengir_path is None:
        # bishengir-compile is on PATH once CANN is sourced (cann-9.0.0/bin).
        bishengir_path = "bishengir-compile"

    # Upload edited HIVM
    subprocess.run(
        ["ssh", remote_host, f"mkdir -p {remote_path}/tmp_edits"],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ["scp", str(hivm_path), f"{remote_host}:{remote_hivm}"],
        check=True, capture_output=True, text=True,
    )

    # Compile on remote
    script = (
        f"{_remote_env_preamble(remote_path)}"
        f" && {bishengir_path} {remote_hivm} -o {remote_bin}"
    )
    result = subprocess.run(
        ["ssh", remote_host, script],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"bishengir compilation failed (exit {result.returncode}): "
            f"{result.stderr[:500]}"
        )

    return remote_bin


def run_remote_bench(
    remote_host: str,
    kernel_name: str,
    kernel_script: Path | None = None,
    output_csv: Path | None = None,
    remote_path: str = "~/vTriton",
    hivm_in: Path | None = None,
    output_npy: Path | None = None,
) -> tuple[Path, Path | None]:
    """Run remote validation benchmark.

    Args:
        remote_host: SSH host (user@hostname).
        kernel_name: Kernel identifier.
        kernel_script: Path to kernel run script (.py for Triton kernels).
        output_csv: Local path for output CSV (auto-generated if None).
        remote_path: Remote repo path.
        hivm_in: Path to edited NPUIR/MLIR file (triggers recompile before profiling).
        output_npy: Local path for kernel output .npy (auto-generated if None).

    Returns:
        (csv_path, npy_path) tuple. npy_path may be None if no output was produced.
    """
    local_repo = Path(__file__).parent.parent

    if output_csv is None:
        tmpdir = Path(tempfile.mkdtemp())
        output_csv = tmpdir / f"{kernel_name}_op_summary.csv"
    if output_npy is None:
        tmpdir = output_csv.parent
        output_npy = tmpdir / f"{kernel_name}_output.npy"

    # Sync local → remote
    sync_to_remote(local_repo, remote_host, remote_path)

    # Recompile if compiler-facing IR is provided.
    if hivm_in is not None:
        recompile_remote(
            remote_host, remote_path, hivm_in, kernel_name
        )

    # Run msprof on remote
    msprof_dir = "msprof_output"
    output_dir = "kernel_outputs"

    is_python_kernel = (
        kernel_script is not None
        and str(kernel_script).endswith(".py")
    )

    if is_python_kernel:
        # Triton/Python kernel: use kernel_launcher.py
        run_python_kernel_remote(
            remote_host, remote_path, str(kernel_script),
            msprof_dir, output_dir,
        )
    else:
        # Prebuilt ELF binary
        kernel_exe = f"{remote_path}/build/bin/{kernel_name}"
        run_msprof_remote(remote_host, remote_path, kernel_exe, msprof_dir)

    # Fetch CSV back
    csv_path = fetch_csv_from_remote(remote_host, remote_path, msprof_dir, output_csv)

    # Fetch output .npy (may not exist for ELF binaries)
    npy_path = fetch_output_from_remote(
        remote_host, remote_path, output_dir, output_npy
    )

    return csv_path, npy_path


def _cli():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Remote 910B3 validation runner for A.6.1 / A.6.2",
    )
    parser.add_argument("--host", help="Remote SSH host (user@hostname). Falls back to VTRITON_REMOTE_HOST env var.")
    parser.add_argument("--kernel", required=True, help="Kernel name or identifier")
    parser.add_argument("--output", required=True, help="Local output CSV path")
    parser.add_argument("--script", help="Kernel run script (optional; .py for Triton kernels)")
    parser.add_argument("--remote-path", default=None, help="Remote repo path (default: ~/vTriton)")
    parser.add_argument("--hivm-in", help="Edited NPUIR/MLIR file to recompile on remote (optional)")
    parser.add_argument("--output-npy", help="Local output .npy path for kernel output (optional)")

    args = parser.parse_args()

    # Resolve host from CLI → env → config file
    remote_host, remote_path = resolve_remote_config(args.host, args.remote_path)
    if not remote_host:
        print("Error: no remote host configured. Use --host, VTRITON_REMOTE_HOST env var, "
              "or ~/.vtriton_remote config file.", file=sys.stderr)
        sys.exit(1)

    kernel_script = Path(args.script) if args.script else None
    output_csv = Path(args.output)
    hivm_in = Path(args.hivm_in) if args.hivm_in else None
    output_npy = Path(args.output_npy) if args.output_npy else None

    try:
        csv_path, npy_path = run_remote_bench(
            remote_host=remote_host,
            kernel_name=args.kernel,
            kernel_script=kernel_script,
            output_csv=output_csv,
            remote_path=remote_path,
            hivm_in=hivm_in,
            output_npy=output_npy,
        )
        print(f"CSV fetched: {csv_path}")
        if npy_path:
            print(f"Output fetched: {npy_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli()
