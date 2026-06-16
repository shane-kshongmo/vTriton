#!/usr/bin/env python3
"""Sync CCE microbenchmarks to 910B3, compile, run with msprof, and extract CSV outputs.

This script orchestrates the full CCE microbenchmark pipeline:
1. Sync .cce source files to remote 910B3
2. Compile kernels with the remote CANN compiler (`ccec` preferred, `ccecom` fallback)
3. Build the ACL host launcher with CANN's ascendc CMake helpers
4. Run benchmarks under msprof
5. Sync back msprof CSV outputs to local bench_output/

Source spec: .omc/specs/performance_bound_model.md §A.1
Related: perfbound/calibration/microbench/*.cce (kernel sources)
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

# Default configuration
DEFAULT_HOST = "910B3"
DEFAULT_REMOTE_WORKDIR = "/root/hxl/cce_bench"
DEFAULT_LOCAL_CCE_DIR = Path("perfbound/calibration/microbench")
DEFAULT_OUTPUT_DIR = Path("perfbound/calibration/bench_output")
DEFAULT_CANN_ENV = "/usr/local/Ascend/cann/set_env.sh"
DEFAULT_CANN_PACKAGE_PATH = "/usr/local/Ascend/ascend-toolkit/8.2.RC1.alpha003"
DEFAULT_LAUNCHER = "./out/bin/vt_a1_bench_launcher"
DEFAULT_SOC_VERSION = "Ascend910B1"
DIRECT_SSH = False

# CCE kernel list (order must match US-003 acceptance criteria)
CCE_KERNELS = [
    "cube_peak_fp16",
    "cube_peak_int8",
    "cube_peak_bf16",
    "vector_peak_elemwise_add",
    "vector_peak_elemwise_mul",
    "vector_peak_elemwise_max",
    "vector_peak_elemwise_min",
    "vector_peak_transcendental",
    "mte_gm_to_ub",
    "mte_ub_to_gm",
    "mte_gm_to_l1",
    "mte_l1_to_l0a",
    "mte_l0c_to_gm",
    "mte_hbm_allcore",
    "mandatory_handoff",
]

RUN_KERNELS = CCE_KERNELS

CUBE_KERNELS = {
    "cube_peak_fp16",
    "cube_peak_int8",
    "cube_peak_bf16",
    "mandatory_handoff",
    "mte_l0c_to_gm",
    "mte_hbm_allcore",
}

MTE_KERNELS = {
    "mte_gm_to_ub",
    "mte_ub_to_gm",
    "mte_gm_to_l1",
    "mte_l1_to_l0a",
    "mte_l0c_to_gm",
    "mte_hbm_allcore",
}

MANDATORY_HANDOFF_K_VALUES = [128, 256, 384, 512, 1024, 2048]

# SSH output sanitization (suppress banners)
IGNORED_OUTPUT_PREFIXES = (
    "Authorized users only.",
    "All activities may be monitored and reported.",
    "tar: Ignoring unknown extended header keyword",
)


def run(cmd: list[str], *, cwd: Path | None = None, capture_output: bool = False,
        check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a command with optional working directory and output capture."""
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def sanitize_output(text: str) -> str:
    """Remove SSH banners and tar warnings from output."""
    kept_lines = []
    for line in text.splitlines():
        if any(line.startswith(prefix) for prefix in IGNORED_OUTPUT_PREFIXES):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines).strip()


def compact_ssh_cmd(host: str, remote_command: str) -> list[str]:
    """Build SSH command with quiet flags."""
    cmd = [
        "ssh",
        "-q",
        "-S",
        "none",
        "-o",
        "ControlMaster=no",
    ]
    if DIRECT_SSH:
        cmd.extend(["-o", "ProxyCommand=none"])
    cmd.extend([host, f"bash -lc {shlex.quote(remote_command)}"])
    return cmd


def source_cann(command: str, cann_env: str = DEFAULT_CANN_ENV) -> str:
    """Prefix a remote command with CANN environment setup."""
    return f"source {shlex.quote(cann_env)} >/dev/null 2>&1 || true; {command}"


def ascend_runtime_lib_dirs(cann_package_path: str, remote_workdir: str) -> str:
    """Build runtime library search paths for CANN and installed kernels."""
    dirs = [f"{remote_workdir}/out/lib64", f"{remote_workdir}/out/lib"]
    dirs.extend(
        f"{cann_package_path}/{subdir}"
        for subdir in [
            "lib64",
            "fwkacllib/lib64",
            "runtime/lib64",
            "compiler/lib64",
            "atc/lib64",
            "aarch64-linux/lib64",
        ]
    )
    return ":".join(dirs)


def repo_root_from(path: Path) -> Path:
    """Find git repository root from a path."""
    result = run(["git", "rev-parse", "--show-toplevel"], cwd=path, capture_output=True)
    return Path(result.stdout.strip()).resolve()


def sync_cce_files(
    repo_root: Path,
    host: str,
    remote_workdir: str,
    *,
    dry_run: bool
) -> None:
    """Sync CCE source files to remote 910B3 using tar-over-SSH."""
    remote_workdir_quoted = shlex.quote(remote_workdir)
    ssh_cmd = compact_ssh_cmd(host, f"mkdir -p {remote_workdir_quoted}")

    # Build tar command for CCE files only
    local_cce_dir = repo_root / DEFAULT_LOCAL_CCE_DIR
    if not local_cce_dir.exists():
        raise SystemExit(f"CCE directory not found: {local_cce_dir}")

    tar_create_cmd = [
        "tar",
        "--exclude=.git",
        "--exclude=__pycache__",
        "-czf",
        "-",
        ".",
    ]

    tar_extract_cmd = [
        "ssh",
        "-q",
        "-S", "none",
        "-o", "ControlMaster=no",
    ]
    if DIRECT_SSH:
        tar_extract_cmd.extend(["-o", "ProxyCommand=none"])
    tar_extract_cmd.extend([
        host,
        f"mkdir -p {remote_workdir_quoted} && find {remote_workdir_quoted} -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {{}} + && tar -xzf - -C {remote_workdir_quoted}",
    ])

    if dry_run:
        print("[DRY-RUN] Would sync CCE files:")
        print(f"  Source: {local_cce_dir}")
        print(f"  Target: {host}:{remote_workdir}")
        print("+", shlex.join(tar_create_cmd), "|", shlex.join(tar_extract_cmd))
        return

    # Create remote directory
    mkdir_result = subprocess.run(ssh_cmd, check=False, text=True, capture_output=True)
    if mkdir_result.returncode != 0:
        combined = sanitize_output("\n".join(part for part in [mkdir_result.stdout, mkdir_result.stderr] if part))
        if combined:
            print(combined, file=sys.stderr)
        raise subprocess.CalledProcessError(mkdir_result.returncode, ssh_cmd)

    # Tar-over-SSH sync
    tar_env = os.environ.copy()
    tar_env["COPYFILE_DISABLE"] = "1"
    tar_env["COPY_EXTENDED_ATTRIBUTES_DISABLE"] = "1"

    create_proc = subprocess.Popen(
        tar_create_cmd,
        cwd=str(local_cce_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=tar_env
    )
    extract_proc = subprocess.Popen(tar_extract_cmd, stdin=create_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert create_proc.stdout is not None
    create_proc.stdout.close()
    extract_stdout, extract_stderr = extract_proc.communicate()
    create_stderr = create_proc.stderr.read() if create_proc.stderr is not None else b""
    extract_rc = extract_proc.returncode
    create_rc = create_proc.wait()

    if create_rc != 0 or extract_rc != 0:
        combined = sanitize_output(
            "\n".join(
                part
                for part in [
                    create_stderr.decode(errors="replace"),
                    extract_stdout.decode(errors="replace"),
                    extract_stderr.decode(errors="replace"),
                ]
                if part
            )
        )
        if combined:
            print(combined, file=sys.stderr)
        raise subprocess.CalledProcessError(extract_rc or create_rc, tar_extract_cmd if extract_rc != 0 else tar_create_cmd)

    print(f"✓ Synced {len(CCE_KERNELS)} CCE kernels to {host}:{remote_workdir}")


def preflight_remote(
    host: str,
    *,
    cann_env: str,
    launcher: str,
    require_compiler: bool,
    require_msprof: bool,
    dry_run: bool,
) -> dict[str, str]:
    """Detect remote CANN tools and fail early for missing compile/runtime pieces."""
    launcher_probe = (
        f"printf 'launcher=%s\\n' \"$(command -v {shlex.quote(launcher)} || true)\"; "
        if "/" not in launcher
        else f"printf 'launcher=%s\\n' {shlex.quote(launcher)}; "
    )
    check_cmd = source_cann(
        "printf 'ccec=%s\\n' \"$(command -v ccec || true)\"; "
        "printf 'ccecom=%s\\n' \"$(command -v ccecom || true)\"; "
        "printf 'opc=%s\\n' \"$(command -v opc || true)\"; "
        "printf 'msprof=%s\\n' \"$(command -v msprof || true)\"; "
        f"{launcher_probe}"
        "printf 'npu_smi=%s\\n' \"$(command -v npu-smi || true)\"",
        cann_env,
    )

    if dry_run:
        print("[DRY-RUN] Would preflight remote CANN tools")
        return {"compiler": "ccec", "msprof": "msprof", "launcher": launcher}

    result = run(compact_ssh_cmd(host, check_cmd), capture_output=True, check=False)
    if result.returncode != 0:
        combined = sanitize_output("\n".join(part for part in [result.stdout, result.stderr] if part))
        raise RuntimeError(f"Remote CANN preflight failed:\n{combined}")

    tools: dict[str, str] = {}
    for line in sanitize_output(result.stdout).splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            tools[key.strip()] = value.strip()

    compiler = tools.get("ccec") or tools.get("ccecom")
    if require_compiler and not compiler:
        raise RuntimeError("Remote CANN preflight failed: neither ccec nor ccecom is on PATH after set_env.sh")
    if require_msprof and not tools.get("msprof"):
        raise RuntimeError("Remote CANN preflight failed: msprof is not on PATH after set_env.sh")

    if compiler:
        tools["compiler"] = compiler
    if not tools.get("launcher"):
        print(f"⚠ Runtime launcher '{launcher}' not found; compile-only mode can still run.", file=sys.stderr)
    print(f"✓ Remote CANN preflight: compiler={compiler or 'not required'}, msprof={tools.get('msprof') or 'not required'}")
    return tools


def build_bench_launcher(
    host: str,
    remote_workdir: str,
    *,
    cann_env: str,
    cann_package_path: str,
    soc_version: str,
    dry_run: bool,
) -> bool:
    """Build the AscendC ACL runtime launcher on the remote host."""
    runtime_lib_dirs = ascend_runtime_lib_dirs(cann_package_path, remote_workdir)
    build_cmd = (
        f"cd {shlex.quote(remote_workdir)} && "
        f"export LD_LIBRARY_PATH={shlex.quote(runtime_lib_dirs)}:$LD_LIBRARY_PATH && "
        f"export LIBRARY_PATH={shlex.quote(runtime_lib_dirs)}:$LIBRARY_PATH && "
        "rm -rf build out && "
        "cmake -S . -B build "
        "-DRUN_MODE=npu "
        f"-DSOC_VERSION={shlex.quote(soc_version)} "
        f"-DASCEND_CANN_PACKAGE_PATH={shlex.quote(cann_package_path)} "
        "-DCMAKE_INSTALL_PREFIX=$PWD/out && "
        "cmake --build build -j$(nproc) && "
        "cmake --install build"
    )
    build_cmd = source_cann(build_cmd, cann_env)

    if dry_run:
        print("[DRY-RUN] Would build ACL launcher:")
        print(" ", build_cmd)
        return True

    result = run(compact_ssh_cmd(host, build_cmd), capture_output=True, check=False)
    output = sanitize_output("\n".join(part for part in [result.stdout, result.stderr] if part))
    if result.returncode != 0:
        print(output, file=sys.stderr)
        return False
    if output:
        print(output[-1200:])
    print("✓ Built AscendC ACL benchmark launcher")
    return True


def compile_cce_kernels(
    host: str,
    remote_workdir: str,
    *,
    cann_env: str,
    compiler: str,
    dry_run: bool
) -> dict[str, bool]:
    """Compile all CCE kernels on remote."""
    compile_results = {}

    for kernel in CCE_KERNELS:
        cce_file = f"{kernel}.cce"
        if Path(compiler).name == "ccecom":
            compile_cmd = f"cd {shlex.quote(remote_workdir)} && ccecom {cce_file} -o {kernel} --npu-size=1"
        else:
            arch = "dav-c220-cube" if kernel in CUBE_KERNELS else "dav-c220-vec"
            compile_cmd = (
                f"cd {shlex.quote(remote_workdir)} && "
                f"cann_root=$(cd \"$(dirname {shlex.quote(cann_env)})\" && pwd) && "
                f"{shlex.quote(compiler)} --cce-aicore-only --cce-aicore-lang "
                f"--cce-aicore-arch={arch} "
                f"-std=c++17 "
                f"-I\"${{cann_root}}/aarch64-linux/asc\" "
                f"-I\"${{cann_root}}/aarch64-linux/asc/include\" "
                f"-I\"${{cann_root}}/aarch64-linux/asc/include/basic_api\" "
                f"-I\"${{cann_root}}/aarch64-linux/asc/include/interface\" "
                f"-I\"${{cann_root}}/aarch64-linux/asc/impl/basic_api\" "
                f"-I\"${{cann_root}}/aarch64-linux/ascendc/include\" "
                f"-I\"${{cann_root}}/aarch64-linux/ascendc/include/basic_api/impl\" "
                f"-I\"${{cann_root}}/aarch64-linux/include\" "
                f"-O2 -c {shlex.quote(cce_file)} -o {shlex.quote(kernel + '.o')}"
            )
        compile_cmd = source_cann(compile_cmd, cann_env)

        if dry_run:
            print(f"[DRY-RUN] Would compile: {cce_file}")
            compile_results[kernel] = True
            continue

        try:
            result = run(compact_ssh_cmd(host, compile_cmd), capture_output=True)
            output = sanitize_output(result.stdout)
            if output:
                print(f"  {cce_file}: {output}")
            compile_results[kernel] = True
            print(f"✓ Compiled {cce_file}")
        except subprocess.CalledProcessError as e:
            compile_results[kernel] = False
            error_output = sanitize_output(e.stdout or e.stderr or "")
            print(f"✗ Failed to compile {cce_file}: {error_output}", file=sys.stderr)

    return compile_results


def run_cce_benchmark(
    host: str,
    remote_workdir: str,
    kernel: str,
    n_repeat: int = 30,
    *,
    output_name: str | None = None,
    launcher_extra_args: str = "",
    warmup_extra_args: str = "",
    cann_env: str,
    cann_package_path: str,
    kernel_timeout_sec: int,
    launcher: str,
    msprof: str,
    dry_run: bool
) -> bool:
    """Run a single CCE benchmark with msprof profiling."""
    # Build msprof command
    output_name = output_name or kernel
    msprof_output_dir = f"msprof_{output_name}"
    runtime_lib_dirs = ascend_runtime_lib_dirs(cann_package_path, remote_workdir)
    wrapper = f"run_{output_name}.sh"
    warmup_cmd = ""
    if warmup_extra_args:
        warmup_cmd = (
            f"{launcher} --kernel {kernel} --repeat 1 {warmup_extra_args}".rstrip()
            + " && "
        )
    wrapper_body = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -e",
            f"export LD_LIBRARY_PATH={runtime_lib_dirs}:$LD_LIBRARY_PATH",
            f"exec {launcher} --kernel {kernel} --repeat {n_repeat} {launcher_extra_args}".rstrip(),
            "",
        ]
    )
    application = f"./{wrapper}"
    run_cmd = (
        f"cd {shlex.quote(remote_workdir)} && "
        f"export LD_LIBRARY_PATH={shlex.quote(runtime_lib_dirs)}:$LD_LIBRARY_PATH && "
        f"test -x {shlex.quote(launcher)} && "
        f"{warmup_cmd}"
        f"printf '%s' {shlex.quote(wrapper_body)} > {shlex.quote(wrapper)} && "
        f"chmod +x {shlex.quote(wrapper)} && "
        f"rm -rf {shlex.quote(msprof_output_dir)} && "
        f"timeout --kill-after=10 {kernel_timeout_sec} "
        f"{shlex.quote(msprof)} --application={shlex.quote(application)} "
        f"--output={shlex.quote(msprof_output_dir)}"
    )
    run_cmd = source_cann(run_cmd, cann_env)

    if dry_run:
        print(f"[DRY-RUN] Would run benchmark: {output_name}")
        return True

    try:
        result = run(compact_ssh_cmd(host, run_cmd), capture_output=True, check=False)
        output = sanitize_output(result.stdout)

        # Check if msprof succeeded
        if result.returncode != 0:
            error_output = sanitize_output(result.stderr or "")
            if result.returncode == 124:
                error_output = f"timed out after {kernel_timeout_sec}s\n{error_output}".strip()
            print(f"✗ {kernel} benchmark failed (exit {result.returncode}): {error_output}", file=sys.stderr)
            return False

        print(f"✓ {output_name} benchmark completed")
        if output:
            print(f"  Output: {output[:200]}")  # Truncate long output
        return True

    except subprocess.CalledProcessError as e:
        error_output = sanitize_output(e.stdout or e.stderr or "")
        print(f"✗ {output_name} benchmark failed: {error_output}", file=sys.stderr)
        return False


def sync_csv_outputs(
    host: str,
    repo_root: Path,
    remote_workdir: str,
    output_dir: Path,
    *,
    kernels: list[str],
    dry_run: bool
) -> list[Path]:
    """Sync back msprof CSV outputs from remote."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = []

    for kernel in kernels:
        msprof_dir = f"msprof_{kernel}"
        local_csv_path = output_dir / f"{kernel}.csv"
        find_cmd = (
            f"csv=$(find {shlex.quote(remote_workdir + '/' + msprof_dir)} -type f "
            "\\( -name 'op_summary*.csv' -o -name '*op_summary*.csv' \\) "
            "| head -n 1) && test -n \"$csv\" && cat \"$csv\""
        )
        fetch_cmd = compact_ssh_cmd(host, find_cmd)

        if dry_run:
            print(f"[DRY-RUN] Would fetch: {kernel}.csv")
            csv_files.append(local_csv_path)
            continue

        try:
            result = subprocess.run(fetch_cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                print(f"✗ Failed to fetch {kernel}.csv (may not exist yet)", file=sys.stderr)
                continue

            # Write CSV locally
            local_csv_path.write_text(sanitize_output(result.stdout))
            csv_files.append(local_csv_path)
            print(f"✓ Synced {kernel}.csv")

        except Exception as e:
            print(f"✗ Failed to sync {kernel}.csv: {e}", file=sys.stderr)

    return csv_files


def expand_benchmark_jobs(selected_kernels: list[str], mandatory_k_values: list[int]) -> list[tuple[str, str, str, str]]:
    """Expand logical kernel selections into msprof output jobs."""
    jobs = []
    for kernel in selected_kernels:
        if kernel == "mandatory_handoff":
            for k_value in mandatory_k_values:
                jobs.append((kernel, f"mandatory_handoff_K{k_value}", f"--k {k_value}", ""))
        elif kernel == "mte_hbm_allcore":
            # All-core benchmark: launch on all 20 AIC cores
            jobs.append((
                kernel,
                kernel,
                "--mte-start 768 --mte-iters 1280 --block-dim 20",
                "--mte-start 0 --mte-iters 768 --block-dim 20",
            ))
        elif kernel in MTE_KERNELS:
            jobs.append((
                kernel,
                kernel,
                "--mte-start 768 --mte-iters 1280",
                "--mte-start 0 --mte-iters 768",
            ))
        else:
            jobs.append((kernel, kernel, "", ""))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"SSH host alias. Default: {DEFAULT_HOST}"
    )
    parser.add_argument(
        "--remote-workdir",
        default=DEFAULT_REMOTE_WORKDIR,
        help=f"Remote working directory. Default: {DEFAULT_REMOTE_WORKDIR}"
    )
    parser.add_argument(
        "--local-repo",
        default=os.getcwd(),
        help="Any path inside the local checkout. Default: current working directory."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Local output directory for CSV files. Default: {DEFAULT_OUTPUT_DIR}"
    )
    parser.add_argument(
        "--n-repeat",
        type=int,
        default=45,
        help="Number of benchmark repetitions. Default: 45"
    )
    parser.add_argument(
        "--kernel-timeout-sec",
        type=int,
        default=300,
        help="Per-kernel msprof timeout in seconds. Default: 300"
    )
    parser.add_argument(
        "--cann-env",
        default=DEFAULT_CANN_ENV,
        help=f"Remote CANN set_env.sh path. Default: {DEFAULT_CANN_ENV}"
    )
    parser.add_argument(
        "--launcher",
        default=DEFAULT_LAUNCHER,
        help=f"Remote benchmark launcher. Default: {DEFAULT_LAUNCHER}"
    )
    parser.add_argument(
        "--msprof",
        default="",
        help="Explicit remote msprof binary. Default: resolve from CANN environment."
    )
    parser.add_argument(
        "--cann-package-path",
        default=DEFAULT_CANN_PACKAGE_PATH,
        help=f"Remote CANN package path for ascendc CMake. Default: {DEFAULT_CANN_PACKAGE_PATH}"
    )
    parser.add_argument(
        "--soc-version",
        default=DEFAULT_SOC_VERSION,
        help=f"Remote AscendC CMake SOC_VERSION. Default: {DEFAULT_SOC_VERSION}"
    )
    parser.add_argument(
        "--kernels",
        default=",".join(RUN_KERNELS),
        help="Comma-separated kernel subset to run/sync. Default: all runnable kernels."
    )
    parser.add_argument(
        "--mandatory-k-values",
        default=",".join(str(k) for k in MANDATORY_HANDOFF_K_VALUES),
        help="Comma-separated K values for mandatory_handoff. Default: 128,256,384,512,1024,2048."
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check remote CANN tools and exit."
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Sync and compile kernels, then exit before running msprof."
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Use sources already present in the remote workdir."
    )
    parser.add_argument(
        "--skip-direct-compile",
        action="store_true",
        help="Skip standalone ccec object compilation and rely on the CMake launcher build."
    )
    parser.add_argument(
        "--build-launcher-only",
        action="store_true",
        help="Sync, compile, build the ACL launcher, then exit before running msprof."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output."
    )
    parser.add_argument(
        "--direct-ssh",
        action="store_true",
        help="Disable any ProxyCommand inherited from SSH config."
    )

    args = parser.parse_args()
    global DIRECT_SSH
    DIRECT_SSH = args.direct_ssh

    # Resolve paths
    repo_root = repo_root_from(Path(args.local_repo).resolve())
    output_dir = Path(args.output_dir)
    selected_kernels = [name.strip() for name in args.kernels.split(",") if name.strip()]
    mandatory_k_values = [int(value.strip()) for value in args.mandatory_k_values.split(",") if value.strip()]
    unknown_kernels = sorted(set(selected_kernels) - set(CCE_KERNELS))
    if unknown_kernels:
        raise SystemExit(f"Unknown kernel(s): {', '.join(unknown_kernels)}")

    print(f"Repo root: {repo_root}")
    print(f"Remote host: {args.host}")
    print(f"Remote workdir: {args.remote_workdir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of repetitions: {args.n_repeat}")
    print(f"Selected kernels: {', '.join(selected_kernels)}")
    print()

    tools = preflight_remote(
        args.host,
        cann_env=args.cann_env,
        launcher=args.launcher,
        require_compiler=not args.skip_direct_compile,
        require_msprof=not (args.compile_only or args.build_launcher_only),
        dry_run=args.dry_run,
    )
    if args.preflight_only:
        return 0
    msprof = args.msprof or tools.get("msprof") or "msprof"

    # Step 1: Sync CCE files
    print("=" * 60)
    print("Step 1: Syncing CCE source files to remote...")
    print("=" * 60)
    if args.skip_sync:
        print("Skipped source sync (--skip-sync).")
    else:
        sync_cce_files(repo_root, args.host, args.remote_workdir, dry_run=args.dry_run)
    print()

    # Step 2: Compile kernels
    print("=" * 60)
    print("Step 2: Compiling CCE kernels on remote...")
    print("=" * 60)
    if args.skip_direct_compile:
        compile_results = {kernel: True for kernel in CCE_KERNELS}
        print("Skipped standalone ccec compilation (--skip-direct-compile).")
    else:
        compile_results = compile_cce_kernels(
            args.host,
            args.remote_workdir,
            cann_env=args.cann_env,
            compiler=tools["compiler"],
            dry_run=args.dry_run,
        )
    failed_compiles = [k for k, ok in compile_results.items() if not ok]
    if failed_compiles and not args.dry_run:
        print(f"\n✗ {len(failed_compiles)} kernels failed to compile:", file=sys.stderr)
        for kernel in failed_compiles:
            print(f"  - {kernel}", file=sys.stderr)
        return 1
    print()

    if args.compile_only:
        return 0

    # Step 3: Build host launcher
    print("=" * 60)
    print("Step 3: Building AscendC ACL benchmark launcher...")
    print("=" * 60)
    launcher_ok = build_bench_launcher(
        args.host,
        args.remote_workdir,
        cann_env=args.cann_env,
        cann_package_path=args.cann_package_path,
        soc_version=args.soc_version,
        dry_run=args.dry_run,
    )
    if not launcher_ok and not args.dry_run:
        return 1
    print()

    if args.build_launcher_only:
        return 0

    # Step 4: Run benchmarks
    print("=" * 60)
    print("Step 4: Running CCE benchmarks with msprof profiling...")
    print("=" * 60)
    benchmark_results = {}
    benchmark_jobs = expand_benchmark_jobs(selected_kernels, mandatory_k_values)
    for kernel, output_name, extra_args, warmup_extra_args in benchmark_jobs:
        if not args.dry_run and not compile_results.get(kernel, True):
            print(f"Skipping {output_name} (compile failed)")
            benchmark_results[output_name] = False
            continue

        print(f"Running: {output_name}...")
        success = run_cce_benchmark(
            args.host,
            args.remote_workdir,
            kernel,
            n_repeat=args.n_repeat,
            output_name=output_name,
            launcher_extra_args=extra_args,
            warmup_extra_args=warmup_extra_args,
            cann_env=args.cann_env,
            cann_package_path=args.cann_package_path,
            kernel_timeout_sec=args.kernel_timeout_sec,
            launcher=args.launcher,
            msprof=msprof,
            dry_run=args.dry_run
        )
        benchmark_results[output_name] = success

    failed_runs = [k for k, ok in benchmark_results.items() if not ok]
    if failed_runs and not args.dry_run:
        print(f"\n⚠ {len(failed_runs)} kernels failed to run:", file=sys.stderr)
        for kernel in failed_runs:
            print(f"  - {kernel}", file=sys.stderr)
    print()

    # Step 5: Sync CSV outputs
    print("=" * 60)
    print("Step 5: Syncing msprof CSV outputs back to local...")
    print("=" * 60)
    csv_files = sync_csv_outputs(
        args.host,
        repo_root,
        args.remote_workdir,
        output_dir,
        kernels=[output_name for _, output_name, _, _ in benchmark_jobs],
        dry_run=args.dry_run
    )
    print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Kernels compiled: {sum(compile_results.values())}/{len(CCE_KERNELS)}")
    print(f"Benchmarks run: {sum(benchmark_results.values())}/{len(benchmark_jobs)}")
    print(f"CSV files synced: {len(csv_files)}")
    print(f"Output directory: {output_dir}")

    if args.dry_run:
        print("\n[DRY-RUN] No actual execution performed. Remove --dry-run to run.")

    if failed_runs and not args.dry_run:
        return 1
    if len(csv_files) != len(benchmark_jobs) and not args.dry_run:
        return 1

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            sys.stdout.write(exc.stdout)
        if exc.stderr:
            sys.stderr.write(exc.stderr)
        raise SystemExit(exc.returncode)
