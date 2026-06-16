# Tests for remote_bench.py (A.6.2)
#
# CI-runnable tests using string/command inspection — no actual SSH needed.
# Verifies: binary naming, SSH command structure, Triton kernel launcher
# invocation, environment contract conformance, preflight checks, and
# host configuration resolution.
#
# Source spec: .omc/plans/a6_2_blockers_scope.md Task 9

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.remote_bench import (
    _remote_preamble,
    _remote_env_preamble,
    resolve_remote_config,
    run_msprof_remote,
    run_python_kernel_remote,
    recompile_remote,
    run_remote_bench,
)


# ── Helpers ─────────────────────────────────────────────────────────

def _make_subprocess_result(returncode=0, stdout="", stderr=""):
    """Build a mock CompletedProcess that won't trip check=True."""
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    result.check_returncode = MagicMock(return_value=None)
    return result


def _all_ssh_commands(mock_run):
    """Extract all ssh command lists from mock calls."""
    cmds = []
    for call in mock_run.call_args_list:
        args = call[0]
        if args and isinstance(args[0], list) and args[0][0] == "ssh":
            cmds.append(args[0])
    return cmds


# ══════════════════════════════════════════════════════════════════════
# Test: Binary naming
# ══════════════════════════════════════════════════════════════════════


class TestBinaryNaming:
    """Verify bishengir-compile (not bishengir-compile-a5) is used."""

    def test_recompile_uses_bishengir_compile_not_a5(self):
        """recompile_remote references 'bishengir-compile', not '-a5'."""
        # The default bishengir_path should be 'bishengir-compile' (no -a5)
        with patch("scripts.remote_bench.subprocess") as mock_sub:
            mock_sub.run = MagicMock(
                return_value=_make_subprocess_result()
            )
            try:
                recompile_remote(
                    remote_host="user@host",
                    remote_path="~/vTriton",
                    hivm_path=Path("/tmp/edited.npuir.mlir"),
                    kernel_name="test_kernel",
                )
            except Exception:
                pass  # We only care about the commands issued

        # Find the ssh command that runs bishengir
        ssh_cmds = _all_ssh_commands(mock_sub.run)
        compile_cmds = [
            cmd for cmd in ssh_cmds
            if len(cmd) >= 3 and "bishengir" in cmd[2]
        ]
        assert len(compile_cmds) > 0, "No ssh command invoking bishengir found"

        for cmd in compile_cmds:
            script = cmd[2]  # The script passed to ssh
            assert "bishengir-compile" in script, (
                f"Expected 'bishengir-compile' in script, got: {script[:200]}"
            )
            assert "bishengir-compile-a5" not in script, (
                f"Stale '-a5' suffix found in script: {script[:200]}"
            )

    def test_recompile_rejects_des_json(self):
        """DES JSON edits are analytical-only and must not reach bishengir."""
        with patch("scripts.remote_bench.subprocess") as mock_sub:
            mock_sub.run = MagicMock(return_value=_make_subprocess_result())
            with pytest.raises(ValueError, match="not DES JSON"):
                recompile_remote(
                    remote_host="user@host",
                    remote_path="~/vTriton",
                    hivm_path=Path("/tmp/edited.json"),
                    kernel_name="test_kernel",
                )

        mock_sub.run.assert_not_called()


# ══════════════════════════════════════════════════════════════════════
# Test: SSH command structure
# ══════════════════════════════════════════════════════════════════════


class TestSSHCommandStructure:
    """Verify ssh is called as ['ssh', host, script] (3-arg), not
    ['ssh', host, 'bash', '-c', script] (5-arg, which breaks quoting)."""

    def test_ssh_command_is_single_arg_run_msprof(self):
        """run_msprof_remote passes the script as one argument to ssh."""
        with patch("scripts.remote_bench.subprocess") as mock_sub:
            mock_sub.run = MagicMock(
                return_value=_make_subprocess_result()
            )
            try:
                run_msprof_remote("user@host", "~/vTriton", "/path/to/exe")
            except Exception:
                pass

        ssh_cmds = _all_ssh_commands(mock_sub.run)
        for cmd in ssh_cmds:
            assert len(cmd) == 3, (
                f"ssh command should be ['ssh', host, script] (3 args), "
                f"got {len(cmd)} args: {cmd}"
            )
            assert cmd[0] == "ssh"
            assert cmd[1] == "user@host"

    def test_ssh_command_is_single_arg_python_kernel(self):
        """run_python_kernel_remote passes the script as one argument."""
        with patch("scripts.remote_bench.subprocess") as mock_sub:
            mock_sub.run = MagicMock(
                return_value=_make_subprocess_result()
            )
            try:
                run_python_kernel_remote(
                    "user@host", "~/vTriton",
                    "test/kernel.py", "msprof_output",
                )
            except Exception:
                pass

        ssh_cmds = _all_ssh_commands(mock_sub.run)
        for cmd in ssh_cmds:
            assert len(cmd) == 3, (
                f"ssh command should be ['ssh', host, script] (3 args), "
                f"got {len(cmd)} args: {cmd}"
            )


# ══════════════════════════════════════════════════════════════════════
# Test: Python kernel launcher invocation
# ══════════════════════════════════════════════════════════════════════


class TestPythonKernelLauncher:
    """Verify msprof wraps 'python launcher.py' for .py scripts."""

    def test_python_kernel_launcher_invocation(self):
        """run_python_kernel_remote invokes msprof with python launcher."""
        with patch("scripts.remote_bench.subprocess") as mock_sub:
            mock_sub.run = MagicMock(
                return_value=_make_subprocess_result()
            )
            try:
                run_python_kernel_remote(
                    "user@host", "~/vTriton",
                    "test/kernel.py", "msprof_output",
                    output_dir="kernel_outputs", iters=5,
                )
            except Exception:
                pass

        ssh_cmds = _all_ssh_commands(mock_sub.run)
        # Find the msprof command (should contain kernel_launcher.py)
        msprof_cmds = [
            cmd for cmd in ssh_cmds
            if len(cmd) >= 3 and "msprof" in cmd[2] and "kernel_launcher" in cmd[2]
        ]
        assert len(msprof_cmds) > 0, (
            "No msprof command wrapping kernel_launcher.py found"
        )

        script = msprof_cmds[0][2]
        assert "python" in script, (
            f"msprof should wrap 'python launcher.py', got: {script[:300]}"
        )
        assert "kernel_launcher.py" in script
        assert "--kernel test/kernel.py" in script
        assert "--output-dir kernel_outputs" in script
        assert "--iters 5" in script

    def test_run_remote_bench_dispatches_python_kernel(self):
        """run_remote_bench detects .py kernel_script → python kernel path."""
        with patch("scripts.remote_bench.subprocess") as mock_sub, \
             patch("scripts.remote_bench.sync_to_remote"), \
             patch("scripts.remote_bench.fetch_csv_from_remote") as mock_csv, \
             patch("scripts.remote_bench.fetch_output_from_remote") as mock_npy:
            mock_sub.run = MagicMock(
                return_value=_make_subprocess_result()
            )
            mock_csv.return_value = Path("/tmp/test.csv")
            mock_npy.return_value = None

            try:
                run_remote_bench(
                    remote_host="user@host",
                    kernel_name="chunk_kda",
                    kernel_script=Path("test/kernel.py"),
                    output_csv=Path("/tmp/test.csv"),
                )
            except Exception:
                pass

        # Should have an ssh command with kernel_launcher (python kernel path)
        ssh_cmds = _all_ssh_commands(mock_sub.run)
        python_cmds = [
            cmd for cmd in ssh_cmds
            if len(cmd) >= 3 and "kernel_launcher" in cmd[2]
        ]
        assert len(python_cmds) > 0, (
            "run_remote_bench should dispatch to python kernel path for .py scripts"
        )


# ══════════════════════════════════════════════════════════════════════
# Test: Contract conformance
# ══════════════════════════════════════════════════════════════════════


class TestContractConformance:
    """Verify environment preamble matches the real 910B3 contract."""

    def test_contract_conformance_env_preamble(self):
        """PYTHONPATH, conda triton_hxl, CANN ascend-toolkit all present."""
        preamble = _remote_env_preamble("~/vTriton")

        # CANN environment — real machine uses the ascend-toolkit set_env.sh
        assert "source /usr/local/Ascend/ascend-toolkit/set_env.sh" in preamble, (
            "Missing CANN ascend-toolkit set_env.sh in preamble"
        )

        # Conda activation — real machine env is triton_hxl
        assert "conda activate triton_hxl" in preamble, (
            "Missing conda activate triton_hxl in preamble"
        )

        # PYTHONPATH for triton-ascend
        assert "PYTHONPATH" in preamble, (
            "Missing PYTHONPATH export in preamble"
        )
        assert "triton-ascend/python" in preamble, (
            "PYTHONPATH should include triton-ascend/python"
        )

    def test_preamble_no_double_brace(self):
        """Generated shell must use single-brace '{ ...; }' groups, not '{{'.

        Regression guard: the old plain-string preamble emitted literal
        '{{ ... }}' which is invalid bash and broke the '||' fallbacks on the
        first real hardware run.
        """
        preamble = _remote_env_preamble("~/vTriton")
        assert "{{" not in preamble, f"literal '{{{{' leaked into shell: {preamble}"
        assert "}}" not in preamble
        # The fail-loud group should be a valid single-brace group.
        assert "|| { echo 'CANN env not found'" in preamble

    def test_preamble_overridable_via_env(self):
        """CANN path + conda env are overridable via env vars."""
        with patch.dict(os.environ, {
            "VTRITON_REMOTE_CANN_SETENV": "/custom/cann/set_env.sh",
            "VTRITON_REMOTE_CONDA_ENV": "myenv",
        }):
            preamble = _remote_env_preamble("~/vTriton")
        assert "source /custom/cann/set_env.sh" in preamble
        assert "conda activate myenv" in preamble

    def test_preamble_fail_loud_on_cann_error(self):
        """Preamble exits with error if CANN env source fails."""
        preamble = _remote_preamble()
        assert "exit 1" in preamble, (
            "Preamble should fail loud on CANN activation errors"
        )
        assert "CANN env not found" in preamble

    def test_preamble_fail_loud_on_conda_error(self):
        """Preamble exits with error if conda activation fails."""
        preamble = _remote_preamble()
        assert "conda not found" in preamble
        assert "conda activate triton_hxl failed" in preamble

    def test_preamble_cd_to_remote_path(self):
        """Preamble starts with cd to remote_path."""
        preamble = _remote_env_preamble("/opt/vTriton")
        assert preamble.startswith("cd /opt/vTriton"), (
            f"Preamble should start with 'cd /opt/vTriton', got: {preamble[:80]}"
        )


# ══════════════════════════════════════════════════════════════════════
# Test: Preflight checks
# ══════════════════════════════════════════════════════════════════════


class TestPreflightChecks:
    """Verify msprof --version guard is present before profiling."""

    def test_preflight_msprof_version_in_elf_path(self):
        """run_msprof_remote checks msprof --version before profiling."""
        with patch("scripts.remote_bench.subprocess") as mock_sub:
            mock_sub.run = MagicMock(
                return_value=_make_subprocess_result()
            )
            try:
                run_msprof_remote("user@host", "~/vTriton", "/path/to/exe")
            except Exception:
                pass

        ssh_cmds = _all_ssh_commands(mock_sub.run)
        msprof_cmds = [
            cmd for cmd in ssh_cmds
            if len(cmd) >= 3 and "msprof" in cmd[2]
        ]
        assert len(msprof_cmds) > 0

        script = msprof_cmds[0][2]
        assert "command -v msprof" in script, (
            f"Missing 'command -v msprof' preflight check in: {script[:300]}"
        )

    def test_preflight_msprof_version_in_python_path(self):
        """run_python_kernel_remote checks msprof --version before profiling."""
        with patch("scripts.remote_bench.subprocess") as mock_sub:
            mock_sub.run = MagicMock(
                return_value=_make_subprocess_result()
            )
            try:
                run_python_kernel_remote(
                    "user@host", "~/vTriton",
                    "test/kernel.py", "msprof_output",
                )
            except Exception:
                pass

        ssh_cmds = _all_ssh_commands(mock_sub.run)
        msprof_cmds = [
            cmd for cmd in ssh_cmds
            if len(cmd) >= 3 and "kernel_launcher" in cmd[2]
        ]
        assert len(msprof_cmds) > 0

        script = msprof_cmds[0][2]
        assert "command -v msprof" in script, (
            f"Missing 'command -v msprof' preflight check in: {script[:300]}"
        )


# ══════════════════════════════════════════════════════════════════════
# Test: Host configuration resolution
# ══════════════════════════════════════════════════════════════════════


class TestResolveRemoteConfig:
    """Verify resolve_remote_config follows CLI → env → file hierarchy."""

    def test_cli_args_take_precedence(self):
        """CLI args override env vars and config file."""
        with patch.dict(os.environ, {
            "VTRITON_REMOTE_HOST": "env@host",
            "VTRITON_REMOTE_PATH": "/env/path",
        }):
            host, path = resolve_remote_config("cli@host", "/cli/path")
        assert host == "cli@host"
        assert path == "/cli/path"

    def test_env_var_fallback(self):
        """Environment variables used when CLI args are None."""
        with patch.dict(os.environ, {
            "VTRITON_REMOTE_HOST": "env@host",
            "VTRITON_REMOTE_PATH": "/env/path",
        }):
            host, path = resolve_remote_config(None, None)
        assert host == "env@host"
        assert path == "/env/path"

    def test_env_var_host_only(self):
        """VTRITON_REMOTE_HOST used, default path when no env path."""
        with patch.dict(os.environ, {"VTRITON_REMOTE_HOST": "env@host"}, clear=False):
            # Remove VTRITON_REMOTE_PATH if present
            env = os.environ.copy()
            env.pop("VTRITON_REMOTE_PATH", None)
            with patch.dict(os.environ, env, clear=True):
                os.environ["VTRITON_REMOTE_HOST"] = "env@host"
                host, path = resolve_remote_config(None, None)
        assert host == "env@host"
        assert path == "~/vTriton"  # default

    def test_config_file_fallback(self, tmp_path):
        """~/.vtriton_remote config file used as last resort."""
        config_file = tmp_path / ".vtriton_remote"
        config_file.write_text("[remote]\nhost = file@host\npath = /file/path\n")

        with patch("scripts.remote_bench.Path.home", return_value=tmp_path), \
             patch.dict(os.environ, {}, clear=False):
            # Clear env vars
            env = os.environ.copy()
            env.pop("VTRITON_REMOTE_HOST", None)
            env.pop("VTRITON_REMOTE_PATH", None)
            with patch.dict(os.environ, env, clear=True):
                host, path = resolve_remote_config(None, None)
        assert host == "file@host"
        assert path == "/file/path"

    def test_no_config_returns_none_host(self):
        """No config anywhere → host is None."""
        with patch("scripts.remote_bench.Path.home", return_value=Path("/nonexistent")), \
             patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("VTRITON_REMOTE_HOST", None)
            env.pop("VTRITON_REMOTE_PATH", None)
            with patch.dict(os.environ, env, clear=True):
                host, path = resolve_remote_config(None, None)
        assert host is None
        assert path == "~/vTriton"  # default

    def test_cli_remote_path_overrides_env_path(self):
        """CLI --remote-path overrides VTRITON_REMOTE_PATH."""
        with patch.dict(os.environ, {
            "VTRITON_REMOTE_HOST": "env@host",
            "VTRITON_REMOTE_PATH": "/env/path",
        }):
            host, path = resolve_remote_config(None, "/cli/path")
        assert host == "env@host"  # from env (no CLI host)
        assert path == "/cli/path"  # CLI overrides env


# ══════════════════════════════════════════════════════════════════════
# Test: CSV fallback (rglob)
# ══════════════════════════════════════════════════════════════════════


class TestCSVFallback:
    """Verify rglob fallback for locating op_summary CSVs."""

    def test_csv_find_includes_rglob_fallback(self):
        """fetch_csv_from_remote uses both find and rglob as fallback."""
        import inspect
        from scripts.remote_bench import fetch_csv_from_remote
        source = inspect.getsource(fetch_csv_from_remote)

        assert "find" in source, "Should use 'find' for CSV location"
        assert "rglob" in source, "Should have rglob fallback for CSV location"
        assert "op_summary" in source, "Should search for op_summary CSVs"
