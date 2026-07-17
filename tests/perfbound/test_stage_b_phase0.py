"""Phase-0 foundation tests for the Stage-B experiment campaign."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from perfbound.combine.run_report import report_from_npuir
from perfbound.experiments.artifacts import (
    ExperimentResult,
    load_experiment_result,
    write_experiment_result,
)
from perfbound.experiments.registry import (
    KernelGroup,
    get_kernel,
    iter_kernel_specs,
    load_user_registry,
)
from scripts.make_figures import collect_results, metric_summary_rows, write_metric_summary
from scripts.profile_suite import (
    Card,
    build_jobs,
    execute_assignments,
    schedule_round_robin,
    shape_env,
    write_schedule,
)
from scripts.run_bound import (
    build_plan,
    build_tritonsim_command,
    dumped_npuir,
    newest_npuir,
    parse_grid,
    run_bound_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_builtin_registry_adapts_existing_benches():
    seeded = get_kernel("seeded_serial")
    assert seeded.group == KernelGroup.GAP_SEEDED
    assert seeded.optimized is False
    assert seeded.path.exists()

    module = seeded.load_module()
    assert hasattr(module, "build_inputs")
    assert hasattr(module, "Model")
    assert callable(seeded.reference())

    gap_kernels = iter_kernel_specs(groups=("V",))
    assert {"seeded_gap1", "seeded_gap2", "seeded_serial"}.issubset(
        {k.name for k in gap_kernels}
    )


def test_user_registry_manifest_ingests_external_kernel(tmp_path):
    kernel = tmp_path / "user_kernel.py"
    kernel.write_text(
        "\n".join(
            [
                "def build_inputs():",
                "    return {'x': 1}",
                "class Model:",
                "    def forward(self, data):",
                "        return data['x']",
                "def reference(x):",
                "    return x",
            ]
        )
    )
    manifest = tmp_path / "kernels.json"
    manifest.write_text(
        json.dumps(
            {
                "kernels": [
                    {
                        "name": "user_fused",
                        "path": "user_kernel.py",
                        "group": "III",
                        "optimized": True,
                        "shapes": [{"id": "s0", "M": 128}],
                    }
                ]
            }
        )
    )

    spec = load_user_registry(manifest)[0]
    spec.validate_interface()
    assert spec.group == KernelGroup.MIXED_CUBE_VECTOR
    assert spec.build_inputs() == {"x": 1}
    assert spec.model_factory().forward({"x": 7}) == 7
    assert spec.shapes[0]["M"] == 128


def test_kernel_spec_loads_module_once_per_process(tmp_path):
    counter = tmp_path / "imports.txt"
    kernel = tmp_path / "side_effect_kernel.py"
    kernel.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                f"p = Path({str(counter)!r})",
                "p.write_text(str(int(p.read_text() or '0') + 1) if p.exists() else '1')",
                "def build_inputs():",
                "    return {'x': 1}",
                "class Model:",
                "    pass",
                "def reference(x):",
                "    return x",
            ]
        )
    )
    manifest = tmp_path / "kernels.json"
    manifest.write_text(
        json.dumps(
            {
                "kernels": [
                    {
                        "name": "side_effect",
                        "path": "side_effect_kernel.py",
                        "group": "III",
                    }
                ]
            }
        )
    )

    spec = load_user_registry(manifest)[0]
    spec.validate_interface()
    spec.build_inputs()
    spec.model_factory()
    spec.reference()

    assert counter.read_text() == "1"


def test_bound_pipeline_plan_and_dry_run_are_ci_safe(tmp_path):
    plan = build_plan(
        kernel_name="seeded_serial",
        kernel_script="test/seeded_serial_bench.py",
        output_dir=tmp_path / "bounds",
        grid_dims=parse_grid("128,32"),
        raw_npuir=tmp_path / "kernel.npuir.mlir",
    )

    dry = run_bound_pipeline(plan, tritonsim_hivm="/bin/tritonsim-hivm", dry_run=True)
    assert dry["dry_run"] is True
    assert dry["plan"]["grid_dims"] == [128, 32]
    assert dry["dump_env"]["TRITON_KERNEL_DUMP"] == "1"
    assert dry["tritonsim_command"] == build_tritonsim_command(
        tritonsim_hivm="/bin/tritonsim-hivm",
        npuir_path=plan.clean_npuir,
        des_json=plan.des_json,
        python_path=sys.executable,
    )


def test_bound_pipeline_real_path_calls_report_with_bindable_kwargs(tmp_path):
    """The non-dry-run path must call report_from_npuir with kwargs it accepts.

    Regression: the pipeline passed ``des_json_path=``, which report_from_npuir
    did not accept, so every real invocation died with TypeError while the
    dry-run-only coverage above stayed green.
    """
    raw = tmp_path / "kernel.npuir.mlir"
    raw.write_text("module {}")
    plan = build_plan(
        kernel_name="seeded_serial",
        kernel_script="test/seeded_serial_bench.py",
        output_dir=tmp_path / "bounds",
        grid_dims=parse_grid("4096"),
        raw_npuir=raw,
    )

    captured: dict = {}

    def fake_report(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(to_json=lambda path: Path(path).write_text("{}"))

    with patch("scripts.run_bound.subprocess.run"), patch(
        "scripts.run_bound.report_from_npuir", fake_report
    ):
        run_bound_pipeline(plan, tritonsim_hivm="/bin/tritonsim-hivm")

    # Bind against the real signature, not the stub: this is what caught the bug.
    inspect.signature(report_from_npuir).bind(**captured)
    assert captured["des_json_path"] == plan.des_json


def test_newest_npuir_selects_latest_file(tmp_path):
    older = tmp_path / "a.npuir.mlir"
    newer = tmp_path / "nested" / "b.npuir.mlir"
    newer.parent.mkdir()
    older.write_text("older")
    newer.write_text("newer")
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (1_000_010, 1_000_010))

    assert newest_npuir((tmp_path,)).read_text() == "newer"


def test_dumped_npuir_prefers_run_local_dump_over_cache(tmp_path):
    dump = tmp_path / "dump"
    cache = tmp_path / "cache"
    dump.mkdir()
    cache.mkdir()
    local = dump / "kernel.npuir.mlir"
    cached = cache / "other.npuir.mlir"
    local.write_text("local")
    cached.write_text("cache")
    os.utime(local, (1_000_000, 1_000_000))
    os.utime(cached, (1_000_010, 1_000_010))

    assert dumped_npuir(dump, cache_dir=cache).read_text() == "local"


def test_stage_b_scripts_support_documented_invocation():
    scripts = (
        "scripts/run_bound.py",
        "scripts/profile_suite.py",
        "scripts/make_figures.py",
        "scripts/component_attribution_prototype.py",
        "scripts/overlap_walker_prototype.py",
    )
    for script in scripts:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=PROJECT_ROOT,
            env={},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_profile_suite_round_robin_schedule_and_artifact(tmp_path):
    specs = iter_kernel_specs(names=("vector_add", "seeded_serial"))
    jobs = build_jobs(specs)
    assignments = schedule_round_robin(
        jobs,
        (Card("card0"), Card("card1")),
        tmp_path,
    )

    assert len(assignments) == 3  # vector_add default + seeded_serial default/small
    assert [a.card.host for a in assignments] == ["card0", "card1", "card0"]
    assert assignments[0].output_csv.name.endswith("_op_summary.csv")
    assert shape_env(assignments[-1].job.shape) == {
        "SEED_N_A": "262144",
        "SEED_N_B": "8192",
        "SEED_NITER": "64",
    }

    schedule_path = write_schedule(assignments, output_root=tmp_path)
    data = load_experiment_result(schedule_path)
    assert data["experiment"] == "profile_suite_schedule"
    assert data["metrics"]["job_count"] == 3
    assert data["metrics"]["card_count"] == 2


def test_profile_suite_execution_keeps_partial_failures(tmp_path):
    specs = iter_kernel_specs(names=("vector_add", "seeded_gap1"))
    assignments = schedule_round_robin(build_jobs(specs), (Card("card0"),), tmp_path)

    def fake_remote_bench(**kwargs):
        if kwargs["kernel_name"] == "seeded_gap1":
            raise RuntimeError("remote failed")
        return kwargs["output_csv"], kwargs["output_npy"]

    with patch("scripts.profile_suite.run_remote_bench", side_effect=fake_remote_bench):
        rows = execute_assignments(assignments, max_workers=2)

    assert len(rows) == len(assignments)
    by_kernel = {row["kernel_name"]: row for row in rows}
    assert by_kernel["vector_add"]["status"] == "complete"
    assert by_kernel["seeded_gap1"]["status"] == "failed"
    assert "RuntimeError: remote failed" in by_kernel["seeded_gap1"]["error"]


def test_experiment_artifact_and_figure_summary(tmp_path):
    result = ExperimentResult(
        experiment="exp1_soundness",
        rows=({"kernel": "vector_add", "sound": True},),
        metrics={"soundness_rate": 1.0, "kernel_count": 1},
        provenance={"source": "fixture"},
    )
    path = write_experiment_result(result, root=tmp_path)
    loaded = collect_results(tmp_path)
    assert loaded[0]["_source"] == str(path)

    rows = metric_summary_rows(loaded)
    assert {row["metric"] for row in rows} == {"kernel_count", "soundness_rate"}

    csv_path = write_metric_summary(loaded, tmp_path / "figures" / "metrics.csv")
    text = csv_path.read_text()
    assert "exp1_soundness,soundness_rate,1.0,complete" in text
