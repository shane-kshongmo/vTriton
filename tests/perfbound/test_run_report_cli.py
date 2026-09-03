import json

import pytest

from perfbound.calibration.constants import CalibrationDB
from perfbound.combine import run_report
from perfbound.combine.report import KernelReport


class _FakeReport:
    def to_text(self):
        return "ok"

    def to_dict(self):
        return KernelReport("kernel", 1.0, "component").to_dict()

    def to_json(self, path):
        raise AssertionError("unexpected JSON write")


def _run_cli(monkeypatch, argv, report_factory):
    monkeypatch.setattr(run_report, "load_calibration", lambda path: CalibrationDB())
    monkeypatch.setattr(run_report, report_factory.__name__, report_factory)
    monkeypatch.setattr("sys.argv", ["run_report", *argv])
    run_report._cli()


def test_cli_leaves_core_count_auto_by_default(monkeypatch, capsys):
    captured = {}

    def report_from_desgraph(**kwargs):
        captured.update(kwargs)
        return _FakeReport()

    _run_cli(
        monkeypatch,
        ["--desgraph", "des.json", "--grid", "7"],
        report_from_desgraph,
    )

    assert captured["n_cores"] is None
    assert captured["hardware_config"] is None
    assert captured["measured_metric"] == "msprof_task_duration"
    assert "ok" in capsys.readouterr().out


def test_cli_passes_event_elapsed_metric(monkeypatch, capsys):
    captured = {}

    def report_from_desgraph(**kwargs):
        captured.update(kwargs)
        return _FakeReport()

    _run_cli(
        monkeypatch,
        [
            "--desgraph", "des.json", "--grid", "1",
            "--measured-us", "12.5", "--measured-metric", "event_elapsed",
        ],
        report_from_desgraph,
    )

    assert captured["t_measured_us"] == 12.5
    assert captured["measured_metric"] == "event_elapsed"
    assert "ok" in capsys.readouterr().out


def test_cli_forwards_semantic_ir_and_bindings(monkeypatch, capsys):
    captured = {}

    def report_from_npuir(**kwargs):
        captured.update(kwargs)
        return _FakeReport()

    _run_cli(
        monkeypatch,
        [
            "--npuir", "kernel.npuir.mlir", "--grid", "1",
            "--semantic-ir", "kernel.ttadapter.mlir",
            "--arg-bindings", "arg0=128",
        ],
        report_from_npuir,
    )

    assert captured["semantic_ir_path"] == "kernel.ttadapter.mlir"
    assert captured["arg_bindings"] == "arg0=128"
    assert "ok" in capsys.readouterr().out


def test_cli_can_emit_legacy_aliases_for_migration(monkeypatch, tmp_path):
    output = tmp_path / "legacy.json"

    def report_from_desgraph(**kwargs):
        return _FakeReport()

    _run_cli(
        monkeypatch,
        [
            "--desgraph", "des.json", "--grid", "1",
            "--output-json", str(output), "--legacy-report-aliases",
        ],
        report_from_desgraph,
    )

    data = json.loads(output.read_text())
    assert data["compatibility"]["legacy_report_aliases"] == "v1"
    assert "reachability" in data
    assert "profile" in data
    assert "headroom_assessment" in data


def test_cli_rejects_legacy_aliases_without_json_output(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_report", "--desgraph", "des.json", "--grid", "1",
            "--legacy-report-aliases",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        run_report._cli()
