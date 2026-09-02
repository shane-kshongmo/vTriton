import json

from perfbound.calibration.constants import CalibrationDB, DType, VecOpType
from perfbound.combine import run_report
from perfbound.combine.two_limit import _build_idealized_extract
from perfbound.extract.hivm_extractor import HIVMExtract, HandoffRecord, OpRecord
from perfbound.extract.op_classifier import Component, Precision


class _FakeReport:
    def to_text(self):
        return "ok"

    def to_json(self, path):
        raise AssertionError("unexpected JSON write")


def test_cli_leaves_core_count_auto_by_default(monkeypatch, capsys):
    captured = {}

    def fake_report_from_desgraph(**kwargs):
        captured.update(kwargs)
        return _FakeReport()

    monkeypatch.setattr(run_report, "load_calibration", lambda path: CalibrationDB())
    monkeypatch.setattr(run_report, "report_from_desgraph", fake_report_from_desgraph)
    monkeypatch.setattr(
        "sys.argv",
        ["run_report", "--desgraph", "des.json", "--grid", "7"],
    )

    run_report._cli()

    assert captured["n_cores"] is None
    assert captured["hardware_config"] is None
    assert captured["measured_metric"] == "msprof_task_duration"
    assert "ok" in capsys.readouterr().out


def test_cli_passes_event_elapsed_metric_without_relabeling(monkeypatch, capsys):
    captured = {}

    monkeypatch.setattr(run_report, "load_calibration", lambda path: CalibrationDB())
    monkeypatch.setattr(
        run_report,
        "report_from_desgraph",
        lambda **kwargs: captured.update(kwargs) or _FakeReport(),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_report", "--desgraph", "des.json", "--grid", "1",
            "--measured-us", "12.5", "--measured-metric", "event_elapsed",
        ],
    )

    run_report._cli()

    assert captured["t_measured_us"] == 12.5
    assert captured["measured_metric"] == "event_elapsed"
    assert "ok" in capsys.readouterr().out


def test_cli_forwards_semantic_ir_and_bindings_for_npuir(monkeypatch, capsys):
    captured = {}

    monkeypatch.setattr(run_report, "load_calibration", lambda path: CalibrationDB())
    monkeypatch.setattr(
        run_report,
        "report_from_npuir",
        lambda **kwargs: captured.update(kwargs) or _FakeReport(),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_report", "--npuir", "kernel.npuir.mlir", "--grid", "1",
            "--semantic-ir", "kernel.ttadapter.mlir",
            "--arg-bindings", "arg0=128",
        ],
    )

    run_report._cli()

    assert captured["semantic_ir_path"] == "kernel.ttadapter.mlir"
    assert captured["arg_bindings"] == "arg0=128"
    assert "ok" in capsys.readouterr().out


def test_hardware_config_core_counts_override_calibration_core_topology(tmp_path):
    hw_config = tmp_path / "hw.json"
    hw_config.write_text(
        json.dumps(
            {
                "calibration": {
                    "parallelism": {
                        "num_aic_cores": 13,
                        "num_aiv_cores": 29,
                    },
                    "startup_latencies": {"vector_startup_cycles": 41},
                    "vector_op_cycles_per_vec_instruction": {"exp": 73},
                }
            }
        )
    )
    calib_db = CalibrationDB()

    configured = run_report._calib_with_hardware_core_config(calib_db, hw_config)

    assert configured.core.aic_core_num == 13
    assert configured.core.aiv_core_num == 29
    assert configured.startup_latency["vector"] == 41
    assert configured.vector.op_cycles[(VecOpType.EXP, DType.FP16)] == 73
    assert calib_db.core.aic_core_num == 20
    assert calib_db.core.aiv_core_num == 40


def test_hardware_config_vector_cycles_support_dtype_overrides(tmp_path):
    hw_config = tmp_path / "hw.json"
    hw_config.write_text(
        json.dumps(
            {
                "calibration": {
                    "vector_op_cycles_per_vec_instruction": {"exp": 21},
                    "vector_op_cycles_per_vec_instruction_by_dtype": {
                        "exp": {"fp32": 17}
                    },
                }
            }
        )
    )

    configured = run_report._calib_with_hardware_core_config(
        CalibrationDB(), hw_config
    )

    assert configured.vector.op_cycles[(VecOpType.EXP, DType.FP16)] == 21
    assert configured.vector.op_cycles[(VecOpType.EXP, DType.FP32)] == 17


def test_idealized_handoffs_use_reassigned_components():
    misplaced_matmul = OpRecord(
        op_id=1,
        op_name="matmul",
        component=Component.VECTOR,
        precision=Precision.FP16,
        pipe="Vector",
        flops=1024,
    )
    vector_consumer = OpRecord(
        op_id=2,
        op_name="add",
        component=Component.VECTOR,
        precision=Precision.FP16,
        pipe="Vector",
        elements=128,
        depends_on=[1],
    )
    extract = HIVMExtract(
        operations=[misplaced_matmul, vector_consumer],
        handoffs=[
            HandoffRecord(
                producer_op_id=1,
                consumer_op_id=2,
                producer_component=Component.VECTOR,
                consumer_component=Component.VECTOR,
                bytes_transferred=256,
            )
        ],
    )

    ideal = _build_idealized_extract(extract)

    assert ideal.operations[0].component == Component.CUBE
    assert len(ideal.handoffs) == 1
    assert ideal.handoffs[0].producer_component == Component.CUBE
    assert ideal.handoffs[0].consumer_component == Component.VECTOR
