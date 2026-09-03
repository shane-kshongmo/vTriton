import json

from perfbound.calibration.constants import CalibrationDB, DType, VecOpType
from perfbound.combine.run_report import _calib_with_hardware_core_config


def test_hardware_config_overlays_topology_and_shared_vector_costs(tmp_path):
    hardware = tmp_path / "hardware.json"
    hardware.write_text(json.dumps({
        "calibration": {
            "parallelism": {"num_aic_cores": 13, "num_aiv_cores": 29},
            "startup_latencies": {"vector_startup_cycles": 41},
            "vector_op_cycles_per_vec_instruction": {"exp": 73},
        }
    }))
    original = CalibrationDB()

    configured = _calib_with_hardware_core_config(original, hardware)

    assert configured.core.aic_core_num == 13
    assert configured.core.aiv_core_num == 29
    assert configured.startup_latency["vector"] == 41
    assert configured.vector.op_cycles[(VecOpType.EXP, DType.FP16)] == 73
    assert original.core.aic_core_num == 20
    assert original.core.aiv_core_num == 40


def test_hardware_config_dtype_cost_overrides_shared_cost(tmp_path):
    hardware = tmp_path / "hardware.json"
    hardware.write_text(json.dumps({
        "calibration": {
            "vector_op_cycles_per_vec_instruction": {"exp": 21},
            "vector_op_cycles_per_vec_instruction_by_dtype": {
                "exp": {"fp32": 17}
            },
        }
    }))

    configured = _calib_with_hardware_core_config(CalibrationDB(), hardware)

    assert configured.vector.op_cycles[(VecOpType.EXP, DType.FP16)] == 21
    assert configured.vector.op_cycles[(VecOpType.EXP, DType.FP32)] == 17
