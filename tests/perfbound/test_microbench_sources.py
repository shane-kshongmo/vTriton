from pathlib import Path


MICROBENCH_DIR = Path(__file__).parents[2] / "perfbound/calibration/microbench"


def test_microbench_sources_are_current_ascendc_entrypoints():
    sources = sorted(MICROBENCH_DIR.glob("*.cce"))

    # 16 prior kernels plus isolated FP16 and FP32 transcendental measurements.
    assert len(sources) == 24
    for source in sources:
        text = source.read_text()
        assert "TODO" not in text
        assert "hb_compute" not in text
        assert "kernel_operator.h" in text or "vt_microbench_common.h" in text
        assert 'extern "C" __global__ __aicore__ void' in text
