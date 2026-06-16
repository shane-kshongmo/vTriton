from pathlib import Path


MICROBENCH_DIR = Path(__file__).parents[2] / "perfbound/calibration/microbench"


def test_microbench_sources_are_current_ascendc_entrypoints():
    sources = sorted(MICROBENCH_DIR.glob("*.cce"))

    # 13 original + scalar_peak + mte_l0c_to_gm + mte_hbm_allcore
    assert len(sources) == 16
    for source in sources:
        text = source.read_text()
        assert "TODO" not in text
        assert "hb_compute" not in text
        assert "kernel_operator.h" in text or "vt_microbench_common.h" in text
        assert 'extern "C" __global__ __aicore__ void' in text
