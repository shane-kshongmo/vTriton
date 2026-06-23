"""US-SB-006 — absolute Gap-2 gate, held-out validation on 910B3.

Closes the *absolute* (not just mechanism) Gap-2 counterfactual gate.  The
``stream_copy`` microbench (``test/stream_copy_bench.py``) is bandwidth-bound, so
its large-tile form hits the true HBM peak and its small-tile form exposes the
pure packet-amortization penalty (unlike the overhead-bound ``amort`` kernel,
whose grid=2048 tiny programs are launch-limited and only reach 325 GB/s even
when coalesced — it cannot anchor an *absolute* bandwidth bound).

We hold out interior packet sizes from the calibrated ``pkt_efficiency`` curve,
predict them with the shipped log-log interpolator, and require the model to
reproduce, within 20 %:

  1. the measured small-packet bandwidth penalty η at the held-out packets;
  2. the absolute achieved bandwidth (⇒ absolute MTE time) at those packets;
  3. the coalesced-vs-seed counterfactual *delta* — the Gap-2 quantity the PRD
     gate (`quantification_error < 0.20`) is defined on.

Because the held-out packets are removed before interpolation, (1)–(3) are
non-circular: the calibration never saw them.

Evidence: tests/perfbound/fixtures/stream_copy_eta_sweep_910b3.json
(device-timed, warmup=8, iters=40, median; block=524288, grid=48, 40 AIV cores
saturated).  Provenance in .omc/research/hw_runs/hbm_calib/ and
.omc/plans/peak_hbm_contention_calibration.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from perfbound.calibration.calib_loader import load_default_calib_db
from perfbound.calibration.constants import MemHierarchy

_FIXTURE = Path(__file__).parent / "fixtures" / "stream_copy_eta_sweep_910b3.json"

# Interior packets withheld from the curve before interpolation (non-circular).
HELD_OUT_PKTS = (1024, 16384)
THRESHOLD = 0.20
# Coalesced anchor: the largest measured packet runs at the HBM peak (η ≡ 1).
COALESCED_PKT = 65536


@pytest.fixture(scope="module")
def sweep() -> dict:
    if not _FIXTURE.exists():
        pytest.skip(f"missing stream_copy sweep evidence: {_FIXTURE}")
    return json.loads(_FIXTURE.read_text())


@pytest.fixture(scope="module")
def by_pkt(sweep) -> dict[int, dict]:
    return {int(r["packet_bytes"]): r for r in sweep["runs"]}


def _held_out_memory() -> MemHierarchy:
    """A MemHierarchy whose pkt_efficiency curve omits the held-out packets."""
    base = load_default_calib_db().memory
    return MemHierarchy(
        hbm_peak_aggregate_bw=base.hbm_peak_aggregate_bw,
        pkt_efficiency={
            k: v for k, v in base.pkt_efficiency.items() if k not in HELD_OUT_PKTS
        },
    )


def test_calibration_uses_streamcopy_curve():
    """The shipped curve must be the true-peak (bandwidth-bound) calibration."""
    mem = load_default_calib_db().memory
    assert mem.hbm_peak_aggregate_bw > 1_000_000.0  # > 1000 GB/s in B/us
    # Both held-out interior points are present in the shipped curve (so the
    # held-out test below actually removes something).
    for pkt in HELD_OUT_PKTS:
        assert pkt in mem.pkt_efficiency


@pytest.mark.parametrize("pkt", HELD_OUT_PKTS)
def test_held_out_packet_efficiency(by_pkt, pkt):
    """η predicted by interpolation matches measured η within 20 %."""
    mem = _held_out_memory()
    peak_gbps = max(r["agg_gbps"] for r in by_pkt.values())
    eta_meas = by_pkt[pkt]["agg_gbps"] / peak_gbps
    eta_pred = mem._packet_efficiency(pkt)
    err = abs(eta_pred - eta_meas) / eta_meas
    assert err < THRESHOLD, (
        f"pkt={pkt}B: η_pred={eta_pred:.4f} vs η_meas={eta_meas:.4f} "
        f"(err {err:.1%} ≥ {THRESHOLD:.0%})"
    )


@pytest.mark.parametrize("pkt", HELD_OUT_PKTS)
def test_held_out_absolute_bandwidth(by_pkt, pkt):
    """Predicted absolute achieved bandwidth (⇒ MTE time) within 20 %."""
    mem = _held_out_memory()
    peak_gbps = max(r["agg_gbps"] for r in by_pkt.values())
    agg_meas = by_pkt[pkt]["agg_gbps"]
    agg_pred = peak_gbps * mem._packet_efficiency(pkt)
    err = abs(agg_pred - agg_meas) / agg_meas
    assert err < THRESHOLD, (
        f"pkt={pkt}B: agg_pred={agg_pred:.1f} vs agg_meas={agg_meas:.1f} GB/s "
        f"(err {err:.1%} ≥ {THRESHOLD:.0%})"
    )


def test_gap2_counterfactual_delta_absolute(by_pkt):
    """The PRD gate: coalesced→seed bound delta vs measured delta < 20 %.

    Seed = a *held-out* small packet (1024 B); coalesced = the large-packet
    anchor (65536 B, η≡1).  Predicted times use total bytes / (peak·η); measured
    times are the device medians.  This is the bandwidth-bound Gap-2
    counterfactual the absolute gate is defined on.
    """
    mem = _held_out_memory()
    peak_bus = mem.hbm_peak_aggregate_bw  # B/us
    seed_pkt = 1024  # held out

    seed, coalesced = by_pkt[seed_pkt], by_pkt[COALESCED_PKT]
    total_bytes = seed["bytes_moved"]
    assert coalesced["bytes_moved"] == total_bytes  # same volume, only packet differs

    eta_seed = mem._packet_efficiency(seed_pkt)
    eta_coalesced = mem._packet_efficiency(COALESCED_PKT)  # ≈ 1.0
    t_pred_seed = total_bytes / (peak_bus * eta_seed)
    t_pred_coalesced = total_bytes / (peak_bus * eta_coalesced)
    bound_delta = t_pred_seed - t_pred_coalesced

    measured_delta = seed["median_us"] - coalesced["median_us"]
    err = abs(bound_delta - measured_delta) / measured_delta
    assert err < THRESHOLD, (
        f"Gap-2 delta: bound={bound_delta:.1f}us vs measured={measured_delta:.1f}us "
        f"(quantification_error {err:.1%} ≥ {THRESHOLD:.0%})"
    )
