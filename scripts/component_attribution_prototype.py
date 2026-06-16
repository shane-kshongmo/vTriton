#!/usr/bin/env python3
"""Author-headroom component attribution (prototype).

Decomposes the T_measured - T_bound author-headroom gap using the two inputs
that are actually available — the msprof op-summary CSV and the HIVM/DES graph
— instead of literature-derived gap estimates.

Two independent views are produced and cross-checked:

  1. msprof MEASURED engine-time split  (where the hardware actually spent time)
       AIV: aiv_scalar / aiv_vec / aiv_mte2 / aiv_mte3
       AIC: aic_mac / aic_scalar / aic_mte1 / aic_mte2 / aic_fixpipe
  2. HIVM STRUCTURAL busy-cycle split   (where the model predicts time goes)
       sum(duration * loop_multiplier) per PIPE, converted to us via clock_ghz

Both should agree on the dominant engine.  For chunk_kda they do: scalar.

Usage:
    python scripts/component_attribution_prototype.py \
        --csv .omc/research/hw_runs/chunk_kda_op_summary.csv \
        --des .omc/research/hw_runs/kda_des.json \
        --kernel chunk_kda_bwd \
        --t-bound 46109.91
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math


def msprof_measured(csv_path: str, kernel_substr: str) -> dict:
    """Return measured engine-time (us) from the kernel's msprof row."""
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if kernel_substr in r.get("Op Name", "")]
    if not rows:
        raise SystemExit(f"no rows matching {kernel_substr!r} in {csv_path}")
    r = rows[0]  # invocations are near-identical; take the first

    def fnum(key: str) -> float:
        try:
            return float(r[key])
        except (TypeError, ValueError):
            return math.nan

    t = fnum("Task Duration(us)")
    aiv = fnum("aiv_time(us)")
    aic = fnum("aicore_time(us)")
    # engine-time = core busy time * that engine's ratio of the core
    eng = {
        "scalar (AIV)": fnum("aiv_scalar_ratio") * aiv,
        "vector (AIV)": fnum("aiv_vec_ratio") * aiv,
        "mte2-load (AIV)": fnum("aiv_mte2_ratio") * aiv,
        "mte3-store (AIV)": fnum("aiv_mte3_ratio") * aiv,
        "mac-cube (AIC)": fnum("aic_mac_ratio") * aic,
        "scalar (AIC)": fnum("aic_scalar_ratio") * aic,
        "mte1 (AIC)": fnum("aic_mte1_ratio") * aic,
        "mte2-load (AIC)": fnum("aic_mte2_ratio") * aic,
        "fixpipe (AIC)": fnum("aic_fixpipe_ratio") * aic,
    }
    return {
        "t_measured_us": t,
        "aiv_time_us": aiv,
        "aic_time_us": aic,
        "engine_us": eng,
        "scalar_total_us": eng["scalar (AIV)"] + eng["scalar (AIC)"],
        "icache_miss": (fnum("aic_icache_miss_rate"), fnum("aiv_icache_miss_rate")),
        "cube_util_pct": fnum("cube_utilization(%)"),
    }


def hivm_structural(des_path: str) -> dict:
    """Return per-PIPE structural busy time (us) from the DES graph."""
    d = json.load(open(des_path))
    ghz = d["clock_ghz"]
    ops = d["operations"]
    cyc = collections.Counter()
    n = collections.Counter()
    for o in ops:
        lm = o.get("loop_multiplier", 1) or 1
        cyc[o["pipe"]] += o["duration"] * lm
        n[o["pipe"]] += 1
    pipe_us = {p: c / (ghz * 1000.0) for p, c in cyc.items()}  # cycles -> us
    return {
        "clock_ghz": ghz,
        "schedule_truncated": d.get("schedule_truncated"),
        "n_ops": len(ops),
        "pipe_us": pipe_us,
        "pipe_n": dict(n),
        "total_us": sum(pipe_us.values()),
    }


def _bar(frac: float, width: int = 28) -> str:
    return "#" * int(round(frac * width))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--des", required=True)
    ap.add_argument("--kernel", default="chunk_kda_bwd")
    ap.add_argument("--t-bound", type=float, default=None,
                    help="T_bound_DSL (us) to report author headroom")
    args = ap.parse_args()

    m = msprof_measured(args.csv, args.kernel)
    h = hivm_structural(args.des)

    print(f"Kernel: {args.kernel}")
    print(f"T_measured = {m['t_measured_us']:.0f} us   "
          f"aiv_time={m['aiv_time_us']:.0f}  aicore_time={m['aic_time_us']:.0f}")
    print()

    print("=== [1] msprof MEASURED engine-time (where the HW spent time) ===")
    eng = m["engine_us"]
    base = max(m["aiv_time_us"], m["aic_time_us"])
    for name, us in sorted(eng.items(), key=lambda kv: -kv[1]):
        frac = us / base if base else 0.0
        print(f"  {name:18s} {us:10.0f} us  {frac*100:5.1f}%  {_bar(frac)}")
    print(f"  -> scalar total (AIV+AIC) = {m['scalar_total_us']:.0f} us "
          f"({m['scalar_total_us']/m['t_measured_us']*100:.1f}% of T_measured)")
    print(f"  -> icache_miss (aic,aiv) = {m['icache_miss']}   "
          f"cube_util = {m['cube_util_pct']:.1f}% (occupied; mac_ratio tiny => stalled)")
    print()

    print("=== [2] HIVM STRUCTURAL busy per PIPE (where the model puts time) ===")
    tot = h["total_us"]
    for p, us in sorted(h["pipe_us"].items(), key=lambda kv: -kv[1]):
        frac = us / tot if tot else 0.0
        print(f"  {p:14s} n={h['pipe_n'][p]:4d} {us:9.1f} us  {frac*100:5.1f}%  {_bar(frac)}")
    print(f"  schedule_truncated={h['schedule_truncated']}  n_ops={h['n_ops']}")
    print()

    print("=== [3] CROSS-CHECK: do both inputs agree on the dominant engine? ===")
    meas_scalar_frac = m["scalar_total_us"] / m["t_measured_us"]
    struct_scalar_frac = h["pipe_us"].get("PIPE_S", 0.0) / tot if tot else 0.0
    print(f"  msprof  scalar share = {meas_scalar_frac*100:.1f}%  (measured)")
    print(f"  HIVM    PIPE_S share = {struct_scalar_frac*100:.1f}%  (structural)")
    print(f"  => AGREE on scalar-bound" if meas_scalar_frac > 0.5 and struct_scalar_frac > 0.5
          else "  => disagree")
    print()

    if args.t_bound is not None:
        head = m["t_measured_us"] - args.t_bound
        print("=== [4] author-headroom attribution ===")
        print(f"  T_bound_DSL        = {args.t_bound:.0f} us")
        print(f"  T_measured         = {m['t_measured_us']:.0f} us")
        print(f"  author headroom    = {head:.0f} us "
              f"({head/m['t_measured_us']*100:.1f}% of T_measured)")
        print(f"  measured scalar    = {m['scalar_total_us']:.0f} us "
              f"-- ALONE exceeds the entire bound "
              f"({m['scalar_total_us']/args.t_bound:.2f}x T_bound)")
        print("  ROOT CAUSE: scalar throughput is modeled at the full VECTOR rate")
        print("    (component_model.py:290, constants.py:252 — US-SB-007 soundness")
        print("     fallback). Scalar pipe is ~unmodeled; its real cost reappears")
        print("     here as 'author headroom'. This is MODEL headroom, not author.")


if __name__ == "__main__":
    main()
