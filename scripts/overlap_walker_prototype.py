#!/usr/bin/env python3
"""Pipeline-overlap walker (prototype) — validate Gap-OVL before wiring it in.

Tests the reshaped hypothesis: the chunk_kda author headroom is *exposed
control/sync overhead* (pipeline-overlap inefficiency), not scalar compute and
not VEC->scalar fallback.

Method (inputs = HIVM/DES schedule + msprof):
  1. Classify every DES op by pipe category:
       control_sync : PIPE_S + barriers/flags (is_sync/is_barrier) + PIPE_ALL
       compute      : PIPE_V, PIPE_M
       memory       : PIPE_MTE2_V/C, PIPE_MTE3, PIPE_MTE1, PIPE_FIX
  2. Sweep the emitted schedule timeline [0, critical_path) using start/end_cycle
     events.  For each interval, record which categories are simultaneously
     active.
  3. MODEL-predicted exposed control = fraction of the critical path where
     control_sync is active and NO compute/memory op overlaps it.
  4. MEASURED exposed control = msprof aiv_scalar_ratio (scalar share of the
     vector core's wall-clock).
  5. Gap-OVL = measured - model.  Interpreting the two outcomes:
       - model_exposed LOW, measured HIGH  -> model over-assumes overlap (true
         overlap deficit; the schedule hides control the hardware exposes).
       - model_exposed already HIGH        -> model predicts serialization but
         prices each control op at ~1 cycle; the gap is under-priced control
         cost, not overlap.

Usage:
    python scripts/overlap_walker_prototype.py \
        --des .omc/research/hw_runs/kda_des.json \
        --csv .omc/research/hw_runs/chunk_kda_op_summary.csv \
        --kernel chunk_kda_bwd
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math

_COMPUTE = {"PIPE_V", "PIPE_M"}
_MEMORY = {"PIPE_MTE2_V", "PIPE_MTE2_C", "PIPE_MTE3", "PIPE_MTE1", "PIPE_FIX"}
_CONTROL = {"PIPE_S", "PIPE_ALL"}


def categorize(op: dict) -> str:
    if op.get("is_sync") or op.get("is_barrier"):
        return "control_sync"
    p = op["pipe"]
    if p in _CONTROL:
        return "control_sync"
    if p in _COMPUTE:
        return "compute"
    if p in _MEMORY:
        return "memory"
    return "other"


def sweep(ops: list[dict]) -> dict:
    """Sweep the schedule; return per-state cycle counts keyed by active set."""
    # Build +1/-1 events per category at op boundaries.
    events = collections.defaultdict(lambda: collections.Counter())
    for o in ops:
        s, e = o["start_cycle"], o["end_cycle"]
        if e <= s:
            continue
        cat = categorize(o)
        if cat == "other":
            continue
        events[s][cat] += 1
        events[e][cat] -= 1

    times = sorted(events)
    active = collections.Counter()
    state_cycles = collections.Counter()  # frozenset(active cats) -> cycles
    prev = None
    for t in times:
        if prev is not None and t > prev:
            cats = frozenset(c for c, n in active.items() if n > 0)
            state_cycles[cats] += (t - prev)
        for c, d in events[t].items():
            active[c] += d
        prev = t
    return state_cycles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--des", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--kernel", default="chunk_kda_bwd")
    args = ap.parse_args()

    d = json.load(open(args.des))
    ops = d["operations"]
    ghz = d["clock_ghz"]
    crit = max(o["end_cycle"] for o in ops)

    state_cycles = sweep(ops)
    total = sum(state_cycles.values())

    # Aggregate the states of interest.
    exposed_control = 0   # control active, no compute, no memory
    control_overlapped = 0
    compute_active = 0
    idle_or_other = 0
    for cats, cyc in state_cycles.items():
        has_c = "control_sync" in cats
        has_comp = "compute" in cats
        has_mem = "memory" in cats
        if has_c and not has_comp and not has_mem:
            exposed_control += cyc
        elif has_c and (has_comp or has_mem):
            control_overlapped += cyc
        if has_comp:
            compute_active += cyc
        if not cats:
            idle_or_other += cyc

    print(f"Kernel: {args.kernel}")
    print(f"clock_ghz={ghz}  critical_path={crit} cyc ({crit/(ghz*1000):.2f} us)")
    print(f"swept span={total} cyc\n")

    print("=== schedule overlap states (model view) ===")
    def pct(x): return f"{x/total*100:5.1f}%" if total else "  n/a"
    print(f"  exposed control/sync (no compute, no mem) : {exposed_control:7d} cyc  {pct(exposed_control)}")
    print(f"  control overlapped w/ compute or memory   : {control_overlapped:7d} cyc  {pct(control_overlapped)}")
    print(f"  compute active (any overlap)              : {compute_active:7d} cyc  {pct(compute_active)}")

    # op-count + busy by category (loop_multiplier scaled), for context
    busy = collections.Counter(); nops = collections.Counter()
    for o in ops:
        cat = categorize(o)
        if cat == "other": continue
        busy[cat] += o["duration"] * (o.get("loop_multiplier", 1) or 1)
        nops[cat] += 1
    print("\n=== structural busy by category (loop-scaled) ===")
    sb = sum(busy.values())
    for c, b in busy.most_common():
        print(f"  {c:14s} n={nops[c]:4d}  {b:8d} cyc  {b/sb*100:5.1f}%")

    model_exposed_frac = exposed_control / total if total else 0.0

    if args.csv:
        with open(args.csv) as f:
            rows = [r for r in csv.DictReader(f) if args.kernel in r.get("Op Name", "")]
        if rows:
            r = rows[0]
            def ff(k):
                try: return float(r[k])
                except: return math.nan
            measured_scalar = ff("aiv_scalar_ratio")
            print("\n=== model vs measured (Gap-OVL) ===")
            print(f"  model  exposed-control fraction (of crit path) : {model_exposed_frac*100:5.1f}%")
            print(f"  measured scalar share (aiv_scalar_ratio)       : {measured_scalar*100:5.1f}%")
            gap = measured_scalar - model_exposed_frac
            print(f"  Gap-OVL (measured - model)                     : {gap*100:+5.1f} pts")
            if model_exposed_frac < 0.4 and measured_scalar > 0.6:
                print("  => MODEL OVER-ASSUMES OVERLAP: schedule hides control the HW exposes.")
            elif model_exposed_frac > 0.6:
                print("  => model already predicts serialization; gap is UNDER-PRICED control")
                print("     cost (1 cyc/op vs real stall), not an overlap deficit.")
            else:
                print("  => mixed; inspect both terms.")


if __name__ == "__main__":
    main()
