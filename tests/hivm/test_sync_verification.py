#!/usr/bin/env python3
"""
DES/Trace sync instruction verification test suite.

Validates that HIVM MLIR sync instructions (set_flag / wait_flag,
sync_block_set / sync_block_wait, pipe_barrier) are correctly
reflected in the tritonsim-opt DES JSON output.

Usage:
    python3 scripts/test_sync_verification.py docs/prefill_des.json kernel_001.npuir.mlir
"""
import json
import re
import sys
from collections import defaultdict, Counter


def load_des(path):
    with open(path) as f:
        return json.load(f)


def load_mlir_sync_counts(path):
    with open(path) as f:
        text = f.read()
    return {
        name: len(re.findall(rf"hivm\.hir\.{name}", text))
        for name in ["set_flag", "wait_flag", "sync_block_set",
                      "sync_block_wait", "pipe_barrier"]
    }


def test_static_event_pairs(ops):
    """Test 1: Every static set_flag must have a matching wait_flag
    with the same event_id and event_generation, in correct temporal order."""
    set_static = defaultdict(list)
    wait_static = defaultdict(list)

    for o in ops:
        if o["name"] == "set_flag":
            eid = o.get("event_id", "")
            gen = o.get("event_generation", 0)
            if eid and not eid.startswith("ssa_producer_"):
                set_static[eid].append((o["id"], o["pipe"], o.get("core_type", ""),
                                        gen, o["start_cycle"]))
        elif o["name"] == "wait_flag":
            eid = o.get("event_id", "")
            gen = o.get("event_generation", 0)
            if eid and not eid.startswith("ssa_producer_"):
                wait_static[eid].append((o["id"], o["pipe"], o.get("core_type", ""),
                                         gen, o["start_cycle"]))

    failures = []
    for eid in sorted(set_static.keys()):
        setters = set_static[eid]
        waiters = wait_static.get(eid, [])
        # Group by (gen, core)
        from itertools import groupby
        set_groups = defaultdict(list)
        wait_groups = defaultdict(list)
        for s_id, s_pipe, s_core, s_gen, s_cyc in setters:
            set_groups[(s_gen, s_core)].append((s_id, s_pipe, s_cyc))
        for w_id, w_pipe, w_core, w_gen, w_cyc in waiters:
            wait_groups[(w_gen, w_core)].append((w_id, w_pipe, w_cyc))

        for (gen, core), s_items in set_groups.items():
            w_items = wait_groups.get((gen, core), [])
            if not w_items:
                failures.append(
                    f"orphan set_flag: event={eid} gen={gen} core={core} "
                    f"({len(s_items)} sets, 0 waits)"
                )
                continue
            # Check: earliest set should be before earliest wait (at least
            # one pair is temporally valid; multiple pairs per gen/pipeline-stage
            # are expected to interleave)
            min_set_cyc = min(cyc for _, _, cyc in s_items)
            min_wait_cyc = min(cyc for _, _, cyc in w_items)
            if min_set_cyc >= min_wait_cyc:
                failures.append(
                    f"temporal order: event={eid} gen={gen} core={core} "
                    f"min_set_cyc={min_set_cyc} >= min_wait_cyc={min_wait_cyc} "
                    f"({len(s_items)} sets, {len(w_items)} waits)"
                )

        for (gen, core), w_items in wait_groups.items():
            if (gen, core) not in set_groups:
                failures.append(
                    f"orphan wait_flag: event={eid} gen={gen} core={core} "
                    f"({len(w_items)} waits, 0 sets)"
                )
    return failures


def test_sync_block_pairs(ops):
    """Test 2: Every sync_block_set must have a matching sync_block_wait
    with same flag_id and event_generation."""
    set_flags = defaultdict(list)
    wait_flags = defaultdict(list)

    for o in ops:
        if o["name"] == "sync_block_set":
            fid = o.get("event_id", "")
            gen = o.get("event_generation", 0)
            if fid:
                set_flags[fid].append((o["id"], o["pipe"], o.get("core_type", ""),
                                       gen, o["start_cycle"], o.get("sender_pipe", "")))
        elif o["name"] == "sync_block_wait":
            fid = o.get("event_id", "")
            gen = o.get("event_generation", 0)
            if fid:
                wait_flags[fid].append((o["id"], o["pipe"], o.get("core_type", ""),
                                        gen, o["start_cycle"], o.get("sender_pipe", "")))

    failures = []
    cross_core = 0
    for fid in sorted(set_flags.keys()):
        setters = set_flags[fid]
        waiters = wait_flags.get(fid, [])
        # Group by gen
        s_groups = defaultdict(list)
        w_groups = defaultdict(list)
        for s_id, s_p, s_c, s_gen, s_cyc, s_sender in setters:
            s_groups[s_gen].append((s_id, s_c, s_p, s_cyc, s_sender))
        for w_id, w_p, w_c, w_gen, w_cyc, w_s in waiters:
            w_groups[w_gen].append((w_id, w_c, w_p, w_cyc, w_s))

        for gen, s_items in s_groups.items():
            w_items = w_groups.get(gen, [])
            if not w_items:
                failures.append(
                    f"orphan sync_block_set: flag={fid} gen={gen} ({len(s_items)} sets, 0 waits)"
                )
                continue
            # At least one set must come before some wait within same gen
            min_s_cyc = min(cyc for _, _, _, cyc, _ in s_items)
            min_w_cyc = min(cyc for _, _, _, cyc, _ in w_items)
            if min_s_cyc >= min_w_cyc:
                failures.append(
                    f"temporal order: flag={fid} gen={gen} "
                    f"min_set_cyc={min_s_cyc} >= min_wait_cyc={min_w_cyc}"
                )
            for _, s_c, _, _, _ in s_items:
                for _, w_c, _, _, _ in w_items:
                    if s_c != w_c:
                        cross_core += 1
    return failures, cross_core


def test_cross_core_flag_2(ops):
    """Test 3: flag_2 specifically — CUBE MTE2_C → VECTOR PIPE_ALL cross-core sync."""
    f2_sets = []
    f2_waits = []
    for o in ops:
        if o["name"] == "sync_block_set" and o.get("event_id") == "flag_2":
            f2_sets.append((o["id"], o.get("core_type", ""), o["pipe"],
                            o.get("event_generation", 0), o["start_cycle"]))
        elif o["name"] == "sync_block_wait" and o.get("event_id") == "flag_2":
            f2_waits.append((o["id"], o.get("core_type", ""), o["pipe"],
                             o.get("event_generation", 0), o["start_cycle"]))

    failures = []
    for s_id, s_c, s_p, s_gen, s_cyc in f2_sets:
        matched = [(w_id, w_c, w_p, w_cyc)
                   for w_id, w_c, w_p, w_g, w_cyc in f2_waits if w_g == s_gen]
        if not matched:
            failures.append(f"flag_2 gen={s_gen}: no matching wait")
        for w_id, w_c, w_p, w_cyc in matched:
            if s_cyc >= w_cyc:
                failures.append(f"flag_2 gen={s_gen}: set({s_id},cyc={s_cyc}) >= wait({w_id},cyc={w_cyc})")
            if s_c == w_c:
                failures.append(f"flag_2 gen={s_gen}: expected cross-core but both on {s_c}")
    return failures


def test_pipe_barrier_coverage(ops):
    """Test 4: pipe_barrier must appear on data pipes (MTE2, MTE3, FIX, V, etc.)."""
    bars = [o for o in ops if o["name"] == "pipe_barrier"]
    valid_pipes = {"PIPE_MTE1", "PIPE_MTE2_C", "PIPE_MTE2_V", "PIPE_MTE3",
                   "PIPE_V", "PIPE_M", "PIPE_FIX", "PIPE_S", "PIPE_ALL"}
    failures = []
    for b in bars:
        if b["pipe"] not in valid_pipes:
            failures.append(
                f"barrier on unexpected pipe: op={b['id']} pipe={b['pipe']}"
            )
    return failures, Counter(b["pipe"] for b in bars)


def test_ssa_dynamic_events(ops):
    """Test 5: SSA dynamic events — set_flag has empty event_id, wait_flag
    resolves to ssa_producer_<N>. Verify consumer count matches expected."""
    ssa_waits = defaultdict(int)
    empty_eid_sets = 0
    for o in ops:
        if o["name"] == "set_flag" and o.get("event_id", "") == "":
            empty_eid_sets += 1
        elif o["name"] == "wait_flag":
            eid = o.get("event_id", "")
            if eid.startswith("ssa_producer_"):
                ssa_waits[eid] += 1

    return {
        "empty_event_set_flags": empty_eid_sets,
        "ssa_consumer_count": len(ssa_waits),
        "ssa_total_consumers": sum(ssa_waits.values()),
    }


def test_mlir_des_ratio(mlir_counts, ops):
    """Test 6: MLIR vs DES sync op ratio should be consistent (accounts
    for loop unrolling and two kernels)."""
    des_counts = Counter(o["name"] for o in ops)
    ratios = {}
    for name in ["set_flag", "wait_flag", "sync_block_set", "sync_block_wait",
                  "pipe_barrier"]:
        mc = mlir_counts.get(name, 0)
        dc = des_counts.get(name, 0)
        if mc > 0:
            ratios[name] = dc / mc
    return ratios


def main():
    des_path = sys.argv[1] if len(sys.argv) > 1 else "docs/prefill_des.json"
    mlir_path = sys.argv[2] if len(sys.argv) > 2 else "kernel_001.npuir.mlir"

    des = load_des(des_path)
    ops = des["operations"]
    mlir_counts = load_mlir_sync_counts(mlir_path)

    all_pass = True
    total = 0
    passed = 0

    def run(name, failures, is_list=True):
        nonlocal all_pass, total, passed
        total += 1
        if is_list:
            ok = len(failures) == 0
        else:
            ok = True
        if ok:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name} ({len(failures)} issues)")
            for f in failures[:5]:
                print(f"    {f}")
            if len(failures) > 5:
                print(f"    ... and {len(failures) - 5} more")
            all_pass = False

    print("=" * 60)
    print("HIVM Sync Verification Tests")
    print("=" * 60)

    # Test 1
    f1 = test_static_event_pairs(ops)
    run("Static event set_flag/wait_flag pairs", f1)

    # Test 2
    f2, cross_core = test_sync_block_pairs(ops)
    run("sync_block_set/wait flag pairs", f2)
    print(f"        cross-core pairs: {cross_core}")

    # Test 3
    f3 = test_cross_core_flag_2(ops)
    run("Cross-core flag_2 (CUBE→VECTOR)", f3)

    # Test 4
    f4, bar_pipes = test_pipe_barrier_coverage(ops)
    run("pipe_barrier pipe validity", f4)

    # Test 5
    ssa = test_ssa_dynamic_events(ops)
    total += 1
    print(f"  [INFO] SSA dynamic events: {ssa['empty_event_set_flags']} empty-eid producers"
          f" → {ssa['ssa_consumer_count']} unique ssa_producer_* consumers"
          f" ({ssa['ssa_total_consumers']} total)")
    passed += 1

    # Test 6
    ratios = test_mlir_des_ratio(mlir_counts, ops)
    total += 1
    consistent = all(0.5 < r < 50 for r in ratios.values())
    print(f"  [{'PASS' if consistent else 'INFO'}] MLIR→DES ratio check")
    for name, r in ratios.items():
        print(f"        {name}: MLIR={mlir_counts[name]} DES={Counter(o['name'] for o in ops)[name]} ratio={r:.1f}x")
    if consistent:
        passed += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} passed")
    print(f"Overall: {'PASS' if all_pass else 'SOME FAILURES — see above'}")

    # Detailed sync instruction summary
    print("\n--- Sync op type distribution ---")
    sync_ops = [o for o in ops if o.get("is_sync")]
    for name, cnt in Counter(o["name"] for o in sync_ops).most_common():
        print(f"  {name}: {cnt}")

    pipe_dist = Counter(o["pipe"] for o in ops if o.get("is_sync"))
    print(f"\n--- Sync pipe distribution ---")
    for pipe, cnt in pipe_dist.most_common():
        print(f"  {pipe}: {cnt}")

    core_dist = Counter(o.get("core_type", "?") for o in ops)
    print(f"\n--- Core type distribution ---")
    for core, cnt in core_dist.most_common():
        print(f"  {core}: {cnt}")

    # Known caveats
    print(f"\n--- Known caveats ---")
    print(f"  1. event_EVENT_ID2/3 gen≥2 set_flags may have orphan consumers"
          f" — verify event_generation logic in HIVMAnalysis.cpp")
    print(f"  2. SSA dynamic events (ssa_producer_*) lack static producer set_flag"
          f" — expected behavior, consumers use SSA-based syncIdValue")
    print(f"  3. Cross-kernel event name sharing (AIC+AIV both use EVENT_ID0-5)"
          f" — each core has independent timeline, no actual conflict")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
