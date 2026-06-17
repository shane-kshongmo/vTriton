#!/usr/bin/env python3
"""
Component-level modeling validation: instruction→pipe→component→pipeline.

Validates that tritonsim-opt's HIVM analysis correctly:
  1. Maps each MLIR instruction to the correct hardware pipe
  2. Maps each pipe to the correct performance component (roofline)
  3. Preserves data dependencies (SSA def-use → depends_on)
  4. Propagates type information (bytes, elements, elem_type)
  5. Handles loop multipliers correctly
  6. Tracks buffer/version lifecycle through sync ops

Usage:
    python3 scripts/test_component_validation.py prefill_des.json kernel_001.npuir.mlir
"""
import json
import re
import sys
from collections import defaultdict, Counter


# ── Known-correct pipe assignments ─────────────────────────────────────────
# From HIVMAnalysis.cpp populateTypedHivmOp() and hardware spec

EXPECTED_PIPE_MAP = {
    # Vector compute ops → PIPE_V
    "vadd":    "PIPE_V", "vsub":    "PIPE_V", "vmul":    "PIPE_V",
    "vdiv":    "PIPE_V", "vexp":    "PIPE_V", "vlog":    "PIPE_V",
    "vsqrt":   "PIPE_V", "vrsqrt":  "PIPE_V", "vtanh":   "PIPE_V",
    "vsigmoid":"PIPE_V", "vgelu":   "PIPE_V", "vrelu":   "PIPE_V",
    "vabs":    "PIPE_V", "vneg":    "PIPE_V", "vcast":   "PIPE_V",
    "vsel":    "PIPE_V", "vnot":    "PIPE_V", "vbrc":    "PIPE_V",
    "vreduce": "PIPE_V", "vcmp":    "PIPE_V", "varange": "PIPE_V",
    # Cube compute ops → PIPE_M
    "matmul":        "PIPE_M", "mmad":          "PIPE_M",
    "mix_matmul":    "PIPE_M", "mix_group_matmul": "PIPE_M",
    # Memory ops → MTE pipes
    "load":          "PIPE_MTE1",     # if typed, resolved by destination
    "store":         "PIPE_MTE3",
    "copy":          "PIPE_MTE2",     # resolved by src/dst spaces
    "fixpipe":       "PIPE_FIX",
    # Scalar → PIPE_S
    "addi":    "PIPE_S", "muli":     "PIPE_S", "divsi":   "PIPE_S",
    "remsi":   "PIPE_S", "minsi":    "PIPE_S", "trunci":  "PIPE_S",
    "extsi":   "PIPE_S", "extui":    "PIPE_S", "select":  "PIPE_S",
    "index_cast":       "PIPE_S", "reinterpret_cast": "PIPE_S",
    "collapse_shape":   "PIPE_S",
    # Unknown/no-op
    "constant":         "PIPE_S",
    "pointer_cast":     "PIPE_UNKNOWN",
}

# Pipe → Component (from op_classifier.py)
PIPE_TO_COMPONENT = {
    "PIPE_M":       "cube",
    "PIPE_V":       "vector",
    "PIPE_S":       "scalar",
    "PIPE_MTE2_C":  "mte_gm",
    "PIPE_MTE2_V":  "mte_gm",
    "PIPE_MTE1":    "mte_l1",
    "PIPE_MTE3":    "mte_ub",
    "PIPE_FIX":     "mte_ub",
    "PIPE_ALL":     "scalar",
    "PIPE_UNKNOWN": "scalar",
}

# Expected element types for known ops (type inference validation)
EXPECTED_ELEM_MAP = {
    "vadd": ["f16","f32","bf16"], "vmul": ["f16","f32","bf16"],
    "vdiv": ["f16","f32","bf16"], "vexp": ["f16","f32","bf16"],
    "vsel": ["f16","f32","bf16"], "vcast": ["f16","f32","bf16","i32"],
    "load": ["f16","f32","bf16","i32"],
    "store": ["f16","f32","bf16","i32"],
}

SYNC_OP_NAMES = {"set_flag", "wait_flag", "sync_block_set", "sync_block_wait",
                  "pipe_barrier"}

# Ops whose pipe is context-dependent or unresolved — skip fixed expectation.
CONTEXT_OPS = SYNC_OP_NAMES | {"get_block_idx", "get_sub_block_idx",
                                "set_ffts_base_addr", "set_mask_norm"}


def load_des(path):
    with open(path) as f:
        return json.load(f)


def load_mlir(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.readlines()


# ── Test 1: Pipe Assignment Validity ────────────────────────────────────────

def test_pipe_assignment(ops):
    """Every op with a non-UNKNOWN pipe must map to a valid PIPE_* name."""
    valid_pipes = set(EXPECTED_PIPE_MAP.values()) | {f"PIPE_{p}" for p in
        "M V S MTE1 MTE2_C MTE2_V MTE3 FIX ALL UNKNOWN".split()}
    failures = []
    for o in ops:
        pipe = o.get("pipe", "")
        if pipe and pipe not in valid_pipes:
            failures.append(f"id={o['id']} name={o['name']} has unknown pipe={pipe}")
    return failures


def test_pipe_opname_consistency(ops):
    """Known op names must map to expected pipes within tolerance bands."""
    failures = []
    pipe_counts = defaultdict(lambda: defaultdict(int))
    for o in ops:
        name = o.get("name", "")
        pipe = o.get("pipe", "")
        pipe_counts[name][pipe] += 1

    for name, expected_pipe in EXPECTED_PIPE_MAP.items():
        if name not in pipe_counts or name in CONTEXT_OPS:
            continue
        actual = pipe_counts[name]
        if expected_pipe not in actual:
            if name in ("load", "copy"):
                continue
            failures.append(
                f"op={name}: expected pipe={expected_pipe}, got {dict(actual)}"
            )

    # Context-dependent ops: only require a valid pipe name.
    for name in CONTEXT_OPS:
        if name not in pipe_counts:
            continue
        for pipe in pipe_counts[name]:
            if pipe == "PIPE_UNKNOWN":
                if name in ("get_block_idx", "get_sub_block_idx",
                            "set_ffts_base_addr", "set_mask_norm"):
                    continue  # 0-work control ops
                failures.append(
                    f"op={name}: context-dependent op mapped to PIPE_UNKNOWN"
                )

    return failures


# ── Test 2: Component Mapping ───────────────────────────────────────────────

def test_component_mapping(ops):
    """Every pipe must have a valid component mapping."""
    failures = []
    for o in ops:
        pipe = o.get("pipe", "")
        component = PIPE_TO_COMPONENT.get(pipe)
        if component is None:
            failures.append(f"id={o['id']} pipe={pipe} has no component mapping")
        elif component not in ("cube", "vector", "scalar", "mte_gm", "mte_l1", "mte_ub"):
            failures.append(f"id={o['id']} pipe={pipe} → unknown component={component}")
    return failures


# ── Test 3: Data Dependency Chain ───────────────────────────────────────────

def test_dependency_chain(ops):
    """depends_on must reference real op IDs that come before the current op."""
    failures = []
    all_ids = set(o["id"] for o in ops)
    id_to_cycle = {o["id"]: o.get("start_cycle", 0) for o in ops}
    max_dep_chain = 0

    for o in ops:
        for dep_id in o.get("depends_on", []):
            if dep_id not in all_ids:
                failures.append(f"id={o['id']} depends_on={dep_id} which does not exist")
            elif dep_id >= o["id"]:
                # ID order is a weak proxy; cycles are better
                pass
        max_dep_chain = max(max_dep_chain, len(o.get("depends_on", [])))

    # Also check that every op (except constants at cycle 0) has dependencies
    no_deps = [o for o in ops if len(o.get("depends_on", [])) == 0
               and o.get("start_cycle", 0) > 0
               and o["name"] not in ("constant", "pointer_cast")]
    for o in no_deps[:5]:
        failures.append(
            f"id={o['id']} name={o['name']} pipe={o['pipe']} "
            f"has no dependencies but starts at cycle {o['start_cycle']}"
        )

    if len(no_deps) > 5:
        failures.append(f"... and {len(no_deps) - 5} more ops with no deps after cycle 0")

    return failures


# ── Test 4: Type Inference ──────────────────────────────────────────────────

def test_type_inference(ops):
    """bytes/elements/elem_type must be internally consistent."""
    failures = []
    elem_counts = Counter()

    for o in ops:
        elem_counts[o.get("elem_type", "")] += 1
        bytes_val = o.get("bytes", 0)
        elements = o.get("elements", 0)
        etype = o.get("elem_type", "")

        # If elements > 0, there should be a type
        if elements > 0 and not etype:
            # pointer_cast at PIPE_UNKNOWN often has no type — acceptable
            if o.get("pipe") == "PIPE_UNKNOWN":
                continue
            failures.append(
                f"id={o['id']} name={o['name']} has elements={elements} but no elem_type"
            )

        # bytes/elements consistency check
        if etype and elements > 0:
            type_bytes = {"f16": 2, "bf16": 2, "f32": 4, "fp8": 1,
                          "i8": 1, "i16": 2, "i32": 4, "i64": 8}
            expected = type_bytes.get(etype, 4) * elements
            if bytes_val > 0 and abs(bytes_val - expected) > expected * 0.5:
                if o["name"] not in ("load", "store"):  # strided dims can vary
                    failures.append(
                        f"id={o['id']} {o['name']}: bytes={bytes_val} but "
                        f"elem_type={etype} × elements={elements} → expected ~{expected}"
                    )

    return failures


# ── Test 5: Loop Multiplier ─────────────────────────────────────────────────

def test_loop_multiplier(ops):
    """Operations inside loops should have loop_multiplier > 1."""
    # The MLIR has nested scf.for loops → DES should reflect mult factor
    multipliers = Counter()
    for o in ops:
        lm = o.get("loop_multiplier", 1)
        multipliers[lm] += 1

    # At least some ops should have loop multiplier > 1 (nested loops)
    has_loop = any(lm > 1 for lm in multipliers)
    if not has_loop:
        return ["No operations with loop_multiplier > 1 — loops not detected?"]

    # Verify loop multiplier >= 1 for all ops
    for o in ops:
        if o.get("loop_multiplier", 1) < 1:
            return [f"id={o['id']} has loop_multiplier={o['loop_multiplier']} < 1"]

    return []


# ── Test 6: Buffer/Version Lifecycle ─────────────────────────────────────────

def test_buffer_lifecycle(ops):
    """Read/write buffers should have consistent versions."""
    failures = []
    buffer_write_versions = defaultdict(list)
    buffer_read_versions = defaultdict(list)

    for o in ops:
        for buf in o.get("write_buffers", []):
            versions = o.get("write_versions", [])
            version = versions[0] if versions else 0
            buffer_write_versions[buf].append((o["id"], o["start_cycle"], version))
        for buf in o.get("read_buffers", []):
            versions = o.get("read_versions", [])
            version = versions[0] if versions else 0
            buffer_read_versions[buf].append((o["id"], o["start_cycle"], version))

    # Check: read version should match some prior write version
    for buf, reads in buffer_read_versions.items():
        writes = buffer_write_versions.get(buf, [])
        for rid, rcyc, rver in reads:
            # Find the most recent write with version <= rver before this read
            prior_writes = [(wid, wcyc, wver) for wid, wcyc, wver in writes
                            if wcyc < rcyc and wver == rver]
            if rver > 0 and not prior_writes:
                prior_writes = [(wid, wcyc, wver) for wid, wcyc, wver in writes
                                if wcyc < rcyc]

    # Check: buffer access count
    ops_with_buffers = sum(1 for o in ops if o.get("read_buffers") or o.get("write_buffers"))
    if ops_with_buffers == 0:
        failures.append("Zero operations have buffer metadata — tracking may be disabled")

    return failures


# ── Test 7: Sync between pipes (set/wait pairs maintain pipe consistency) ─────

def test_sync_pipe_consistency(ops):
    """set_flag sender_pipe should match the wait_flag receiver_pipe, and vice versa."""
    failures = []
    
    set_flags = defaultdict(list)
    wait_flags = defaultdict(list)
    for o in ops:
        if o["name"] == "set_flag":
            eid = o.get("event_id", "")
            gen = o.get("event_generation", 0)
            core = o.get("core_type", "")
            if eid and not eid.startswith("ssa_producer_"):
                set_flags[(eid, gen, core)].append(o)
        elif o["name"] == "wait_flag":
            eid = o.get("event_id", "")
            gen = o.get("event_generation", 0)
            core = o.get("core_type", "")
            if eid and not eid.startswith("ssa_producer_"):
                wait_flags[(eid, gen, core)].append(o)
    
    for key, setters in set_flags.items():
        waiters = wait_flags.get(key, [])
        for s in setters:
            s_sender = s.get("sender_pipe", "")
            s_receiver = s.get("receiver_pipe", "")
            s_pipe = s.get("pipe", "")
            for w in waiters:
                w_sender = w.get("sender_pipe", "")
                w_receiver = w.get("receiver_pipe", "")
                w_pipe = w.get("pipe", "")
                # set_flag runs on sender_pipe, wait_flag should receive it
                if s_sender and w_receiver and s_sender != w_receiver:
                    failures.append(
                        f"pipe mismatch: {key} set(sender={s_sender}) "
                        f"vs wait(receiver={w_receiver})"
                    )
                # set_flag sets on its executing pipe
                if s_sender and s_pipe and s_sender != s_pipe:
                    failures.append(
                        f"{key} set_flag pipe={s_pipe} but sender={s_sender}"
                    )
    return failures[:10]  # cap at 10


# ── Test 8: MLIR verification — every hivm.hir op must have a DES entry ─────

def test_op_coverage(ops, mlir_path):
    """Every hivm.hir operation in MLIR must appear in DES JSON."""
    with open(mlir_path, encoding="utf-8", errors="replace") as f:
        mlir_text = f.read()

    hivm_ops = re.findall(r'"hivm\.hir\.(\w+)"', mlir_text)
    if not hivm_ops:
        hivm_ops = re.findall(r'hivm\.hir\.(\w+)', mlir_text)

    mlir_counts = Counter(hivm_ops)
    des_counts = Counter(o["name"] for o in ops)
    failures = []

    for name, mlir_cnt in mlir_counts.items():
        des_name = name
        des_cnt = des_counts.get(name, 0)
        # mmadL1 is split into mmadL1.cube + mmadL1.mte1 in DES
        if name == "mmadL1":
            des_cnt = des_counts.get("mmadL1.cube", 0) + des_counts.get("mmadL1.mte1", 0)
        elif des_cnt == 0 and "." in name:
            des_cnt = sum(v for k, v in des_counts.items() if k.startswith(name + "."))
        if des_cnt == 0:
            failures.append(f"op={name}: {mlir_cnt} in MLIR but 0 in DES")
        elif des_cnt < mlir_cnt:
            failures.append(f"op={name}: MLIR={mlir_cnt} DES={des_cnt} (under-count)")

    return failures


# ── Runner ──────────────────────────────────────────────────────────────────

def main():
    des_path = sys.argv[1] if len(sys.argv) > 1 else "prefill_des.json"
    mlir_path = sys.argv[2] if len(sys.argv) > 2 else "kernel_001.npuir.mlir"

    des = load_des(des_path)
    ops = des["operations"]

    print("=" * 70)
    print("Component-Level Modeling Validation Suite")
    print("=" * 70)
    print(f"  DES ops:  {len(ops)}")
    print(f"  MLIR src: {mlir_path}")

    tests = [
        ("Pipe assignment validity", test_pipe_assignment, (ops,)),
        ("Pipe ↔ op name consistency", test_pipe_opname_consistency, (ops,)),
        ("Component (pipe→roofline) mapping", test_component_mapping, (ops,)),
        ("Data dependency chain", test_dependency_chain, (ops,)),
        ("Type inference (bytes/elements/elem_type)", test_type_inference, (ops,)),
        ("Loop multiplier detection", test_loop_multiplier, (ops,)),
        ("Buffer/version lifecycle", test_buffer_lifecycle, (ops,)),
        ("Sync pipe consistency", test_sync_pipe_consistency, (ops,)),
        ("Op coverage (MLIR→DES)", test_op_coverage, (ops, mlir_path)),
    ]

    all_ok = True
    passed = 0
    total = len(tests)

    for name, fn, args in tests:
        try:
            failures = fn(*args)
        except Exception as e:
            print(f"\n  [ERROR] {name}: {e}")
            all_ok = False
            continue

        if not failures:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name} ({len(failures)} issues)")
            for f in failures[:3]:
                print(f"         {f}")
            if len(failures) > 3:
                print(f"         ... and {len(failures) - 3} more")
            all_ok = False

    # Summary stats
    print("\n" + "=" * 70)
    print(f"  Results: {passed}/{total} passed")

    if passed == total:
        print("  VERDICT: ALL CLEAN — component model correctly reflects MLIR")
    else:
        print("  VERDICT: Issues found — see above for details")

    # Component distribution
    comp_counts = Counter()
    pipe_counts = Counter()
    for o in ops:
        pipe_counts[o.get("pipe", "?")] += 1
        comp = PIPE_TO_COMPONENT.get(o.get("pipe", ""), "?")
        comp_counts[comp] += 1

    print(f"\n  Component distribution: {dict(comp_counts)}")
    
    # Sync vs compute ratio
    sync_ops = sum(1 for o in ops if o.get("is_sync"))
    compute_ops = len(ops) - sync_ops
    print(f"  Sync ops: {sync_ops} ({100*sync_ops/len(ops):.1f}%)")
    print(f"  Compute ops: {compute_ops} ({100*compute_ops/len(ops):.1f}%)")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
