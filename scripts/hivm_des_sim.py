#!/usr/bin/env python3
"""
HIVM Pipe-based Discrete Event Simulation (DES).

Models each hardware pipe as an independent FIFO pipeline, organized into
two cores (AIC / AIV) that run in parallel:

  AIC (Cube) core pipes : PIPE_M, PIPE_MTE1, PIPE_MTE2_C, PIPE_FIX, PIPE_S
  AIV (Vector) core pipes: PIPE_V, PIPE_MTE2_V, PIPE_MTE3, PIPE_S

Synchronization primitives:

  Intra-core (set_flag / wait_flag):
    Synchronizes different pipes within the SAME core.
    set_flag[set_pipe, wait_pipe, event_id] on set_pipe signals wait_pipe.
    wait_flag[set_pipe, wait_pipe, event_id] on wait_pipe blocks until
    the matching set_flag completes.
    Matching key: (sender_pipe, receiver_pipe, event_id), paired sequentially.

  Inter-core (sync_block_set / sync_block_wait):
    Synchronizes between AIC and AIV cores through FFTS hardware.
    sync_block_set on one core signals sync_block_wait on the OTHER core.
    Matching key: flag_id only.  CUBE set -> VECTOR wait, VECTOR set -> CUBE wait.

  pipe_barrier[pipe]:
    Drains all in-flight instructions on the target pipe. Implicit in FIFO
    model — when it reaches the queue head, all prior ops have completed.
    pipe_barrier[PIPE_ALL] waits for ALL pipes on the current core to drain.

Reference: https://gitcode.com/Ascend/AscendNPU-IR/blob/master/docs/source/en/
           developer_guide/dialects/HIVMDialect.md
"""

import json
import heapq
import argparse
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Core / Pipe model
# ---------------------------------------------------------------------------

AIC_PIPES = {"PIPE_M", "PIPE_MTE1", "PIPE_MTE2_C", "PIPE_FIX"}
AIV_PIPES = {"PIPE_V", "PIPE_MTE2_V", "PIPE_MTE3"}
# PIPE_S exists on both cores — disambiguated at load time
# PIPE_ALL / PIPE_UNKNOWN are pseudo-pipes


class PipeFIFO:
    """A hardware pipe with an internal micro-pipeline.

    Each pipe has two timing concepts:
      issue_at:  when the pipe can ACCEPT the next instruction (issue slot free)
      drain_at:  when all in-flight instructions have COMPLETED execution

    For serial pipes (DMA engines): issue_at == drain_at (can't issue new
    until current finishes).
    For pipelined pipes (Cube): issue_at < drain_at (can accept new
    instruction every cycle while previous ones are still in-flight).

    pipe_barrier and set_flag DRAIN the pipeline — they wait until drain_at
    before proceeding.  This ensures set_flag signals actual completion,
    not just issue.
    """

    def __init__(self, name):
        self.name = name
        self.queue = []        # ops in program order
        self.head = 0          # index of next op to dispatch
        self.issue_at = 0      # cycle when pipe can accept next instruction
        self.drain_at = 0      # cycle when all in-flight instructions complete

    def has_pending(self):
        return self.head < len(self.queue)

    def peek(self):
        return self.queue[self.head] if self.has_pending() else None

    def advance(self):
        self.head += 1


def classify_core(op):
    """Return 'AIC' or 'AIV' for an op, based on its core_type and pipe."""
    ct = op["core_type"]
    if ct == "CUBE":
        return "AIC"
    if ct == "VECTOR":
        return "AIV"
    pipe = op["pipe"]
    if pipe in AIC_PIPES:
        return "AIC"
    if pipe in AIV_PIPES:
        return "AIV"
    # CUBE_OR_VECTOR: check pipe
    if pipe == "PIPE_ALL":
        # PIPE_ALL barriers apply to all pipes on a core; determine from context
        return "SHARED"
    return "UNKNOWN"


def disambiguate_pipe(op):
    """Return a unique pipe name, splitting PIPE_S by core."""
    pipe = op["pipe"]
    if pipe == "PIPE_S":
        core = classify_core(op)
        return f"PIPE_S_{core}"
    return pipe


def build_pipe_queues(ops):
    """Separate ops into per-pipe FIFO queues, preserving program order.

    PIPE_S is split into PIPE_S_AIC and PIPE_S_AIV.
    PIPE_UNKNOWN ops get their own queue (instant execution).
    """
    pipes = {}
    for op in ops:
        pipe_name = op["_pipe"]  # disambiguated name
        if pipe_name not in pipes:
            pipes[pipe_name] = PipeFIFO(pipe_name)
        pipes[pipe_name].queue.append(op)
    return pipes


# ---------------------------------------------------------------------------
# Sync key helpers
# ---------------------------------------------------------------------------

def flag_sync_key(op):
    """Intra-core set_flag/wait_flag matching key (no generation)."""
    return (op["sender_pipe"], op["receiver_pipe"], op["event_id"])


def block_sync_key(op):
    """Inter-core sync_block matching key: (flag_id, source_core).

    CUBE sync_block_set -> VECTOR sync_block_wait (and vice versa).
    We key by (flag_id, setting_core) so that CUBE sets pair with VECTOR waits.
    """
    core = classify_core(op)
    if op["name"] == "sync_block_set":
        return (op["event_id"], core)
    elif op["name"] == "sync_block_wait":
        # Wait consumes signals from the OTHER core
        other_core = "AIV" if core == "AIC" else "AIC"
        return (op["event_id"], other_core)
    return None


def build_sync_index(ops):
    """Pre-compute which sync waits have matching sets.

    Returns set of unmatched flag keys and block keys.
    """
    flag_set_keys = defaultdict(int)
    flag_wait_keys = defaultdict(int)
    block_set_keys = defaultdict(int)
    block_wait_keys = defaultdict(int)

    for op in ops:
        if not op["event_id"]:
            continue
        if op["name"] == "set_flag":
            flag_set_keys[flag_sync_key(op)] += 1
        elif op["name"] == "wait_flag":
            flag_wait_keys[flag_sync_key(op)] += 1
        elif op["name"] == "sync_block_set":
            block_set_keys[block_sync_key(op)] += 1
        elif op["name"] == "sync_block_wait":
            block_wait_keys[block_sync_key(op)] += 1

    unmatched_flag = {k for k in flag_wait_keys if k not in flag_set_keys}
    unmatched_block = {k for k in block_wait_keys if k not in block_set_keys}
    return unmatched_flag, unmatched_block


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(ops, clock_ghz):
    """Run pipe-based DES simulation with two-core model."""
    n = len(ops)
    if n == 0:
        return []

    # Disambiguate PIPE_S and annotate ops
    for op in ops:
        op["_pipe"] = disambiguate_pipe(op)
        op["_core"] = classify_core(op)

    # Build per-pipe FIFO queues
    pipes = build_pipe_queues(ops)

    # Sync index
    unmatched_flag, unmatched_block = build_sync_index(ops)
    total_unmatched = len(unmatched_flag) + len(unmatched_block)
    if total_unmatched:
        print(f"  Note: {total_unmatched} unmatched sync keys "
              f"({len(unmatched_flag)} flag, {len(unmatched_block)} block)")

    # --- Signal state ---
    # Intra-core: set_flag/wait_flag
    flag_signal_time = {}       # (flag_key, seq_idx) -> cycle
    flag_set_counter = defaultdict(int)
    flag_wait_counter = defaultdict(int)

    # Inter-core: sync_block_set/wait
    block_signal_time = {}      # (block_key, seq_idx) -> cycle
    block_set_counter = defaultdict(int)
    block_wait_counter = defaultdict(int)

    # --- Simulation state ---
    start_cycle = {}
    end_cycle = {}
    completed = set()

    def core_pipes_drain_at(core):
        """Return the cycle when all pipes of a core have drained."""
        t = 0
        for p in pipes.values():
            if p.name == "PIPE_UNKNOWN":
                continue
            if core == "AIC" and (p.name in AIC_PIPES or p.name == "PIPE_S_AIC"):
                t = max(t, p.drain_at)
            elif core == "AIV" and (p.name in AIV_PIPES or p.name == "PIPE_S_AIV"):
                t = max(t, p.drain_at)
            elif core == "SHARED":
                t = max(t, p.drain_at)
        return t

    def try_dispatch(pipe):
        """Try to dispatch the head op of this pipe.

        Returns True if dispatched, False if blocked.

        Key semantics:
        - Normal ops: start at issue_at (when pipe accepts the instruction)
        - set_flag: DRAINS the pipe first — waits for drain_at, then signals.
          "All prior instructions on set_pipe have completed."
        - pipe_barrier: DRAINS the target pipe — waits for drain_at.
        - pipe_barrier[PIPE_ALL]: DRAINS all core pipes.
        - wait_flag: blocks until matching set_flag has fired.
        """
        op = pipe.peek()
        if op is None:
            return False

        oid = op["id"]
        name = op["name"]
        earliest = pipe.issue_at

        # --- Cross-pipe SSA dependencies (e.g., scalar op -> hivm op) ---
        for dep_id in op.get("depends_on", []):
            if dep_id not in completed:
                return False  # dependency not yet completed
            earliest = max(earliest, end_cycle[dep_id])

        # --- set_flag drains the pipeline before signaling ---
        # "set_flag on set_pipe" means all prior set_pipe instructions have
        # COMPLETED execution.  The set_flag itself is issued after drain.
        if name == "set_flag":
            earliest = max(earliest, pipe.drain_at)

        # --- sync_block_set also drains (cross-core signal) ---
        elif name == "sync_block_set":
            earliest = max(earliest, pipe.drain_at)

        # --- pipe_barrier drains the target pipe ---
        elif name == "pipe_barrier" and op["is_barrier"]:
            if op["pipe"] == "PIPE_ALL":
                # Drain ALL pipes on this core
                core = op["_core"]
                earliest = max(earliest, core_pipes_drain_at(core))
            else:
                # Drain this specific pipe (implicit: we're on this pipe)
                earliest = max(earliest, pipe.drain_at)

        # --- Intra-core sync: wait_flag ---
        if name == "wait_flag" and op["event_id"]:
            key = flag_sync_key(op)
            if key in unmatched_flag:
                pass  # unresolvable — pass through
            else:
                idx = flag_wait_counter[key]
                signal_key = (key, idx)
                if signal_key in flag_signal_time:
                    earliest = max(earliest, flag_signal_time[signal_key])
                else:
                    return False  # blocked

        # --- Inter-core sync: sync_block_wait ---
        elif name == "sync_block_wait" and op["event_id"]:
            key = block_sync_key(op)
            if key in unmatched_block:
                pass
            else:
                idx = block_wait_counter[key]
                signal_key = (key, idx)
                if signal_key in block_signal_time:
                    earliest = max(earliest, block_signal_time[signal_key])
                else:
                    return False

        # --- Dispatch the op ---
        start = earliest
        dur = op["duration"]
        end = start + dur

        start_cycle[oid] = start
        end_cycle[oid] = end
        completed.add(oid)

        # Update pipe timing:
        # issue_at  = when the pipe can accept the NEXT instruction
        # drain_at  = when this instruction's execution COMPLETES
        #
        # For pipelined pipes (Cube): issue_at = start + 1 (accept new every cycle)
        # For serial pipes (DMA, etc): issue_at = end (can't accept until done)
        pipe.drain_at = max(pipe.drain_at, end)

        if pipe.name == "PIPE_M":
            # Cube engine is deeply pipelined — can accept new tile every cycle
            pipe.issue_at = start + max(1, dur)
        else:
            # All other pipes: serial (DMA, vector element processing, etc.)
            pipe.issue_at = end

        pipe.advance()

        # --- Signal production ---
        if name == "set_flag" and op["event_id"]:
            key = flag_sync_key(op)
            idx = flag_set_counter[key]
            flag_set_counter[key] += 1
            flag_signal_time[(key, idx)] = end

        elif name == "sync_block_set" and op["event_id"]:
            key = block_sync_key(op)
            idx = block_set_counter[key]
            block_set_counter[key] += 1
            block_signal_time[(key, idx)] = end

        # --- Signal consumption ---
        if name == "wait_flag" and op["event_id"]:
            key = flag_sync_key(op)
            if key not in unmatched_flag:
                flag_wait_counter[key] += 1

        elif name == "sync_block_wait" and op["event_id"]:
            key = block_sync_key(op)
            if key not in unmatched_block:
                block_wait_counter[key] += 1

        return True

    # --- Main simulation loop ---
    max_iterations = n * 100
    iteration = 0

    while any(p.has_pending() for p in pipes.values()):
        iteration += 1
        if iteration > max_iterations:
            pending = [(p.name, len(p.queue) - p.head)
                       for p in pipes.values() if p.has_pending()]
            print(f"  WARNING: exceeded {max_iterations} iterations. "
                  f"Pending: {pending}", file=sys.stderr)
            break

        # Keep dispatching until no pipe can make progress
        made_progress = True
        while made_progress:
            made_progress = False
            for pipe in pipes.values():
                if not pipe.has_pending():
                    continue
                if try_dispatch(pipe):
                    made_progress = True

        # Check if still blocked
        if not any(p.has_pending() for p in pipes.values()):
            break

        # Force-unblock the lowest-ID blocked op (deadlock from dynamic events)
        blocked_pipes = [p for p in pipes.values() if p.has_pending()]
        blocked_pipes.sort(key=lambda p: p.peek()["id"])
        bp = blocked_pipes[0]
        op = bp.peek()
        print(f"  Force-unblock: op {op['id']} ({op['name']} on {bp.name}, "
              f"event={op['event_id']})", file=sys.stderr)

        oid = op["id"]
        start = max(bp.issue_at, bp.drain_at)
        end = start + op["duration"]
        start_cycle[oid] = start
        end_cycle[oid] = end
        completed.add(oid)
        bp.issue_at = end
        bp.drain_at = max(bp.drain_at, end)
        bp.advance()

        # Still produce/consume signals for forced ops
        name = op["name"]
        if name == "set_flag" and op["event_id"]:
            key = flag_sync_key(op)
            idx = flag_set_counter[key]
            flag_set_counter[key] += 1
            flag_signal_time[(key, idx)] = end
        elif name == "sync_block_set" and op["event_id"]:
            key = block_sync_key(op)
            idx = block_set_counter[key]
            block_set_counter[key] += 1
            block_signal_time[(key, idx)] = end

        if name == "wait_flag" and op["event_id"]:
            key = flag_sync_key(op)
            if key not in unmatched_flag:
                flag_wait_counter[key] += 1
        elif name == "sync_block_wait" and op["event_id"]:
            key = block_sync_key(op)
            if key not in unmatched_block:
                block_wait_counter[key] += 1

    # Build results
    results = []
    for op in ops:
        oid = op["id"]
        results.append({
            "id": oid,
            "name": op["name"],
            "pipe": op["_pipe"],     # disambiguated
            "raw_pipe": op["pipe"],  # original
            "core": op["_core"],
            "start_cycle": start_cycle.get(oid, 0),
            "end_cycle": end_cycle.get(oid, 0),
            "duration": op["duration"],
            "line": op["line"],
            "bytes": op["bytes"],
            "elements": op["elements"],
            "event_id": op["event_id"],
            "event_generation": op["event_generation"],
            "sender_pipe": op["sender_pipe"],
            "receiver_pipe": op["receiver_pipe"],
            "core_type": op["core_type"],
            "is_sync": op["is_sync"],
            "is_barrier": op["is_barrier"],
            "loop_multiplier": op["loop_multiplier"],
            "read_buffers": op["read_buffers"],
            "write_buffers": op["write_buffers"],
            "read_versions": op["read_versions"],
            "write_versions": op["write_versions"],
        })
    return results


# ---------------------------------------------------------------------------
# Perfetto trace output
# ---------------------------------------------------------------------------

def emit_perfetto(results, clock_ghz, out_path):
    """Emit a Perfetto-compatible trace JSON.

    Two processes: AIC (pid=1) and AIV (pid=2).
    Each pipe is a thread within its core's process.
    Sync flow arrows connect set_flag->wait_flag and sync_block_set->wait.
    """
    # AIC pipes as pid=1, AIV pipes as pid=2
    aic_tids = {
        "PIPE_M": 1, "PIPE_MTE1": 2, "PIPE_MTE2_C": 3,
        "PIPE_FIX": 4, "PIPE_S_AIC": 5,
    }
    aiv_tids = {
        "PIPE_V": 1, "PIPE_MTE2_V": 2, "PIPE_MTE3": 3,
        "PIPE_S_AIV": 4,
    }
    misc_tids = {"PIPE_ALL": 1, "PIPE_UNKNOWN": 2}

    def pipe_pid_tid(pipe_name):
        if pipe_name in aic_tids:
            return 1, aic_tids[pipe_name]
        if pipe_name in aiv_tids:
            return 2, aiv_tids[pipe_name]
        return 3, misc_tids.get(pipe_name, 9)

    def cycles_to_us(cycles):
        return cycles / (clock_ghz * 1000)

    events = []

    # Process metadata
    events.append({"ph": "M", "pid": 1, "tid": 0,
                   "name": "process_name",
                   "args": {"name": "AIC (Cube Core)"}})
    events.append({"ph": "M", "pid": 2, "tid": 0,
                   "name": "process_name",
                   "args": {"name": "AIV (Vector Core)"}})
    events.append({"ph": "M", "pid": 3, "tid": 0,
                   "name": "process_name",
                   "args": {"name": "Shared"}})

    # Thread (pipe) metadata
    for pipe_name, tid in aic_tids.items():
        events.append({"ph": "M", "pid": 1, "tid": tid,
                       "name": "thread_name",
                       "args": {"name": pipe_name}})
    for pipe_name, tid in aiv_tids.items():
        events.append({"ph": "M", "pid": 2, "tid": tid,
                       "name": "thread_name",
                       "args": {"name": pipe_name}})
    for pipe_name, tid in misc_tids.items():
        events.append({"ph": "M", "pid": 3, "tid": tid,
                       "name": "thread_name",
                       "args": {"name": pipe_name}})

    # Op duration events
    for r in results:
        pid, tid = pipe_pid_tid(r["pipe"])
        ts = cycles_to_us(r["start_cycle"])
        dur = cycles_to_us(max(r["end_cycle"] - r["start_cycle"], 0))

        args = {
            "op_id": r["id"],
            "line": r["line"],
            "cycles": r["duration"],
            "start_cycle": r["start_cycle"],
            "end_cycle": r["end_cycle"],
            "core_type": r["core_type"],
            "core": r["core"],
        }
        if r["loop_multiplier"] > 1:
            args["loop_multiplier"] = r["loop_multiplier"]
        if r["bytes"]:
            args["bytes"] = r["bytes"]
        if r["elements"]:
            args["elements"] = r["elements"]
        if r["event_id"]:
            args["event_id"] = r["event_id"]
            args["sender_pipe"] = r["sender_pipe"]
            args["receiver_pipe"] = r["receiver_pipe"]
        if r["read_buffers"]:
            args["read_buffers"] = ";".join(r["read_buffers"])
        if r["write_buffers"]:
            args["write_buffers"] = ";".join(r["write_buffers"])

        events.append({
            "ph": "X", "pid": pid, "tid": tid,
            "ts": round(ts, 4), "dur": round(max(dur, 0.001), 4),
            "name": r["name"],
            "args": args,
        })

    # Flow arrows: set_flag -> wait_flag (intra-core)
    flow_id = 0
    flag_sets = defaultdict(list)
    flag_waits = defaultdict(list)
    block_sets_aic = defaultdict(list)
    block_sets_aiv = defaultdict(list)
    block_waits_aic = defaultdict(list)
    block_waits_aiv = defaultdict(list)

    for r in results:
        if not r["event_id"]:
            continue
        if r["name"] == "set_flag":
            key = (r["sender_pipe"], r["receiver_pipe"], r["event_id"])
            flag_sets[key].append(r)
        elif r["name"] == "wait_flag":
            key = (r["sender_pipe"], r["receiver_pipe"], r["event_id"])
            flag_waits[key].append(r)
        elif r["name"] == "sync_block_set":
            fid = r["event_id"]
            if r["core"] == "AIC":
                block_sets_aic[fid].append(r)
            else:
                block_sets_aiv[fid].append(r)
        elif r["name"] == "sync_block_wait":
            fid = r["event_id"]
            if r["core"] == "AIC":
                block_waits_aic[fid].append(r)
            else:
                block_waits_aiv[fid].append(r)

    # Intra-core flow arrows
    for key in flag_sets:
        for s, w in zip(flag_sets[key], flag_waits.get(key, [])):
            s_pid, s_tid = pipe_pid_tid(s["pipe"])
            w_pid, w_tid = pipe_pid_tid(w["pipe"])
            events.append({
                "ph": "s", "pid": s_pid, "tid": s_tid,
                "ts": round(cycles_to_us(s["end_cycle"]), 4),
                "id": flow_id, "name": "flag", "cat": "sync",
            })
            events.append({
                "ph": "f", "pid": w_pid, "tid": w_tid,
                "ts": round(cycles_to_us(w["start_cycle"]), 4),
                "id": flow_id, "name": "flag", "cat": "sync", "bp": "e",
            })
            flow_id += 1

    # Inter-core flow arrows: AIC set -> AIV wait
    for fid in block_sets_aic:
        for s, w in zip(block_sets_aic[fid], block_waits_aiv.get(fid, [])):
            s_pid, s_tid = pipe_pid_tid(s["pipe"])
            w_pid, w_tid = pipe_pid_tid(w["pipe"])
            events.append({
                "ph": "s", "pid": s_pid, "tid": s_tid,
                "ts": round(cycles_to_us(s["end_cycle"]), 4),
                "id": flow_id, "name": "block_sync", "cat": "cross_core",
            })
            events.append({
                "ph": "f", "pid": w_pid, "tid": w_tid,
                "ts": round(cycles_to_us(w["start_cycle"]), 4),
                "id": flow_id, "name": "block_sync", "cat": "cross_core",
                "bp": "e",
            })
            flow_id += 1

    # Inter-core flow arrows: AIV set -> AIC wait
    for fid in block_sets_aiv:
        for s, w in zip(block_sets_aiv[fid], block_waits_aic.get(fid, [])):
            s_pid, s_tid = pipe_pid_tid(s["pipe"])
            w_pid, w_tid = pipe_pid_tid(w["pipe"])
            events.append({
                "ph": "s", "pid": s_pid, "tid": s_tid,
                "ts": round(cycles_to_us(s["end_cycle"]), 4),
                "id": flow_id, "name": "block_sync", "cat": "cross_core",
            })
            events.append({
                "ph": "f", "pid": w_pid, "tid": w_tid,
                "ts": round(cycles_to_us(w["start_cycle"]), 4),
                "id": flow_id, "name": "block_sync", "cat": "cross_core",
                "bp": "e",
            })
            flow_id += 1

    trace = {"traceEvents": events, "displayTimeUnit": "us"}
    with open(out_path, "w") as f:
        json.dump(trace, f, indent=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HIVM pipe-based DES simulator (two-core model)")
    parser.add_argument("graph_json",
                        help="DES graph JSON from tritonsim-hivm --des-graph-file")
    parser.add_argument("-o", "--output", required=True,
                        help="Output Perfetto trace JSON path")
    args = parser.parse_args()

    with open(args.graph_json) as f:
        graph = json.load(f)

    clock_ghz = graph["clock_ghz"]
    ops = graph["operations"]
    print(f"Loaded {len(ops)} operations, clock={clock_ghz} GHz")

    # Show pipe distribution
    pipe_counts = defaultdict(int)
    core_counts = defaultdict(int)
    for op in ops:
        op["_pipe"] = disambiguate_pipe(op)
        op["_core"] = classify_core(op)
        pipe_counts[op["_pipe"]] += 1
        core_counts[op["_core"]] += 1

    print("\nCore distribution:")
    for core, count in sorted(core_counts.items()):
        print(f"  {core}: {count} ops")

    print("\nPipe queues:")
    for pipe, count in sorted(pipe_counts.items()):
        print(f"  {pipe}: {count} ops")

    results = simulate(ops, clock_ghz)

    # Summary
    max_cycle = max((r["end_cycle"] for r in results), default=0)
    print(f"\nSimulation complete: {max_cycle} cycles "
          f"({max_cycle / (clock_ghz * 1000):.3f} us)")

    # Per-core timing
    for core_name in ("AIC", "AIV"):
        core_ops = [r for r in results if r["core"] == core_name
                    and r["pipe"] not in ("PIPE_UNKNOWN", "PIPE_ALL")]
        if not core_ops:
            continue
        core_end = max(r["end_cycle"] for r in core_ops)
        core_start = min(r["start_cycle"] for r in core_ops)
        core_busy = sum(r["duration"] for r in core_ops)
        print(f"\n  {core_name}: span={core_end - core_start} cycles, "
              f"busy={core_busy} cycles")

        # Per-pipe within core
        pipe_data = defaultdict(lambda: {"busy": 0, "start": float("inf"), "end": 0})
        for r in core_ops:
            p = pipe_data[r["pipe"]]
            p["busy"] += r["duration"]
            p["start"] = min(p["start"], r["start_cycle"])
            p["end"] = max(p["end"], r["end_cycle"])
        for pname in sorted(pipe_data):
            p = pipe_data[pname]
            span = p["end"] - p["start"]
            util = p["busy"] / span * 100 if span else 0
            print(f"    {pname}: {p['busy']} busy / {span} span ({util:.1f}%)")

    emit_perfetto(results, clock_ghz, args.output)
    print(f"\nTrace written to {args.output}")


if __name__ == "__main__":
    main()
