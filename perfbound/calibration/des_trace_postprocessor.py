"""DES Trace Postprocessor — apply per-opcode v3 cycle costs to DES JSON output.

Replaces the default duration=1 in DES output with real measured cycle costs
from CCE microbenchmarks and profiling data.

Usage:
    from perfbound.calibration.des_trace_postprocessor import postprocess_des
    postprocess_des("chunk_des_20260624.json")

Requires: perfbound/calibration/data/calib_910b3_v3_opcode.json
"""
import json
import os

ROOT = os.environ.get(
    "VTRITON_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
CLOCK = 1.85  # GHz; cycles -> us via cycles / (CLOCK * 1000)


def load_v3():
    v3_path = os.path.join(
        ROOT, "perfbound", "calibration", "data", "calib_910b3_v3_opcode.json"
    )
    with open(v3_path) as f:
        return json.load(f)


_V3 = None


def _get_v3():
    global _V3
    if _V3 is None:
        _V3 = load_v3()
    return _V3


def get_cycles(name, pipe, elements=0, bytes_val=0):
    """Lookup per-opcode cycle cost from the v3 calibration table."""
    opc = _get_v3().get("opcode_cycles", {})
    pv = opc.get("PIPE_V", {})
    ps = opc.get("PIPE_S", {})
    pm = opc.get("PIPE_M", {})
    mte = opc.get("MTE", {})

    if pipe == "PIPE_V":
        entry = pv.get(name) or pv.get("_default", {})
        return entry.get("cycles", 5.0)

    if pipe == "PIPE_S":
        for sub in ("_scalar_alu", "_agu", "_sync", "_misc"):
            entry = ps.get(sub, {}).get(name)
            if entry is not None:
                return entry.get("cycles", 3.1)
        return 3.1

    if pipe == "PIPE_M":
        return pm.get(name, {}).get("cycles", 1)

    if pipe in mte:
        m = mte[pipe]
        if bytes_val > 0:
            cost = m.get("startup_cycles", 1) + bytes_val * m.get("cycles_per_byte", 0)
            return max(1, int(round(cost)))
        return m.get("startup_cycles", 1)

    fb = {"PIPE_ALL": 64, "PIPE_UNKNOWN": 1, "PIPE_MTE1": 10, "PIPE_FIX": 30}
    return fb.get(pipe, 1)


def postprocess_des(des_path, out_path=None):
    """Post-process DES JSON: replace duration=1 with v3 per-opcode cycles."""
    with open(des_path) as f:
        d = json.load(f)

    total = 0
    per_pipe = {}
    for o in d.get("operations", []):
        name = o.get("name", "?")
        pipe = o.get("pipe", "")
        cycles = int(round(get_cycles(
            name, pipe, int(o.get("elements", 0)), int(o.get("bytes", 0)))))
        o["duration"] = cycles
        o["_real_cycles"] = cycles
        total += cycles
        per_pipe[pipe] = per_pipe.get(pipe, 0) + cycles

    d["_postprocessed_v3"] = True
    d["_total_cycles"] = total
    d["_total_us"] = round(total / (CLOCK * 1000), 2)
    d["_per_pipe_cycles"] = dict(sorted(per_pipe.items(), key=lambda kv: -kv[1]))

    if out_path is None:
        out_path = des_path.replace(".json", "_v3.json")
    with open(out_path, "w") as f:
        json.dump(d, f)
    return d
