"""DES Trace Postprocessor — apply per-opcode v3 cycle costs to DES JSON output.

Replaces the default duration=1 in DES output with real measured cycle costs
from CCE microbenchmarks and profiling data.

Usage:
    from perfbound.calibration.des_trace_postprocessor import postprocess_des
    postprocess_des("chunk_des_20260624.json")

Requires: perfbound/calibration/data/calib_910b3_v3_opcode.json
"""
import json, os
from pathlib import Path

ROOT = os.environ.get("VTRITON_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CLOCK = 1.85

def load_v3():
    v3_path = os.path.join(ROOT, "perfbound", "calibration", "data", "calib_910b3_v3_opcode.json")
    return json.load(open(v3_path))

_V3 = None

def _get_v3():
    global _V3
    if _V3 is None:
        _V3 = load_v3()
    return _V3

def get_cycles(name, pipe, elements=0, bytes_val=0):
    """Lookup per-opcode cycle cost from v3 calibration table."""
    v3 = _get_v3()
    opc = v3["opcode_cycles"]; pv = opc["PIPE_V"]; ps = opc["PIPE_S"]; pm = opc.get("PIPE_M", {}); mte = opc.get("MTE", {})

    if pipe == "PIPE_V":
        if name in pv: return pv[name]["cycles"]
        return pv.get("_default", {}).get("cycles", 5.0)

    if pipe == "PIPE_S":
        for sub in ["_scalar_alu", "_agu", "_sync", "_misc"]:
            d = ps.get(sub, {})
            if name in d: return d[name]["cycles"]
        return 3.1

    if pipe == "PIPE_M":
        return pm.get(name, {}).get("cycles", 1)

    if pipe in mte:
        m = mte[pipe]
        return max(1, int(round(m["startup_cycles"] + bytes_val * m.get("cycles_per_byte", 0)))) if bytes_val > 0 else m["startup_cycles"]

    fb = {"PIPE_ALL": 64, "PIPE_UNKNOWN": 1, "PIPE_MTE1": 10, "PIPE_FIX": 30}
    return fb.get(pipe, 1)

def postprocess_des(des_path, out_path=None):
    """Post-process DES JSON: replace duration=1 with v3 per-opcode cycles."""
    d = json.load(open(des_path))
    total = 0; per_pipe = {}
    for o in d["operations"]:
        n, p = o.get("name","?"), o.get("pipe","")
        c = int(round(get_cycles(n, p, int(o.get("elements",0)), int(o.get("bytes",0)))))
        o["duration"] = c; o["_real_cycles"] = c
        total += c; per_pipe[p] = per_pipe.get(p, 0) + c
    d["_postprocessed_v3"] = True
    d["_total_cycles"] = total
    d["_total_us"] = round(total / (CLOCK * 1000), 2)
    d["_per_pipe_cycles"] = dict(sorted(per_pipe.items(), key=lambda x: -x[1]))
    if out_path is None:
        out_path = des_path.replace(".json", "_v3.json")
    json.dump(d, open(out_path, "w"))
    return d
