#!/bin/bash
# =============================================================================
# run_calibration.sh — Step 3: v3 per-opcode calibration
# =============================================================================
# Uses v3 DES JSON (post-processed with per-opcode cycle costs) +
# msprof real profiling → E2E comparison + report.
# =============================================================================
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATE=$(date +%Y%m%d)
OUT_DIR="$ROOT_DIR/scripts/e2e_chunk/output"
MSPROF_CSV="${1:-$OUT_DIR/op_summary.csv}"

# ── Find latest v3 DES ─────────────────────────────────────────────
DES_V3=$(ls -t "$OUT_DIR/chunk_des_"*_v3.json 2>/dev/null | head -1)
if [ -z "$DES_V3" ]; then
    echo "[ERROR] No v3 DES found. Run: python3 temp/apply_v3.py"
    exit 1
fi

REPORT_TXT="$OUT_DIR/calib_report_v3_${DATE}.txt"
CALIB_JSON="$OUT_DIR/calib_910b3_v3.json"

echo "============================================================"
echo "  v3 CALIBRATION: chunk_kda"
echo "  DES v3 : $DES_V3"
echo "  msprof : $MSPROF_CSV"
echo "============================================================"

python3 - "$MSPROF_CSV" "$DES_V3" "$REPORT_TXT" "$CALIB_JSON" "$ROOT_DIR/perfbound/calibration/data/calib_910b3_v3_opcode.json" "$DATE" << 'PYEOF'
import json, csv, math, sys
from collections import defaultdict, Counter

msprof_csv = sys.argv[1]
des_v3     = sys.argv[2]
report_txt = sys.argv[3]
calib_json = sys.argv[4]
v3_opcode  = sys.argv[5]
report_date = sys.argv[6] if len(sys.argv) > 6 else "unknown"

CLOCK = 1.85; AIC = 20; AIV = 40
v3 = json.load(open(v3_opcode))
OPC = v3["opcode_cycles"]; PV = OPC["PIPE_V"]; PS = OPC["PIPE_S"]; MTE = OPC["MTE"]

# ── 1. Real data ──────────────────────────────────────────────────────
rows = list(csv.DictReader(open(msprof_csv)))
ck = [r for r in rows if "chunk_kda" in r.get("Op Name","")]
if not ck: print("[ERROR] chunk_kda not found"); sys.exit(1)
r = ck[0]
REAL_WALL = float(r["Task Duration(us)"])
BLOCKS = int(r.get("Block Dim", 4096))
TT = r.get("Task Type","").strip()
W_AIC = math.ceil(BLOCKS / AIC)
W_AIV = math.ceil(BLOCKS / AIV)
print(f"  Real E2E: {REAL_WALL:,.0f} us  blocks={BLOCKS}  type={TT}")
print(f"  Waves: AIC={W_AIC} AIV={W_AIV}")

# ── 2. Model (v3 DES, already has per-opcode cycles) ──────────────────
des = json.load(open(des_v3))
ops = des["operations"]

per_pipe = defaultdict(float)
per_opcode = defaultdict(float)
for o in ops:
    n = o.get("name","?"); p = o.get("pipe","")
    dur = float(o.get("duration", 1))
    per_pipe[p] += dur
    per_opcode[(p, n)] += dur

aic_cyc = sum(c for p, c in per_pipe.items() if p in ("PIPE_M","PIPE_MTE1","PIPE_MTE2_C","PIPE_FIX","PIPE_ALL"))
aiv_cyc = sum(c for p, c in per_pipe.items() if p in ("PIPE_V","PIPE_MTE2_V","PIPE_MTE3","PIPE_S"))
aic_us = aic_cyc / (CLOCK * 1000)
aiv_us = aiv_cyc / (CLOCK * 1000)
model_e2e = max(aic_us * W_AIC, aiv_us * W_AIV)

ratio = model_e2e / REAL_WALL if REAL_WALL > 0 else 0
print(f"  Model per-block: AIC={aic_us:.1f}us  AIV={aiv_us:.1f}us")
print(f"  Model E2E:   {model_e2e:,.0f} us")
print(f"  Model/Real:  {ratio:.4f} ({ratio*100:.1f}%)")

# ── 3. Per-pipe breakdown ─────────────────────────────────────────────
lines = []
lines.append(f"chunk_kda v3 Calibration Report ({report_date})")
lines.append("=" * 60)
lines.append(f"  Real E2E:   {REAL_WALL:,.0f} us  blocks={BLOCKS}  type={TT}")
lines.append(f"  Model E2E:  {model_e2e:,.0f} us  (AIC={aic_us*W_AIC:,.0f}us  AIV={aiv_us*W_AIV:,.0f}us)")
lines.append(f"  Ratio:      {ratio:.4f}")
lines.append("")

lines.append("Per-pipe (per-block):")
lines.append(f"  {'Pipe':16s} {'cycles':>10s} {'us':>8s} {'%':>6s}")
for p, c in sorted(per_pipe.items(), key=lambda x: -x[1])[:10]:
    us = c / (CLOCK * 1000)
    lines.append(f"  {p:16s} {c:10,.0f} {us:8.2f} {c/sum(per_pipe.values())*100:5.1f}%")

lines.append("")
lines.append("Per-pipe model vs real (per-block):")
lines.append(f"  {'Pipe':16s} {'model_us':>8s} {'real_us':>8s} {'ratio':>8s}")
msprof_pipes = {"PIPE_V":("aiv_vec_time(us)",W_AIV), "PIPE_S":("aiv_scalar_time(us)",W_AIV),
                "PIPE_MTE3":("aiv_mte3_time(us)",W_AIV), "PIPE_MTE2_V":("aiv_mte2_time(us)",W_AIV),
                "PIPE_M":("aic_mac_time(us)",W_AIC)}
for pipe, (col, w) in msprof_pipes.items():
    m_c = per_pipe.get(pipe, 0)
    m_us = m_c / (CLOCK * 1000)
    rv = float(r.get(col, 0))
    r_us = rv / w
    rp = m_us / r_us if r_us > 0 else 0
    lines.append(f"  {pipe:16s} {m_us:8.1f} {r_us:8.1f} {rp:8.4f}")

lines.append("")
lines.append("Top PIPE_V opcodes (v3 cycles):")
for (p, n), c in sorted(per_opcode.items(), key=lambda x: -x[1]):
    if p == "PIPE_V": lines.append(f"  {n:20s}: {c:10,.0f} cyc")

with open(report_txt, 'w') as f: f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))

# ── 4. Save v3 calibration JSON ────────────────────────────────────────
calib = {
    "version": "v3",
    "kernel": "chunk_kda_bwd_kernel",
    "real_e2e_us": REAL_WALL, "model_e2e_us": round(model_e2e, 1),
    "ratio": round(ratio, 4),
    "aic_per_block_us": round(aic_us, 2), "aiv_per_block_us": round(aiv_us, 2),
    "waves_aic": W_AIC, "waves_aiv": W_AIV,
    "per_pipe_cycles": {p: int(c) for p, c in per_pipe.items()},
    "opcode_table_used": v3_opcode,
}
json.dump(calib, open(calib_json, 'w'), indent=2)
print(f"\n  Saved: {calib_json}")
PYEOF

echo ""
echo "[DONE] v3 calibration"
