#!/bin/bash
# =============================================================================
# run_calibration.sh — Step 3: v3 per-opcode calibration (split_qkv)
# =============================================================================
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATE=$(date +%Y%m%d)
OUT_DIR="$ROOT_DIR/scripts/e2e_split_qkv/output"
MSPROF_CSV="${1:-$OUT_DIR/op_summary.csv}"

DES_V3=$(ls -t "$OUT_DIR/split_qkv_des_"*_v3.json 2>/dev/null | head -1)
if [ -z "$DES_V3" ]; then
    # Fallback: try old naming
    DES_V3=$(ls -t "$OUT_DIR/chunk_des_"*_v3.json 2>/dev/null | head -1)
fi
if [ -z "$DES_V3" ]; then
    echo "[ERROR] No v3 DES found. Run: python3 temp/apply_v3.py"
    exit 1
fi

REPORT_TXT="$OUT_DIR/calib_report_v3_${DATE}.txt"
CALIB_JSON="$OUT_DIR/calib_910b3_v3.json"

echo "============================================================"
echo "  v3 CALIBRATION: split_qkv"
echo "  DES v3 : $DES_V3"
echo "  msprof : $MSPROF_CSV"
echo "============================================================"

python3 - "$MSPROF_CSV" "$DES_V3" "$REPORT_TXT" "$CALIB_JSON" "$ROOT_DIR/perfbound/calibration/data/calib_910b3_v3_opcode.json" "$DATE" << 'PYEOF'
import json, csv, math, sys
from collections import defaultdict, Counter

msprof_csv  = sys.argv[1]
des_v3      = sys.argv[2]
report_txt  = sys.argv[3]
calib_json  = sys.argv[4]
v3_opcode   = sys.argv[5]
report_date = sys.argv[6] if len(sys.argv) > 6 else "unknown"

CLOCK = 1.85; AIV = 40; TPB = 103  # tokens per block

# ── 1. Real data ──────────────────────────────────────────────────────
rows = list(csv.DictReader(open(msprof_csv)))
sq = [r for r in rows if "split_qkv" in r.get("Op Name","")]
if not sq: print("[ERROR] split_qkv not found"); sys.exit(1)
r = sq[0]
REAL_WALL = float(r["Task Duration(us)"])
BLOCKS = int(r.get("Block Dim", 20))
W_AIV = math.ceil(BLOCKS / AIV)
print(f"  Real E2E: {REAL_WALL:,.1f} us  blocks={BLOCKS}  waves=1  tokens/block={TPB}")

# ── 2. Model (v3 DES per-token, scale to per-block then E2E) ───────────
des = json.load(open(des_v3))
ops = des["operations"]

per_pipe = defaultdict(float)
for o in ops:
    per_pipe[o.get("pipe","?")] += float(o.get("duration", 1)) * TPB  # scale to per-block

aiv_cyc = sum(c for p, c in per_pipe.items() if p in ("PIPE_V","PIPE_MTE2_V","PIPE_MTE3","PIPE_S","PIPE_ALL","PIPE_UNKNOWN"))
aiv_us = aiv_cyc / (CLOCK * 1000)
model_e2e = aiv_us * W_AIV

ratio = model_e2e / REAL_WALL if REAL_WALL > 0 else 0
print(f"  Model per-block: AIV={aiv_us:.1f}us")
print(f"  Model E2E:   {model_e2e:,.0f} us")
print(f"  Model/Real:  {ratio:.4f} ({ratio*100:.1f}%)")

# ── 3. Report ──────────────────────────────────────────────────────────
lines = []
lines.append(f"split_qkv v3 Calibration Report ({report_date})")
lines.append("=" * 60)
lines.append(f"  Real E2E:   {REAL_WALL:,.1f} us  blocks={BLOCKS}  type=AIV-only")
lines.append(f"  Model E2E:  {model_e2e:,.0f} us  (per-block={aiv_us:.1f}us × {W_AIV} waves)")
lines.append(f"  Ratio:      {ratio:.4f}")
lines.append("")

lines.append("Per-pipe (per-block, 103 tokens):")
lines.append(f"  {'Pipe':16s} {'cycles':>10s} {'us':>8s} {'%':>6s}")
total = sum(per_pipe.values())
for p, c in sorted(per_pipe.items(), key=lambda x: -x[1])[:10]:
    us = c / (CLOCK * 1000)
    lines.append(f"  {p:16s} {c:10,.0f} {us:8.2f} {c/total*100:5.1f}%")

with open(report_txt, 'w') as f: f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))

# ── 4. Save JSON ───────────────────────────────────────────────────────
calib = {
    "version": "v3", "kernel": "split_qkv_rmsnorm_mrope",
    "real_e2e_us": REAL_WALL, "model_e2e_us": round(model_e2e, 1),
    "ratio": round(ratio, 4),
    "aiv_per_block_us": round(aiv_us, 2), "waves_aiv": W_AIV,
    "tokens_per_block": TPB,
    "per_pipe_cycles": {p: int(c) for p, c in per_pipe.items()},
}
json.dump(calib, open(calib_json, 'w'), indent=2)
print(f"\n  Saved: {calib_json}")
PYEOF

echo ""
echo "[DONE] v3 calibration"
