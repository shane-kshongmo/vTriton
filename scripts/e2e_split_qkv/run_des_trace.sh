#!/bin/bash
# =============================================================================
# run_des_trace.sh  —  Step 2: DES + Trace for split_qkv kernel
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATE=$(date +%Y%m%d)
OUT="$ROOT/scripts/e2e_split_qkv/output"
mkdir -p "$OUT"
CLEAN="$OUT/kernel_clean_$DATE.npuir.mlir"
echo "=== Step 2: DES + Trace — split_qkv ==="
echo "Input: $ROOT/scripts/e2e_split_qkv/dump/kernel.npuir.mlir ($(wc -l < $ROOT/scripts/e2e_split_qkv/dump/kernel.npuir.mlir) lines)"
python3 "$ROOT/scripts/clean_npuir.py" "$ROOT/scripts/e2e_split_qkv/dump/kernel.npuir.mlir" "$CLEAN" || {
    cp "$ROOT/scripts/e2e_split_qkv/dump/kernel.npuir.mlir" "$CLEAN"
    echo "[WARN] clean_npuir failed, using raw MLIR"
}
echo "Cleaned: $(wc -l < $CLEAN) lines"
"$ROOT/build/bin/tritonsim-opt" --allow-unregistered-dialect \
    "--analyze-hivm=scheduler=des des-graph-file=$OUT/split_qkv_des_$DATE.json perfetto-trace-file=$OUT/split_qkv_trace_$DATE.json" \
    "$CLEAN" 2>&1
if [ -f "$OUT/split_qkv_des_$DATE.json" ] && [ -f "$OUT/split_qkv_trace_$DATE.json" ]; then
    echo "[OK] DES: $(du -h $OUT/split_qkv_des_$DATE.json | cut -f1)"
    echo "[OK] Trace: $(du -h $OUT/split_qkv_trace_$DATE.json | cut -f1)"
else
    echo "[FAIL]"
    exit 1
fi
echo "[DONE]"
