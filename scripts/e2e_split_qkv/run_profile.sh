#!/bin/bash
# =============================================================================
# run_profile.sh  —  Step 0: Run split_qkv on real NPU + capture msprof data
# =============================================================================
# Environment   : triton_profiling (CANN 8.5) — required for profiling
#                 triton_dev (CANN 9) can run all other steps but profiling
#                 requires CANN 8.5 (host driver 25.5.1 compatibility)
# Kernel        : split_qkv_rmsnorm_mrope
# Output        : scripts/e2e_split_qkv/output/op_summary.csv
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/scripts/e2e_split_qkv/output"
MSPROF_DIR="$ROOT/scripts/e2e_split_qkv/msprof_data"
mkdir -p "$OUT"
rm -rf "$MSPROF_DIR"
mkdir -p "$MSPROF_DIR"

echo "=== Step 0: Profile split_qkv on NPU (triton_profiling / CANN 8.5) ==="

unset TRITON_COMPILE_ONLY
export TRITON_ENABLE_TASKQUEUE=0
export TRITON_ASCEND_ARCH=Ascend910_9362
rm -rf /root/.triton/cache/*

msprof --output="$MSPROF_DIR" \
    --aic-metrics=PipeUtilization \
    --ai-core=on \
    --aic-mode=task-based \
    --task-time=on \
    python3 test/split_qkv_rmsnorm_mrope_standalone.py \
    2>&1 | tee "$OUT/profile_log.txt"

OP_SUMMARY=$(find "$MSPROF_DIR" -name 'op_summary*.csv' 2>/dev/null | head -1)
if [ -z "$OP_SUMMARY" ]; then
    echo "[FAIL] No op_summary.csv generated."
    find "$MSPROF_DIR" -mindepth 2 -type f 2>/dev/null | head -20
    echo "--- kernel/output log ---"
    grep -i 'error\|launch\|kernel\|fail' "$OUT/profile_log.txt" 2>/dev/null | head -10
    exit 1
fi
cp "$OP_SUMMARY" "$OUT/op_summary.csv"
grep split_qkv "$OUT/op_summary.csv" | head -3
echo "[OK] $(grep -c split_qkv "$OUT/op_summary.csv") entries"
echo "[DONE]"
