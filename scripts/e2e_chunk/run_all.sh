#!/bin/bash
# =============================================================================
# run_all.sh  —  Full E2E pipeline: Profile → Dump MLIR → DES+Trace → Calibration
#
# Step 0 (triton_profiling)   — profiling with msprof (skipped if op_summary.csv exists)
# Step 1 (triton_dev, CANN 9) — dump HIVM MLIR from kernel source
# Step 2 (triton_dev, CANN 9) — DES simulation + Perfetto trace
# Step 3 (triton_dev, CANN 9) — calibration: model vs real
#
# NOTE: All steps except profiling must run in triton_dev (CANN 9).
#       Profiling requires triton_profiling (CANN 8.5) due to host driver 25.5.1.
# =============================================================================
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/scripts/e2e_chunk"
sed -i 's/\r//' *.sh 2>/dev/null || true
chmod +x *.sh
rm -rf dump

echo ""
echo "=== STEP 0 (profile on NPU) ==="
if [ -f output/op_summary.csv ]; then
    echo "[SKIP] output/op_summary.csv already exists"
else
    bash run_profile.sh
fi

echo ""
echo "=== STEP 1 (dump MLIR) ==="
bash run_dump_mlir.sh
echo ""
echo "=== STEP 2 (DES + trace) ==="
bash run_des_trace.sh dump/kernel.npuir.mlir 2>&1 | tail -8
echo ""
echo "=== STEP 3 (calibration) ==="
bash run_calibration.sh 2>&1 | tail -15
