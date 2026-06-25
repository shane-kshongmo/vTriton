#!/bin/bash
# =============================================================================
# run_dump_mlir.sh   —   Step 1: Prepare HIVM MLIR for DES simulation
# =============================================================================
# Environment   : CANN 9.0 container (triton_dev)
# Kernel        : chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2
# Output        : scripts/e2e_chunk/dump/kernel.npuir.mlir
#
# Strategy (3 tiers):
#   Tier 1 — Fresh dump via triton_dsl_dump_launcher (requires
#            compile_only_mock.py in thirdparty/ path).
#   Tier 2 — Direct bishengir-compile on .ttadapter.mlir (CANN 8.5
#            only; CANN 9 ttadapter has linalg ops bishengir can't read).
#   Tier 3 — Fall back to output/kernel.npuir.mlir (pre-captured from
#            a real CANN 9 NPU run with msprof).
#
# Usage:
#   chmod +x scripts/e2e_chunk/run_dump_mlir.sh
#   cd /home/triton_sim/vTriton && bash scripts/e2e_chunk/run_dump_mlir.sh
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="$ROOT_DIR/scripts/e2e_chunk/dump"
FALLBACK_MLIR="$ROOT_DIR/output/kernel.npuir.mlir"
CACHE_BASE="/root/.triton/cache"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "  STEP 1: PREPARE HIVM MLIR"
echo "============================================================"
echo "  Kernel: chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2"
echo "  Target: CANN 9.0 (Ascend910_9362)"
echo ""

# ── CANN environment ────────────────────────────────────────────────────────
ASCEND_TOOLKIT="/usr/local/Ascend/ascend-toolkit"
[ -f "$ASCEND_TOOLKIT/set_env.sh" ] && . "$ASCEND_TOOLKIT/set_env.sh"

BISHENGIR_COMPILE=""
for cand in \
    /usr/local/Ascend/cann-9.0.0/aarch64-linux/bin/bishengir-compile \
    /usr/local/Ascend/cann/bin/bishengir-compile \
    ; do
    [ -x "$cand" ] && { BISHENGIR_COMPILE="$cand"; break; }
done

export TRITON_COMPILE_ONLY=1
export TRITON_ENABLE_TASKQUEUE=0
export TRITON_ASCEND_ARCH=Ascend910_9362

# ── Tier 1: launcher hook capture ───────────────────────────────────────────
echo "[Tier 1] triton_dsl_dump_launcher (capture via subprocess hook) ..."
rm -rf "$CACHE_BASE"/*

python3 tools/common/triton_dsl_dump_launcher.py \
    --script test/chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2.py \
    --dump-dir "$OUT_DIR" \
    2>&1 | tee "$OUT_DIR/tier1_launcher.log" || true

if compgen -G "$OUT_DIR/kernel_*.npuir.mlir" > /dev/null 2>&1; then
    NPUIR=$(ls "$OUT_DIR"/kernel_*.npuir.mlir 2>/dev/null | head -1)
    cp "$NPUIR" "$OUT_DIR/kernel.npuir.mlir"
    echo "[OK] Tier 1 succeeded: $OUT_DIR/kernel.npuir.mlir ($(wc -l < "$OUT_DIR/kernel.npuir.mlir") lines)"
    echo "[DONE] → Next: bash scripts/e2e_chunk/run_des_trace.sh"
    exit 0
fi
echo "[INFO] Tier 1: no NPUIR captured (compile_only_mock hook not triggered)"

# ── Tier 2: direct bishengir-compile (CANN 8.5 only) ────────────────────────
CACHE_DIR=$(find "$CACHE_BASE" -name '*.ttadapter' 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)

if [ -n "${CACHE_DIR:-}" ] && [ -n "$BISHENGIR_COMPILE" ]; then
    TTADAPTER=$(ls "$CACHE_DIR"/*.ttadapter 2>/dev/null | head -1)
    echo ""
    echo "[Tier 2] Direct bishengir-compile on $TTADAPTER ..."

    BISHENGIR_OUT="$OUT_DIR/tier2_bishengir_output.txt"
    "$BISHENGIR_COMPILE" \
        --bishengir-print-ir-after=hivm-graph-sync-solver \
        "$TTADAPTER" -o /dev/null \
        > "$BISHENGIR_OUT" 2>&1 || true

    # Try to extract HIVM MLIR
    python3 - "$BISHENGIR_OUT" "$OUT_DIR/kernel.npuir.mlir" << 'PYEOF'
import sys, re
text = open(sys.argv[1], encoding='utf-8', errors='replace').read()
out  = sys.argv[2]
START, END = "//-----// IR Dump After", "//-----// IR Dump Before"
modules = []; lines = text.split('\n'); i = 0
while i < len(lines):
    if START not in lines[i]: i += 1; continue
    i += 1; start_i = i
    while i < len(lines) and not (END in lines[i] and i > start_i + 1): i += 1
    block = lines[start_i:i]
    if any('func.func' in l for l in block):
        while block and not block[0].strip(): block.pop(0)
        while block and not block[-1].strip(): block.pop()
        modules.append('\n'.join(block))
    i += 1
if modules:
    with open(out, 'w') as f: f.write(modules[-1] + '\n')
    print(f"  [OK] Extracted {len(modules[-1].splitlines())} lines")
else:
    print("  [FAIL] No func.func in bishengir output (CANN 9 .ttadapter has linalg ops)")
PYEOF
    if [ -s "$OUT_DIR/kernel.npuir.mlir" ]; then
        echo "[OK] Tier 2 succeeded: $OUT_DIR/kernel.npuir.mlir ($(wc -l < "$OUT_DIR/kernel.npuir.mlir") lines)"
        echo "[DONE] -> Next: bash scripts/e2e_chunk/run_des_trace.sh"
        exit 0
    fi
fi

# ── Tier 3: fallback ────────────────────────────────────────────────────────
echo ""
echo "[Tier 3] Fallback to pre-captured MLIR ..."
if [ -f "$FALLBACK_MLIR" ]; then
    cp "$FALLBACK_MLIR" "$OUT_DIR/kernel.npuir.mlir"
    echo "  Source: $FALLBACK_MLIR  ($(wc -l < "$FALLBACK_MLIR") lines)"
    echo ""
    echo "  NOTE: This MLIR was captured from a prior real NPU run with msprof."
    echo "        To regenerate from source on CANN 9, run:"
    echo "          msprof --aic-metrics=PipeUtilization --ai-core=on \\"
    echo "            python3 test/chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2.py"
elif [ -f "$HOME/vTriton/output/kernel.npuir.mlir" ]; then
    cp "$HOME/vTriton/output/kernel.npuir.mlir" "$OUT_DIR/kernel.npuir.mlir"
    echo "  Source: \$HOME/vTriton/output/kernel.npuir.mlir"
else
    echo "  [ERROR] No kernel.npuir.mlir found."
    exit 1
fi

echo ""
echo "============================================================"
echo "  OUTPUT"
echo "============================================================"
echo "  dump/kernel.npuir.mlir   ($(wc -l < "$OUT_DIR/kernel.npuir.mlir") lines)"
echo "============================================================"
echo "[DONE] → Next: bash scripts/e2e_chunk/run_des_trace.sh"
