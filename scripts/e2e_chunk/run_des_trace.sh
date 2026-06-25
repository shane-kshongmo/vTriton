#!/bin/bash
# =============================================================================
# run_des_trace.sh  —  Step 2: Generate DES graph + Perfetto trace from MLIR
# =============================================================================
# Environment   : CANN 9.0 container (recommended: triton_dev)
# Input         : scripts/e2e_chunk/dump/kernel.npuir.mlir
#                 (run_dump_mlir.sh must be executed first)
# Output        : scripts/e2e_chunk/output/chunk_des_<date>.json
#                 scripts/e2e_chunk/output/chunk_trace_<date>.json
#
# Also accepts a direct MLIR file as optional argument:
#   ./scripts/e2e_chunk/run_des_trace.sh  [path/to/kernel.npuir.mlir]
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

# ── Determine input MLIR ────────────────────────────────────────────────────
if [ $# -ge 1 ]; then
    INPUT_MLIR="$1"
    case "$INPUT_MLIR" in
        /*) ;;
        *)  INPUT_MLIR="$ROOT_DIR/$INPUT_MLIR" ;;
    esac
else
    INPUT_MLIR="$ROOT_DIR/scripts/e2e_chunk/dump/kernel.npuir.mlir"
fi

if [ ! -f "$INPUT_MLIR" ]; then
    echo "[ERROR] Input MLIR not found: $INPUT_MLIR"
    echo "  Run run_dump_mlir.sh first, or pass explicit path as argument."
    exit 1
fi

echo "[INPUT] $INPUT_MLIR  ($(wc -l < "$INPUT_MLIR") lines)"

# ── Output directory & filenames ─────────────────────────────────────────────
DATE=$(date +%Y%m%d)
OUT_DIR="$ROOT_DIR/scripts/e2e_chunk/output"
mkdir -p "$OUT_DIR"

CLEAN_MLIR="$OUT_DIR/kernel_clean_${DATE}.npuir.mlir"
DES_OUT="$OUT_DIR/chunk_des_${DATE}.json"
TRACE_OUT="$OUT_DIR/chunk_trace_${DATE}.json"

# ── Clean the MLIR (strip bishengir-specific syntax) ─────────────────────────
echo ""
echo "[STEP 1] Cleaning MLIR ..."
python3 "$ROOT_DIR/scripts/clean_npuir.py" "$INPUT_MLIR" "$CLEAN_MLIR"
echo "  Cleaned MLIR: $CLEAN_MLIR  ($(wc -l < "$CLEAN_MLIR") lines)"

# ── Check tritonsim-opt ─────────────────────────────────────────────────────
TRITONSIM_OPT="$ROOT_DIR/build/bin/tritonsim-opt"
if [ ! -x "$TRITONSIM_OPT" ]; then
    echo "[ERROR] tritonsim-opt not found or not executable: $TRITONSIM_OPT"
    echo "  Please build the project first:  cd build && cmake --build ."
    exit 1
fi
echo "[CHECK] tritonsim-opt  : $TRITONSIM_OPT ($(du -h "$TRITONSIM_OPT" | cut -f1))"

# ── Run tritonsim-opt (DES scheduler) ───────────────────────────────────────
echo ""
echo "============================================================"
echo "  GENERATE DES + PERFETTO TRACE"
echo "============================================================"
echo "  Input  : $CLEAN_MLIR"
echo "  DES    : $DES_OUT"
echo "  Trace  : $TRACE_OUT"
echo "============================================================"
echo ""

"$TRITONSIM_OPT" \
    --allow-unregistered-dialect \
    "--analyze-hivm=scheduler=des \
      des-graph-file=$DES_OUT \
      perfetto-trace-file=$TRACE_OUT" \
    "$CLEAN_MLIR" \
    2>&1

# ── Verify outputs ──────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  OUTPUT"
echo "============================================================"

if [ -f "$DES_OUT" ] && [ -f "$TRACE_OUT" ]; then
    DES_SIZE=$(du -h "$DES_OUT" | cut -f1)
    TRACE_SIZE=$(du -h "$TRACE_OUT" | cut -f1)
    echo "  $DES_OUT   ($DES_SIZE)"
    echo "  $TRACE_OUT ($TRACE_SIZE)"

    # Quick validation: count operations in DES
    DES_OPS=$(python3 -c "import json; d=json.load(open('$DES_OUT')); print(len(d['operations']), 'ops,', d['clock_ghz'], 'GHz')" 2>/dev/null || echo "parse failed")
    echo "  DES summary: $DES_OPS"
else
    echo "  [ERROR] Output files not generated!"
    echo "  DES:  $([ -f "$DES_OUT"  ] && echo 'EXISTS' || echo 'MISSING')"
    echo "  Trace: $([ -f "$TRACE_OUT" ] && echo 'EXISTS' || echo 'MISSING')"
    exit 1
fi

echo ""
echo "[DONE] DES + trace generation complete."
