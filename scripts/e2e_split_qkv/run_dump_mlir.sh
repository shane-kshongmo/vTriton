#!/bin/bash
# =============================================================================
# run_dump_mlir.sh  —  Step 1: Dump HIVM MLIR for split_qkv kernel
# =============================================================================
# Environment   : CANN 9.0 container (triton_dev)
#                Uses triton_patch from CANN 8.5 (copied) for tl.extract_slice
# Kernel        : split_qkv_rmsnorm_mrope
# Output        : scripts/e2e_split_qkv/dump/kernel.npuir.mlir
# =============================================================================
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
OUT_DIR="$ROOT_DIR/scripts/e2e_split_qkv/dump"
mkdir -p "$OUT_DIR"
echo "=== Step 1: Dump HIVM MLIR — split_qkv_rmsnorm_mrope ==="
export TRITON_COMPILE_ONLY=1 TRITON_ENABLE_TASKQUEUE=0 TRITON_ASCEND_ARCH=Ascend910_9362
rm -rf /root/.triton/cache/*
python3 tools/common/triton_dsl_dump_launcher.py \
    --script test/split_qkv_rmsnorm_mrope_standalone.py \
    --dump-dir "$OUT_DIR" 2>&1 | tee "$OUT_DIR/tier1_launcher.log" || true
if compgen -G "$OUT_DIR/kernel_*.npuir.mlir" > /dev/null 2>&1; then
    NPUIR=$(ls "$OUT_DIR"/kernel_*.npuir.mlir 2>/dev/null | head -1)
    cp "$NPUIR" "$OUT_DIR/kernel.npuir.mlir"
    echo "[OK] Tier 1: $(wc -l < "$OUT_DIR/kernel.npuir.mlir") lines"
elif [ -f "$OUT_DIR/kernel.npuir.mlir" ]; then
    echo "[OK] existing: $(wc -l < "$OUT_DIR/kernel.npuir.mlir") lines"
else
    # Tier 2: direct bishengir-compile (CANN 9 uses hivm-graph-sync-solver flag)
    echo "[Tier 2] Direct bishengir-compile..."
    BIS=/usr/local/Ascend/ascend-toolkit/latest/bin/bishengir-compile
    TTAD=$(find /root/.triton/cache -name '*.ttadapter' 2>/dev/null | head -1)
    if [ -z "$TTAD" ]; then
        echo "[FAIL] No ttadapter found"
        exit 1
    fi
    TMP_OUT=/tmp/split_qkv_npuir.txt
    "$BIS" \
        --bishengir-print-ir-after=hivm-graph-sync-solver \
        --target=Ascend910_9362 \
        --enable-auto-multi-buffer=True --enable-auto-bind-sub-block=True \
        --enable-hfusion-compile=true --enable-hivm-compile=true \
        --enable-triton-kernel-compile=true \
        "$TTAD" -o /dev/null > "$TMP_OUT" 2>&1 || true
    python3 - "$TMP_OUT" "$OUT_DIR/kernel.npuir.mlir" << 'PYEOF'
import sys
t=open(sys.argv[1],encoding='utf-8',errors='replace').read()
o=sys.argv[2];S="// -----// IR Dump After";E="// -----// IR Dump Before"
m=[];ls=t.split('\n');i=0
while i<len(ls):
 if S not in ls[i]:i+=1;continue
 i+=1;si=i
 while i<len(ls)and not(E in ls[i]and i>si+1):i+=1
 b=ls[si:i]
 if any('func.func'in l for l in b):
  while b and not b[0].strip():b.pop(0)
  while b and not b[-1].strip():b.pop()
  m.append('\n'.join(b))
 i+=1
if m:open(o,'w').write(m[-1]+'\n');print(f"  [OK] {len(m[-1].splitlines())} lines")
else:sys.exit(1)
PYEOF
    [ -s "$OUT_DIR/kernel.npuir.mlir" ] && echo "[OK] Tier 2: $(wc -l < "$OUT_DIR/kernel.npuir.mlir") lines" || { echo "[FAIL]"; exit 1; }
fi
echo "[DONE] -> run_des_trace.sh"
