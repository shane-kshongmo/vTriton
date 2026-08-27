#!/bin/bash
# Reconfigure vTriton to use AscendNPU-IR's LLVM 19 build and bishengir libs.
#
# Usage:
#   bash scripts/reconfigure_vtriton.sh [--bishengir-root <path>]
#
# The script auto-detects the AscendNPU-IR fast-build layout. Override with
# --bishengir-root if the repo is checked out at a non-default location.
set -euo pipefail

PROJECT_ROOT="/mnt/d/work/git/vTriton"

# --- Parse arguments ---------------------------------------------------------
BISHENGIR_ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bishengir-root)
      BISHENGIR_ROOT="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# --- Resolve AscendNPU-IR path -----------------------------------------------
# Default: sibling checkout under triton-ascend
DEFAULT_NPU_IR="/mnt/d/work/git/triton-ascend/third_party/ascend/AscendNPU-IR"

if [ -n "${BISHENGIR_ROOT}" ]; then
  ASCEND_NPU_IR="${BISHENGIR_ROOT}"
elif [ -d "${DEFAULT_NPU_IR}" ]; then
  ASCEND_NPU_IR="${DEFAULT_NPU_IR}"
else
  echo "ERROR: AscendNPU-IR not found at ${DEFAULT_NPU_IR}"
  echo "  Pass --bishengir-root <path> or clone AscendNPU-IR to the default location."
  exit 1
fi

# fast-build: no install/ prefix; cmake configs are in build/lib/cmake/
LLVM_CMAKE="${ASCEND_NPU_IR}/build/lib/cmake"
BISHENGIR_BUILD="${ASCEND_NPU_IR}/build"
BISHENGIR_SRC="${ASCEND_NPU_IR}/bishengir"

export PATH="$HOME/.local/bin:$PATH"

echo "=== Reconfiguring vTriton for BiShengIR HIVM integration ==="
echo ""
echo "  AscendNPU-IR root : ${ASCEND_NPU_IR}"
echo "  LLVM cmake        : ${LLVM_CMAKE}"
echo "  BiShengIR build   : ${BISHENGIR_BUILD}"
echo "  BiShengIR source  : ${BISHENGIR_SRC}"
echo ""

# --- Verify build artifacts --------------------------------------------------
if [ ! -f "${LLVM_CMAKE}/mlir/MLIRConfig.cmake" ]; then
  echo "ERROR: MLIR cmake config not found at ${LLVM_CMAKE}/mlir/"
  echo "  Build AscendNPU-IR first: bash scripts/build_bishengir.sh"
  exit 1
fi

if [ ! -f "${BISHENGIR_BUILD}/lib/libBiShengIRHIVMDialect.a" ]; then
  echo "ERROR: HIVM dialect lib not found at ${BISHENGIR_BUILD}/lib/"
  echo "  Build AscendNPU-IR first: bash scripts/build_bishengir.sh"
  exit 1
fi

# --- Clean old cmake cache ---------------------------------------------------
echo ">>> Cleaning old cmake cache..."
rm -f "${PROJECT_ROOT}/build/CMakeCache.txt"
rm -rf "${PROJECT_ROOT}/build/CMakeFiles"

# --- Reconfigure -------------------------------------------------------------
echo ">>> Reconfiguring..."
cmake -G Ninja \
  -S "${PROJECT_ROOT}" \
  -B "${PROJECT_ROOT}/build" \
  -DMLIR_DIR="${LLVM_CMAKE}/mlir" \
  -DLLVM_DIR="${LLVM_CMAKE}/llvm" \
  -DTRITONSIM_BISHENGIR_SRC_DIR="${BISHENGIR_SRC}" \
  -DTRITONSIM_BISHENGIR_BUILD_DIR="${BISHENGIR_BUILD}" \
  -DTRITONSIM_ENABLE_BISHENGIR_HIVM=ON \
  -DCMAKE_BUILD_TYPE=Release

# --- Build -------------------------------------------------------------------
echo ""
echo ">>> Building tritonsim-hivm..."
ninja -C "${PROJECT_ROOT}/build" tritonsim-hivm

# --- Verify ------------------------------------------------------------------
echo ""
echo "=== Verifying ==="

echo ">>> Testing fixture parse..."
"${PROJECT_ROOT}/build/bin/tritonsim-hivm" \
  --npuir-file "${PROJECT_ROOT}/test/hivm_add_kernel.npuir.mlir" \
  --des-graph-file /tmp/test_des.json \
  --hardware-config "${PROJECT_ROOT}/configs/ascend_910b3_v4.json" 2>&1

if [ -f /tmp/test_des.json ] && [ -s /tmp/test_des.json ]; then
  echo ""
  echo "SUCCESS: tritonsim-hivm parsed fixture and emitted DES graph"
  echo "  Operations in DES graph:"
  python3 -c "import json; d=json.load(open('/tmp/test_des.json')); print(f'  {len(d.get(\"operations\", d.get(\"nodes\", [])))} operations')"
else
  echo ""
  echo "FAILURE: DES graph was not emitted"
  exit 1
fi

# --- Run Python tests --------------------------------------------------------
echo ""
echo ">>> Running Python test suite..."
cd "${PROJECT_ROOT}"
python3 -m pytest tests/perfbound/test_hivm_cli_integration.py -v 2>&1 | tail -20

echo ""
echo "=== DONE ==="
