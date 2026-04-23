#!/usr/bin/env bash
#===----------------------------------------------------------------------===//
# build_llvm.sh — 构建与 triton-ascend 对齐的 LLVM/MLIR（单份共享安装）
#
# 用法:
#   ./scripts/build_llvm.sh
#   ./scripts/build_llvm.sh Debug
#   LLVM_SRC_DIR=/path/to/llvm-project LLVM_INSTALL_PREFIX=/opt/llvm ./scripts/build_llvm.sh
#===----------------------------------------------------------------------===//

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIN_FILE="${PROJECT_ROOT}/thirdparty/triton-ascend/cmake/llvm-hash.txt"
VALIDATE_SCRIPT="${SCRIPT_DIR}/validate_shared_llvm.sh"

LLVM_SRC="${LLVM_SRC_DIR:-${PROJECT_ROOT}/thirdparty/llvm-project}"
BUILD_TYPE="${1:-Release}"
LLVM_BUILD="${LLVM_BUILD_DIR:-${LLVM_SRC}/build}"
LLVM_INSTALL="${LLVM_INSTALL_PREFIX:-${LLVM_BUILD}/install}"

if [ ! -f "${PIN_FILE}" ]; then
  echo "❌ 找不到 triton-ascend LLVM pin: ${PIN_FILE}"
  echo "   先执行: git submodule update --init --recursive"
  exit 1
fi

EXPECTED_HASH="$(tr -d '[:space:]' < "${PIN_FILE}")"
EXPECTED_SHORT="${EXPECTED_HASH:0:8}"

if [ ! -f "${LLVM_SRC}/llvm/CMakeLists.txt" ]; then
  echo "❌ 找不到 LLVM 源码目录: ${LLVM_SRC}"
  echo "   本仓库默认不再要求独立 llvm submodule。"
  echo "   可选方案:"
  echo "   1) 指向已有源码: LLVM_SRC_DIR=/path/to/llvm-project"
  echo "   2) 本地克隆 pin 版本:"
  echo "      git clone https://github.com/llvm/llvm-project.git ${LLVM_SRC}"
  echo "      git -C ${LLVM_SRC} checkout ${EXPECTED_HASH}"
  exit 1
fi

if git -C "${LLVM_SRC}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  ACTUAL_HASH="$(git -C "${LLVM_SRC}" rev-parse HEAD)"
  ACTUAL_SHORT="${ACTUAL_HASH:0:8}"
  if [ "${ACTUAL_SHORT}" != "${EXPECTED_SHORT}" ] && [ "${ALLOW_UNPINNED_LLVM:-0}" != "1" ]; then
    echo "❌ LLVM 源码版本与 triton-ascend pin 不一致"
    echo "   expected: ${EXPECTED_HASH}"
    echo "   actual:   ${ACTUAL_HASH}"
    echo "   如确认兼容，可临时绕过: ALLOW_UNPINNED_LLVM=1"
    exit 1
  fi
fi

if [ -f "${LLVM_INSTALL}/lib/cmake/mlir/MLIRConfig.cmake" ]; then
  echo "✅ LLVM/MLIR 已存在: ${LLVM_INSTALL}"
  if [ -x "${VALIDATE_SCRIPT}" ]; then
    "${VALIDATE_SCRIPT}" "${LLVM_INSTALL}"
  fi
  echo "   如需重建，请先删除: rm -rf ${LLVM_BUILD}"
  exit 0
fi

NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "============================================"
echo " 构建共享 LLVM/MLIR (${BUILD_TYPE})"
echo " triton-ascend pin: ${EXPECTED_HASH}"
echo " 源码:   ${LLVM_SRC}"
echo " 构建:   ${LLVM_BUILD}"
echo " 安装:   ${LLVM_INSTALL}"
echo " 并行:   ${NPROC} jobs"
echo " 预计耗时 30-60 分钟 (仅首次)"
echo "============================================"
echo ""

cmake -G Ninja \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
  -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL}" \
  -S "${LLVM_SRC}/llvm" \
  -B "${LLVM_BUILD}"

cmake --build "${LLVM_BUILD}" --target install -- -j"${NPROC}"

if [ -x "${VALIDATE_SCRIPT}" ]; then
  "${VALIDATE_SCRIPT}" "${LLVM_INSTALL}"
fi

echo ""
echo "✅ LLVM/MLIR 构建完成!"
echo "   MLIR_DIR=${LLVM_INSTALL}/lib/cmake/mlir"
echo "   LLVM_DIR=${LLVM_INSTALL}/lib/cmake/llvm"
echo "   LLVM_SYSPATH=${LLVM_INSTALL}"
