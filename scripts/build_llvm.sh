#!/usr/bin/env bash
#===----------------------------------------------------------------------===//
# build_llvm.sh — 从 thirdparty/llvm-project submodule 构建 LLVM/MLIR
#
# 用法:
#   ./scripts/build_llvm.sh              # 默认 Release，安装到 submodule 下
#   ./scripts/build_llvm.sh Debug        # Debug 模式
#   LLVM_INSTALL_PREFIX=/opt/llvm ./scripts/build_llvm.sh  # 自定义安装路径
#===----------------------------------------------------------------------===//

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LLVM_SRC="${PROJECT_ROOT}/thirdparty/llvm-project"
BUILD_TYPE="${1:-Release}"
LLVM_BUILD="${LLVM_SRC}/build"
LLVM_INSTALL="${LLVM_INSTALL_PREFIX:-${LLVM_BUILD}/install}"

# ──── 检查 submodule ────
if [ ! -f "${LLVM_SRC}/llvm/CMakeLists.txt" ]; then
  echo "❌ LLVM submodule 未初始化。执行:"
  echo "   git submodule update --init thirdparty/llvm-project"
  exit 1
fi

# ──── 检查是否已构建 ────
if [ -f "${LLVM_INSTALL}/lib/cmake/mlir/MLIRConfig.cmake" ]; then
  echo "✅ LLVM/MLIR 已构建: ${LLVM_INSTALL}"
  echo "   如需重建，请先删除: rm -rf ${LLVM_BUILD}"
  exit 0
fi

# ──── 检测 CPU 数 ────
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "============================================"
echo " 构建 LLVM/MLIR (${BUILD_TYPE})"
echo " 源码:   ${LLVM_SRC}"
echo " 构建:   ${LLVM_BUILD}"
echo " 安装:   ${LLVM_INSTALL}"
echo " 并行:   ${NPROC} jobs"
echo " 预计耗时 30-60 分钟 (仅首次)"
echo "============================================"
echo ""

# ──── Configure ────
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
  -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL}" \
  -S "${LLVM_SRC}/llvm" \
  -B "${LLVM_BUILD}"

# ──── Build + Install ────
cmake --build "${LLVM_BUILD}" --target install -- -j"${NPROC}"

echo ""
echo "✅ LLVM/MLIR 构建完成!"
echo "   MLIR_DIR=${LLVM_INSTALL}/lib/cmake/mlir"
echo ""
echo "   后续构建本项目时无需手动指定路径，CMake 会自动检测。"
echo "   也可以显式指定: cmake -DMLIR_DIR=${LLVM_INSTALL}/lib/cmake/mlir .."
