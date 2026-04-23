#!/usr/bin/env bash
# Validate a shared LLVM installation against triton-ascend's pinned LLVM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIN_FILE="${PROJECT_ROOT}/thirdparty/triton-ascend/cmake/llvm-hash.txt"

if [[ ! -f "${PIN_FILE}" ]]; then
  echo "ERROR: missing pin file: ${PIN_FILE}" >&2
  exit 1
fi

EXPECTED_HASH="$(tr -d '[:space:]' < "${PIN_FILE}")"
EXPECTED_SHORT="${EXPECTED_HASH:0:8}"

LLVM_ROOT="${1:-${LLVM_SYSPATH:-${LLVM_INSTALL_PREFIX:-}}}"
if [[ -z "${LLVM_ROOT}" ]]; then
  echo "ERROR: missing LLVM root. Pass as arg or set LLVM_SYSPATH/LLVM_INSTALL_PREFIX." >&2
  exit 1
fi

MLIR_CONFIG="${LLVM_ROOT}/lib/cmake/mlir/MLIRConfig.cmake"
LLVM_CONFIG="${LLVM_ROOT}/lib/cmake/llvm/LLVMConfig.cmake"
LLVM_CONFIG_BIN="${LLVM_ROOT}/bin/llvm-config"
VCS_HEADER="${LLVM_ROOT}/include/llvm/Support/VCSRevision.h"

if [[ ! -f "${MLIR_CONFIG}" || ! -f "${LLVM_CONFIG}" ]]; then
  echo "ERROR: invalid LLVM root: ${LLVM_ROOT}" >&2
  echo "Expected files:" >&2
  echo "  ${MLIR_CONFIG}" >&2
  echo "  ${LLVM_CONFIG}" >&2
  exit 1
fi

ACTUAL_REV=""
if [[ -f "${VCS_HEADER}" ]]; then
  ACTUAL_REV="$(sed -n 's/^#define LLVM_REVISION "\([0-9a-f]\{8,40\}\)"/\1/p' "${VCS_HEADER}" | head -n1 || true)"
fi

if [[ -n "${ACTUAL_REV}" ]]; then
  ACTUAL_SHORT="${ACTUAL_REV:0:8}"
  if [[ "${ACTUAL_SHORT}" != "${EXPECTED_SHORT}" ]]; then
    echo "ERROR: LLVM commit mismatch." >&2
    echo "Expected (triton-ascend): ${EXPECTED_HASH}" >&2
    echo "Actual (installed):      ${ACTUAL_REV}" >&2
    echo "Use LLVM built from triton-ascend pin (${EXPECTED_SHORT})." >&2
    exit 1
  fi
  echo "OK: LLVM revision matches triton-ascend pin: ${ACTUAL_REV}"
else
  if [[ -x "${LLVM_CONFIG_BIN}" ]]; then
    LLVM_VERSION="$("${LLVM_CONFIG_BIN}" --version 2>/dev/null || true)"
    if [[ -n "${LLVM_VERSION}" ]]; then
      echo "WARN: cannot read LLVM revision hash from install; only version is available: ${LLVM_VERSION}" >&2
      echo "WARN: ensure this install was built from triton-ascend pin ${EXPECTED_HASH}" >&2
    else
      echo "WARN: llvm-config not usable for additional checks." >&2
    fi
  else
    echo "WARN: ${LLVM_CONFIG_BIN} not found; hash check skipped." >&2
  fi
fi

echo "LLVM_ROOT=${LLVM_ROOT}"
echo "MLIR_DIR=${LLVM_ROOT}/lib/cmake/mlir"
echo "LLVM_DIR=${LLVM_ROOT}/lib/cmake/llvm"
