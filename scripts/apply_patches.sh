#!/usr/bin/env bash
# Apply local patches on top of thirdparty submodules after checkout.
#
# Usage:
#   ./scripts/apply_patches.sh           # apply all patches
#   ./scripts/apply_patches.sh --check   # dry-run, check only
#
# Called automatically by build scripts. Safe to run multiple times
# (patches are applied with --check first to avoid errors on re-run).
#===----------------------------------------------------------------------===//

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_DIR="${PROJECT_ROOT}/patches"

CHECK_ONLY=0
if [ "${1:-}" = "--check" ]; then
  CHECK_ONLY=1
fi

apply_patch() {
  local patch_file="$1"
  local target_dir="$2"
  local patch_name
  patch_name="$(basename "$patch_file")"

  if [ ! -f "$patch_file" ]; then
    echo "SKIP: ${patch_name} (file not found)"
    return 0
  fi

  if [ ! -d "$target_dir" ]; then
    echo "SKIP: ${patch_name} (target dir not found: ${target_dir})"
    return 0
  fi

  # Check if already applied
  if git -C "$target_dir" apply --check "$patch_file" 2>/dev/null; then
    if [ "$CHECK_ONLY" -eq 1 ]; then
      echo "OK:   ${patch_name} (would apply cleanly)"
    else
      git -C "$target_dir" apply "$patch_file"
      echo "APPLY: ${patch_name}"
    fi
  else
    echo "SKIP: ${patch_name} (already applied or does not apply)"
  fi
}

echo "=== Applying patches ==="

# Patch 1: compile-only mock for hardware-free IR dumping
apply_patch \
  "${PATCH_DIR}/triton-ascend-compile-only-mock.patch" \
  "${PROJECT_ROOT}/thirdparty/triton-ascend"

# Patch 2: AscendNPU-IR LLVM 20 compatibility (for bishengir compile)
# This patches the nested submodule inside triton-ascend
apply_patch \
  "${PATCH_DIR}/ascendnpu-ir-llvm20-compat.patch" \
  "${PROJECT_ROOT}/thirdparty/triton-ascend/third_party/ascend/AscendNPU-IR"

echo "=== Patches done ==="
