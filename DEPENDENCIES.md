# Dependency Analysis: Triton-Ascend Ecosystem

## triton-ascend

**Repository**: https://gitcode.com/Ascend/triton-ascend
**Description**: Triton compilation framework for Huawei Ascend NPUs
**License**: MIT

### Git Submodules

Initialized via `git submodule update --init --depth 1`:

| Submodule | Description |
|-----------|-------------|
| `third_party/triton` | OpenAI Triton core compiler |
| Additional third-party deps | Bundled in Triton's build system (pybind11, etc.) |

### Key Dependencies

| Dependency | Version / Commit | Notes |
|-----------|------------------|-------|
| LLVM/MLIR | commit `b5cc222d7429fe6f18c787f633d5262fac2e676f` | Must be built from source |
| Python | 3.9 – 3.11 | |
| clang + lld | >= 15 | For building LLVM and Triton |
| cmake | >= 3.20 | |
| CANN | 8.3.RC1 or 8.5.0 | Ascend runtime (for on-device execution) |
| torch_npu | 2.6.0 | PyTorch NPU adapter |

### Build

```bash
git clone https://gitcode.com/Ascend/triton-ascend.git
cd triton-ascend && git submodule update --init --depth 1

LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
TRITON_PLUGIN_DIRS=./ascend \
python3 setup.py develop
```

---

## AscendNPU-IR (BiShengIR)

**Repository**: https://gitcode.com/Ascend/AscendNPU-IR
**Description**: MLIR-based intermediate representation framework for Ascend NPU compilation
**License**: Apache 2.0

### Git Submodules

| Submodule | URL | Description |
|-----------|-----|-------------|
| `third-party/llvm-project` | https://github.com/llvm/llvm-project.git | LLVM/MLIR infrastructure |
| `third-party/torch-mlir` | https://github.com/llvm/torch-mlir.git | PyTorch → MLIR bridge |

### IR Architecture

Three-tier MLIR dialect system:
1. **High-level**: Computation, data movement, synchronization abstractions
2. **Mid-level**: Hardware-agnostic expression mapping
3. **Low-level**: Fine-grained memory addressing and pipeline control

### Build

CMake-based build system. Requires MLIR framework and optionally CANN for device targeting.

---

## vTriton / TritonSim (this project)

**No git submodules** — dependencies are configured via CMake flags.

### Dependency Graph

```
LLVM/MLIR (commit b5cc222d)
├── triton-ascend (optional, for Triton dialect support)
│   └── provides: TritonIR libraries, .ttir files, Triton headers
└── TritonSim (this project)
    ├── -DMLIR_DIR=<llvm-install>/lib/cmake/mlir
    ├── -DLLVM_DIR=<llvm-install>/lib/cmake/llvm
    └── (optional) -DTRITON_SRC_DIR / -DTRITON_BUILD_DIR
```

### Build Modes

| Mode | CMake Flags | Description |
|------|-------------|-------------|
| **A: With Triton** | `-DTRITON_SRC_DIR=... -DTRITON_BUILD_DIR=...` | Full Triton IR parsing + AscendModel conversion |
| **B: Without Triton** | `-DTRITONSIM_ENABLE_TRITON=OFF` | AscendModel IR input only |
