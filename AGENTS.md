# Repository Guidelines

## Project Structure & Module Organization
`include/AscendModel/` contains public headers for analysis, IR, and transform passes. `lib/AscendModel/` holds the corresponding implementations. CLI entrypoints live in `tools/` (`tritonsim-opt`, `tritonsim-hivm`, `ascend-tiling-opt`). Hardware definitions and schemas are under `configs/`, and sample inputs plus smoke tests live in `test/`. External dependencies are vendored in `thirdparty/`; treat them as upstream code unless a task explicitly targets them.

## Build, Test, and Development Commands
Initialize submodules before first build:

```bash
git submodule update --init --recursive
./scripts/build_llvm.sh
mkdir -p build && cd build
cmake -G Ninja .. -DMLIR_DIR=$LLVM_INSTALL_PREFIX/lib/cmake/mlir -DLLVM_DIR=$LLVM_INSTALL_PREFIX/lib/cmake/llvm
ninja
```

Useful local checks:

```bash
./build/bin/tritonsim-opt test/ascend_ops.mlir -ascend-perf-model
./build/bin/tritonsim-hivm --npuir-file test/hivm_add_kernel.npuir.mlir
python3 test/triton_smoke.py
```

Enable CTest integration only when needed with `-DASCEND_MODEL_ENABLE_TESTS=ON`, then run `ctest --test-dir build`.

## Local Environment
Prefer the machine-local setup documented in `LOCAL_ENVIRONMENT.md` before attempting any fresh dependency bootstrap. If that file is missing, first search for an existing venv under `/mnt/c/Users/shane` and a CANN install under `~/Ascend`; only build from scratch and download/install CANN if those local installs are not present.

## Coding Style & Naming Conventions
Follow the existing LLVM/MLIR-oriented C++17 style already used in `lib/` and `tools/`: 2-space indentation for wrapped arguments, braces on their own lines for functions, and grouped includes with LLVM/MLIR headers before STL headers. Use `CamelCase` for types and passes, `lowerCamelCase` for functions and locals, and descriptive filenames such as `HIVMAnalysis.cpp` or `PipelineAnalysisPass.cpp`. Keep new MLIR test files descriptive, for example `feature_name_ascend.mlir`.

## Testing Guidelines
Prefer adding a focused fixture in `test/` whenever you change analysis logic, pass behavior, or hardware modeling. Reproduce the intended workflow with the shipped tools instead of relying on manual inspection alone. If a change affects Triton ingestion, run both an MLIR sample and `python3 test/triton_smoke.py`.

## Commit & Pull Request Guidelines
Recent history uses short imperative subjects such as `Add HIVM-native DES analysis and trace export` and `Fix build errors and add new features for LLVM 20 / GCC 13 compatibility`. Keep commits narrowly scoped and explain the affected subsystem in the title. PRs should describe the modeled behavior change, list the commands you ran, link related issues, and include before/after output snippets when a report format or scheduler result changes.
