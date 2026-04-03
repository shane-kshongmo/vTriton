# TritonSim

基于 MLIR 的 Ascend NPU 性能建模工具。

当前仓库主要覆盖两类输入：

- AscendModel MLIR：用于 pass 级性能分析与报告生成
- HIVM IR：用于调度、同步与 trace 分析

如需更详细的构建说明，见 [BUILD.md](BUILD.md)。硬件配置说明见
[configs/README.md](configs/README.md)。

## 功能概览

- `tritonsim-opt`：运行 AscendModel 相关 pass pipeline
- `tritonsim-hivm`：直接分析 `.npuir.mlir`，也可从 Triton DSL 触发 compile-only dump
- `ascend-tiling-opt`：在构建中存在时提供 tiling 优化入口
- `configs/*.json`：定义硬件参数，默认使用 `configs/ascend_910b.json`

## 快速开始

### 1. 初始化子模块

```bash
git submodule update --init --recursive
```

### 2. 构建 LLVM/MLIR

```bash
export LLVM_INSTALL_PREFIX=$HOME/llvm-install
./scripts/build_llvm.sh
```

### 3. 构建本项目

```bash
mkdir -p build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR=$LLVM_INSTALL_PREFIX/lib/cmake/mlir \
  -DLLVM_DIR=$LLVM_INSTALL_PREFIX/lib/cmake/llvm
ninja
```

如果需要禁用 Triton dialect 支持，在 `cmake` 命令后追加
`-DTRITONSIM_ENABLE_TRITON=OFF`。

## 常用用法

### 分析 AscendModel MLIR

运行完整 pipeline：

```bash
./build/bin/tritonsim-opt test/ascend_ops.mlir -ascend-perf-model
```

分步运行常用 pass：

```bash
./build/bin/tritonsim-opt test/ascend_ops.mlir \
  -assign-op-ids \
  -estimate-cycles \
  -analyze-pipeline \
  -perf-report
```

指定硬件配置：

```bash
./build/bin/tritonsim-opt test/ascend_ops.mlir \
  -ascend-perf-model="hardware-config=configs/ascend_910b.json"
```

### 分析 HIVM IR

直接分析仓库内示例：

```bash
./build/bin/tritonsim-hivm --npuir-file test/hivm_add_kernel.npuir.mlir
./build/bin/tritonsim-hivm --npuir-file test/hivm_mixed_cv_kernel.npuir.mlir
```

导出 Perfetto trace：

```bash
./build/bin/tritonsim-hivm \
  --npuir-file test/hivm_add_kernel.npuir.mlir \
  --perfetto-trace-file /tmp/hivm_trace.json
```

也可在 `tritonsim-opt` 中直接对 HIVM IR 跑 pass：

```bash
./build/bin/tritonsim-opt --analyze-hivm test/hivm_mixed_cv_kernel.npuir.mlir
```

### 从 Triton DSL 触发 HIVM 分析

该模式依赖可用的 Python + `triton-ascend` 环境：

```bash
./build/bin/tritonsim-hivm \
  --triton-script test/triton_smoke.py \
  --python python3
```

如果脚本提供明确入口，优先使用 `--triton-entry` / `--entry-arg`：

```bash
./build/bin/tritonsim-hivm \
  --triton-script path/to/script.py \
  --triton-entry main \
  --entry-arg 1 \
  --python python3
```

### 分析 Triton IR (`.ttir`)

启用 Triton 支持构建时，可直接输入 `.ttir`：

```bash
./build/bin/tritonsim-opt test/flash_attention.ttir -ascend-perf-model
```

如果构建时关闭了 Triton dialect 支持，需要先通过 `triton-opt` 转成 generic MLIR：

```bash
TRITON_OPT=/path/to/triton-opt

$TRITON_OPT kernel.ttir --allow-unregistered-dialect --mlir-print-op-generic | \
./build/bin/tritonsim-opt - \
  --allow-unregistered-dialect \
  -ascend-perf-model="loop-trip-counts=1"
```

## 测试与验证

常用本地检查：

```bash
./build/bin/tritonsim-opt test/ascend_ops.mlir -ascend-perf-model
./build/bin/tritonsim-hivm --npuir-file test/hivm_add_kernel.npuir.mlir
python3 test/triton_smoke.py
```

启用测试后可运行：

```bash
ctest --test-dir build
```

## 仓库结构

```text
include/AscendModel/   公共头文件
lib/AscendModel/       分析、IR 与 transforms 实现
tools/                 命令行工具入口
configs/               硬件配置与 schema
test/                  示例输入与 smoke tests
thirdparty/            外部依赖
```

## 说明

- 默认硬件配置为 Ascend 910B
- 与具体本机路径绑定的示例、临时脚本路径和历史实现细节未保留在本 README 中
- 更深入的构建选项、Triton 集成方式和硬件配置格式请分别查看 [BUILD.md](BUILD.md) 与 [configs/README.md](configs/README.md)

## License

Apache 2.0
