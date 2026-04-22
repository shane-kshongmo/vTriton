# TritonSim

基于 MLIR 的 Ascend NPU 性能建模工具。

当前仓库主要覆盖两类输入：

- AscendModel MLIR：用于 pass 级性能分析与报告生成
- HIVM IR：用于调度、同步与 trace 分析

如需更详细的构建说明，见 [BUILD.md](BUILD.md)。硬件配置说明见
[configs/README.md](configs/README.md)。

## 功能概览

| 工具 | 用途 |
|------|------|
| `tritonsim-opt` | 运行 AscendModel 相关 pass pipeline |
| `tritonsim-hivm` | 直接分析 `.npuir.mlir`，也可从 Triton DSL 触发 compile-only dump |
| `ascend-tiling-opt` | 在构建中存在时提供 tiling 优化入口 |
| `configs/*.json` | 定义硬件参数，默认使用 `configs/ascend_910b.json` |

## 前置要求

在开始构建前，请确认以下工具已安装：

| 依赖 | 最低版本 | 检查命令 |
|------|---------|---------|
| CMake | 3.20 | `cmake --version` |
| Ninja | 任意 | `ninja --version` |
| clang / lld | 15 | `clang --version` |
| Python | 3.8 | `python3 --version` |
| git | 任意 | `git --version` |

**磁盘空间**：至少 40 GB（LLVM 构建产物较大）

**内存**：推荐 16 GB（并行构建建议 32 GB）

**WSL 用户（Windows）**：所有二进制均为 Linux ELF 文件，请在 WSL 终端（Ubuntu 24.04）内执行全部命令。

## 快速开始

### 步骤 1：初始化子模块并应用补丁

```bash
git submodule update --init --recursive
./scripts/apply_patches.sh
```

`thirdparty/triton-ascend` 指向官方 upstream（gitcode.com/Ascend/triton-ascend），本地补丁（`patches/`）在 submodule checkout 后自动应用，提供 compile-only mock 等功能。

> 如果拉取超时或失败，可改用浅克隆：
> ```bash
> git submodule update --init --depth 1 --recursive
> ./scripts/apply_patches.sh
> ```

---

### 步骤 2：构建 LLVM/MLIR

> **注意：首次构建耗时 30–60 分钟，需占用约 30 GB 磁盘空间，仅需执行一次。**

```bash
./scripts/build_llvm.sh
```

脚本会自动完成以下操作：
1. 使用全部 CPU 核心配置并构建 LLVM（启用 `mlir` 和 `llvm` 项目）
2. 将头文件和库安装到 `thirdparty/llvm-project/build/install`

如果脚本输出 `✅ LLVM/MLIR 已构建`，说明 LLVM 已安装，可直接跳到步骤 3。

> **构建失败？** 若出现内存不足错误，可限制并行度：
> ```bash
> cmake --build thirdparty/llvm-project/build --target install -- -j4
> ```

---

### 步骤 3：构建 triton-ascend（可选）

> 如果不需要 Triton DSL / `.ttir` 输入支持，可跳过此步骤，在步骤 4 中改用 `-DTRITONSIM_ENABLE_TRITON=OFF`。
> 注意：Triton DSL 模式依赖完整的 triton-ascend Python 构建，需要 CANN 环境。

```bash
cd thirdparty/triton-ascend
git submodule update --init --depth 1

cd python
LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
TRITON_PLUGIN_DIRS=$(pwd)/../ascend \
TRITON_BUILD_WITH_CCACHE=true \
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_BUILD_PROTON=OFF \
TRITON_WHEEL_NAME="triton-ascend" \
MAX_JOBS=$(nproc) \
python3 setup.py bdist_wheel

# 记录构建目录，供步骤 4 使用
export TRITON_BUILD_DIR=$(ls -d $PWD/build/cmake.* | head -1)
echo "Triton build dir: $TRITON_BUILD_DIR"

cd ../../..
```

---

### 步骤 4：构建 TritonSim

**方式 A：启用 Triton 支持**（推荐，支持 TTIR 建模）

Triton 支持从 `thirdparty/triton-ascend` 的头文件自动启用，无需构建 triton-ascend wheel。

```bash
mkdir -p build && cd build
cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=../thirdparty/llvm-project/build/install/lib/cmake/mlir \
  -DLLVM_DIR=../thirdparty/llvm-project/build/install/lib/cmake/llvm
ninja
cd ..
```

**方式 B：不启用 Triton 支持**（构建更快，无法处理 `.ttir` 输入）

```bash
mkdir -p build && cd build
cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=../thirdparty/llvm-project/build/install/lib/cmake/mlir \
  -DLLVM_DIR=../thirdparty/llvm-project/build/install/lib/cmake/llvm \
  -DTRITONSIM_ENABLE_TRITON=OFF
ninja
cd ..
```

构建成功后，二进制文件位于 `build/bin/`。

---

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

需要 `triton-opt`（来自 triton-ascend 构建）将 TTIR 转为 generic MLIR，再由 `tritonsim-opt` 运行建模 pipeline：

```bash
TRITON_OPT=/path/to/triton-opt
HW_CONFIG=configs/ascend_910b.json

$TRITON_OPT kernel.ttir --allow-unregistered-dialect --mlir-print-op-generic | \
./build/bin/tritonsim-opt - \
  --allow-unregistered-dialect \
  -ascend-perf-model="hardware-config=${HW_CONFIG} arg-bindings=arg7=4096"
```

`arg-bindings` 用于绑定函数参数（如 `scf.for` 的动态上界），根据 `.mlir` 文件内容确定绑定值。

---

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

---

## 端到端使用指南：Triton DSL → HIVM 分析

以 DeepSeek-V3 的稀疏注意力 prefill kernel (`prefill_a5_cvpipe.py`) 为例，演示从 Triton DSL 脚本到性能分析的完整流程。

### 示例 kernel 简介

`prefill_a5_cvpipe.py` 实现了面向 Ascend910D (A5) 的稀疏 Flash Attention prefill kernel，主要优化：

- **QK nope/rope 分裂**：将 K workspace 拆为 `K_nope[T1, N2, K, 512]` 和 `K_rope[T1, N2, K, 64]`，QK matmul 拆成两个子矩阵乘，SV matmul 直接复用 K_nope 数据（潜在 L1 命中）
- **Cube-Vector 混合流水 (cvpipe)**：通过 `enable_mixed_cv=True` 开启 Cube/Vector 双流水
- **Graph-based sync**：通过 `inject_barrier_all=False` 关闭全局 barrier，使用 GraphSyncSolver 进行最小化同步

kernel 入口函数 `test_dsa_prefill` 接收以下参数：

| 参数 | 含义 | 示例值 |
|------|------|--------|
| batch | batch size | 1 |
| q_seq_len | query 序列长度 | 2048 |
| k_seq_len | key 序列长度 | 1024 |
| head_num | 注意力头数 | 16 |
| kv_lora_rank | KV LoRA 维度 (D_v) | 512 |
| qk_rope_head_dim | RoPE head 维度 | 64 |
| dtype | 数据类型 | torch.bfloat16 |

### 步骤 1：准备 kernel 脚本

将 kernel 脚本放置在可访问的路径，例如 `/path/to/prefill_a5_cvpipe.py`。

脚本需要满足以下条件：
- 使用 `@triton.jit` 装饰 kernel 函数
- 提供一个 Python 可调用的入口函数（本例为 `test_dsa_prefill`），负责构造输入 tensor 并调用 kernel
- 入口函数的参数将通过 `--entry-arg` 逐一传入

### 步骤 2：运行 HIVM 分析

使用 `tritonsim-hivm` 的 `--triton-script` 模式，指定入口函数和参数：

```bash
./build/bin/tritonsim-hivm \
  --triton-script /path/to/prefill_a5_cvpipe.py \
  --triton-entry test_dsa_prefill \
  --entry-arg 1 \
  --entry-arg 2048 \
  --entry-arg 1024 \
  --entry-arg 16 \
  --entry-arg 512 \
  --entry-arg 64 \
  --entry-arg torch.bfloat16 \
  --python python3 \
  --scheduler des \
  --des-graph-file /tmp/prefill_a5_des_graph.json \
  --perfetto-trace-file /tmp/prefill_a5_trace.json \
  --keep-dump-dir \
  2>&1
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--triton-script` | Triton kernel 脚本路径 |
| `--triton-entry` | 脚本中的入口函数名 |
| `--entry-arg` | 入口函数的参数，按顺序传入（可多次指定） |
| `--python` | Python 解释器路径 |
| `--scheduler des` | 使用 DES (Discrete Event Simulation) 调度器 |
| `--des-graph-file` | 导出 DES 调度图（JSON 格式） |
| `--perfetto-trace-file` | 导出 Perfetto trace（可在 [ui.perfetto.dev](https://ui.perfetto.dev) 打开） |
| `--keep-dump-dir` | 保留中间编译产物目录，便于调试 |

### 步骤 3：查看分析结果

分析完成后会在终端输出性能报告，包括：

- 各 op 的周期数估计
- Cube/Vector 流水线利用率
- 内存搬运开销

同时可以：

1. **查看 Perfetto trace**：将 `/tmp/prefill_a5_trace.json` 拖入 [ui.perfetto.dev](https://ui.perfetto.dev)，可视化 Cube/Vector/MTE 各单元的时间线
2. **查看 DES 调度图**：`/tmp/prefill_a5_des_graph.json` 包含 op 间依赖关系和调度顺序

### 步骤 4（可选）：调整参数重新分析

修改 `--entry-arg` 即可测试不同 shape 配置下的性能，例如：

```bash
# batch=2, q_seq_len=4096
./build/bin/tritonsim-hivm \
  --triton-script /path/to/prefill_a5_cvpipe.py \
  --triton-entry test_dsa_prefill \
  --entry-arg 2 \
  --entry-arg 4096 \
  --entry-arg 1024 \
  --entry-arg 16 \
  --entry-arg 512 \
  --entry-arg 64 \
  --entry-arg torch.bfloat16 \
  --python python3 \
  --scheduler des \
  --perfetto-trace-file /tmp/prefill_a5_b2s4096_trace.json
```

### WSL 环境下的运行方式

如果在 Windows 环境下使用 WSL，需要将命令写入 shell 脚本再执行：

```bash
# 1. 将脚本写入 WSL 文件系统
cat > /tmp/run_prefill.sh << 'SCRIPT'
#!/bin/bash
set -e

TRITONSIM_HIVM=/mnt/d/work/git/vTriton/build/bin/tritonsim-hivm
PYTHON=/path/to/python3

# 可选：先重新编译
cd /mnt/d/work/git/vTriton/build && ninja -j$(nproc) 2>&1 | tail -5

${TRITONSIM_HIVM} \
  --triton-script /path/to/prefill_a5_cvpipe.py \
  --triton-entry test_dsa_prefill \
  --entry-arg 1 \
  --entry-arg 2048 \
  --entry-arg 1024 \
  --entry-arg 16 \
  --entry-arg 512 \
  --entry-arg 64 \
  --entry-arg torch.bfloat16 \
  --python ${PYTHON} \
  --scheduler des \
  --perfetto-trace-file /tmp/prefill_a5_trace.json \
  2>&1
SCRIPT

# 2. 执行
bash /tmp/run_prefill.sh
```

---

## 仓库结构

```text
include/AscendModel/   公共头文件
lib/AscendModel/       分析、IR 与 transforms 实现
tools/                 命令行工具入口
configs/               硬件配置与 schema
patches/               应用到 thirdparty 子模块的本地补丁
scripts/               构建、补丁应用等辅助脚本
test/                  示例输入与 smoke tests
thirdparty/            外部依赖
```

## 说明

- 默认硬件配置为 Ascend 910B
- 与具体本机路径绑定的示例、临时脚本路径和历史实现细节未保留在本 README 中
- 更深入的构建选项、Triton 集成方式和硬件配置格式请分别查看 [BUILD.md](BUILD.md) 与 [configs/README.md](configs/README.md)

## License

Apache 2.0
