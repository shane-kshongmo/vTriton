# TritonSim - Ascend NPU Performance Modeling

基于 MLIR 的昇腾 NPU 性能建模工具，支持可配置的硬件抽象层。

## 概述

TritonSim 将 Triton IR 转换为硬件感知的 AscendModel IR，并基于目标硬件的配置
进行精确的性能分析。通过 JSON 配置文件定义硬件参数，可以适配不同的硬件平台。

### 核心流程

```
.ttir 文件           AscendModel IR              性能报告
(Triton IR)         (硬件感知 IR)               (Roofline 分析)
    │                    │                          │
    ▼                    ▼                          ▼
┌────────────┐     ┌────────────┐             ┌────────────┐
│ tt.dot     │     │ ascend.    │             │ Pipeline   │
│ tt.load    │ ──▶ │ matmul     │ ──────────▶ │ Analysis   │
│ tt.exp     │     │ vector_*   │             │ Roofline   │
└────────────┘     └────────────┘             └────────────┘
                         ▲
                         │
              ┌──────────────────┐
              │  Hardware Config │
              │  (JSON file)     │
              └──────────────────┘
```

## 快速开始

本仓库默认通过子模块提供 `triton-ascend` 源码，路径为 `thirdparty/triton-ascend`
（当前指向 mocked 仓库 `https://github.com/shane-kshongmo/Trtiton-Ascend`）:

```bash
git submodule update --init --recursive
```

```bash
# 构建 (详见 BUILD.md)
mkdir build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR=$LLVM_INSTALL_PREFIX/lib/cmake/mlir \
  -DLLVM_DIR=$LLVM_INSTALL_PREFIX/lib/cmake/llvm
ninja
```

### 直接分析 AscendModel dialect (.mlir)

```bash
# 完整 pipeline (推荐)
./bin/tritonsim-opt test/ascend_ops.mlir -ascend-perf-model

# 分步执行各 pass
./bin/tritonsim-opt test/ascend_ops.mlir \
  -assign-op-ids \
  -estimate-cycles \
  -analyze-pipeline \
  -perf-report

# 使用自定义硬件配置 (选项传入 pipeline)
./bin/tritonsim-opt test/ascend_ops.mlir \
  -ascend-perf-model="hardware-config=configs/ascend_910b.json"
```

### 直接分析 HIVM IR (.npuir.mlir)

新增的 `tritonsim-hivm` 工具直接消费 `triton-ascend` 转储出的
`kernel.npuir.mlir`，基于显式的 `set_flag` / `wait_flag` /
`pipe_barrier` 和 pipe 资源做调度分析。分析器优先走 MLIR 解析路径，
若本机存在 `triton-ascend` 已构建出的 BiShengIR HIVM dialect 头文件与静态库，
会优先按真实 `hivm.hir.*` typed op 解析 `sync_block_*`、DMA、macro op 和
pipe/core attrs；遇到当前本地仍未注册的 HIVM 语法时才回退到兼容导入器。

```bash
./bin/tritonsim-hivm --npuir-file test/hivm_add_kernel.npuir.mlir
./bin/tritonsim-hivm --npuir-file test/hivm_mixed_cv_kernel.npuir.mlir
./bin/tritonsim-hivm \
  --npuir-file test/hivm_add_kernel.npuir.mlir \
  --perfetto-trace-file /tmp/hivm_add_trace.json
```

### 从 Triton DSL 生成 HIVM IR 并分析

`tritonsim-hivm` 也可以可选地调用 `triton-ascend` 的 compile-only
dump 流程。该模式依赖可工作的 Python + triton-ascend 环境，以及本机可用
的 CANN 工具链。

```bash
./bin/tritonsim-hivm \
  --triton-script test/triton_smoke.py \
  --python /mnt/c/Users/shane/bin/python
```

可通过 `--script-arg <arg>` 透传脚本参数；若 HIVM 中存在动态 loop bound，
可通过 `--arg-bindings=arg10=128,arg11=64` 为分析器提供绑定。若需要时序可视化，
可追加 `--perfetto-trace-file /path/to/trace.json`，输出可直接导入 Perfetto /
Chrome trace viewer 的事件文件。

若环境不完整，工具会在真正执行脚本前做 preflight，并明确提示
`triton` / `torch` / `torch_npu` 依赖缺失，而不是只返回笼统的 Python 失败。
运行时会自动:

- 为 compile-only 模式注入 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`
- 设置默认 `TRITON_ASCEND_ARCH=Ascend910_9599`
- 将 Triton cache 重定向到临时目录，避免污染或依赖用户目录权限
- 尝试从常见安装路径探测 CANN，并补齐 `ASCEND_HOME_PATH` / `PATH` / `LD_LIBRARY_PATH`

`--triton-ascend-root` 仍然保留为兼容选项，但当前 DSL 路径优先使用已安装
的 `triton-ascend` wheel，而不是直接从源码树导入 Python 包，以避免 source tree
shadowing。

### 以 MLIR Pass 方式分析 HIVM IR

`HIVMAnalysis` 现在也作为 `tritonsim-opt` 的原生 `ModuleOp` pass 提供，可直接对
已解析的 HIVM IR 运行调度与同步分析，而不是先走文本扫描器。
当前模型已将 Vector 与 Cube 侧的 `MTE2` 资源拆分为独立资源
`PIPE_MTE2_V` 与 `PIPE_MTE2_C`，避免把两侧 DMA 错误地建模成同一条共享 pipe。

```bash
./build/bin/tritonsim-opt --analyze-hivm test/hivm_mixed_cv_kernel.npuir.mlir
./build/bin/tritonsim-opt \
  --analyze-hivm="scheduler=des hardware-config=configs/ascend_910b.json arg-bindings=arg10=128 perfetto-trace-file=/tmp/hivm_trace.json" \
  /path/to/kernel.npuir.mlir
```

若 `.npuir.mlir` 中包含非 MLIR 内容，例如某些编译转储在文件尾部直接拼接了
LLVM warning 文本，`mlir-opt` 会在 pass 执行前就解析失败。这种情况下需要先清洗
输入文件，只保留合法 MLIR 模块内容，例如：

```bash
sed '/^warning: /,$d' raw.kernel.npuir.mlir > clean.kernel.npuir.mlir
./build/bin/tritonsim-opt -allow-unregistered-dialect --analyze-hivm clean.kernel.npuir.mlir
```

### 分析 Triton IR (.ttir)

**方式 A: 启用 Triton 支持构建 (详见 BUILD.md 方式A)**

`tritonsim-opt` 直接接受 `.ttir` 文件:

```bash
./bin/tritonsim-opt kernel.ttir -ascend-perf-model
```

**方式 B: 未启用 Triton 支持构建 (两阶段 pipeline)**

先用 `triton-opt` 将 `.ttir` 转为通用 MLIR，再传入 `tritonsim-opt`:

```bash
TRITON_OPT=<triton-ascend-build>/bin/triton-opt

$TRITON_OPT kernel.ttir \
  --allow-unregistered-dialect --mlir-print-op-generic | \
./bin/tritonsim-opt - \
  --allow-unregistered-dialect \
  -ascend-perf-model="loop-trip-counts=<N>"
```

> `loop-trip-counts=N` 指定 Triton kernel 中 scf.for 的迭代次数 (程序启动时的实际值)。

**示例: 分析 Flash Attention 转储**

```bash
$TRITON_OPT triton_dumps_fa/<hash>/_attn_fwd.ttir \
  --allow-unregistered-dialect --mlir-print-op-generic | \
./bin/tritonsim-opt - \
  --allow-unregistered-dialect \
  -ascend-perf-model='loop-trip-counts=1'
```

## 硬件抽象架构

### Ascend 910B 数据流

```
┌─────────────────────────────────────────────────────────────┐
│                          HBM                                │
│                      (1.6 TB/s)                             │
└───────┬─────────────────────────────────────────┬───────────┘
        │ MTE2 (Cube)                    MTE2 (Vector) │
        ▼                                              ▼
┌───────────────┐                            ┌───────────────┐
│      L1       │                            │      UB       │
│   (1024KB)    │                            │   (256KB)     │
└───────┬───────┘                            └───────┬───────┘
        │ MTE1                                       │
        ▼                                            ▼
┌───────┴───────┐                            ┌───────────────┐
│  L0A  │  L0B  │                            │    Vector     │
│ (64KB)│(64KB) │                            │  128 × FP16   │
└───┬───┴───┬───┘                            └───────┬───────┘
    │       │                                        │
    └───┬───┘                                        │ MTE3
        ▼                                            ▼
┌───────────────┐                            ┌───────────────┐
│     Cube      │                            │     HBM       │
│  16×16×16     │                            │               │
│  320 TFLOPS   │                            │               │
└───────┬───────┘                            └───────────────┘
        │
        ▼
┌───────────────┐
│      L0C      │
│   (256KB)     │
└───────┬───────┘
        │ FixPipe
        ▼
┌───────────────┐
│     HBM       │
└───────────────┘
```

### 硬件参数 (910B 默认值)

| 组件 | 参数 | 值 |
|------|------|-----|
| **时钟** | 频率 | 1.85 GHz |
| **HBM** | 容量 / 带宽 | 32 GB / 1.6 TB/s |
| **L1** | 容量 | 1024 KB |
| **UB** | 容量 | 256 KB |
| **Cube** | 算力 (FP16) | 320 TFLOPS |
| **Cube** | Tile 大小 | 16×16×16 |
| **Vector** | 宽度 | 128 FP16 |
| **Vector** | 算力 (FP32) | 10 TFLOPS |

## 硬件配置

通过 JSON 文件定义硬件参数，支持:

- **内存空间**: HBM, L2, L1, L0A/B/C, UB
- **计算单元**: Cube (矩阵引擎), Vector (SIMD)
- **数据搬运**: MTE1, MTE2, MTE3, FixPipe
- **流水线**: 数据流路径和并行性

详见 [configs/README.md](configs/README.md)。

## 目录结构

```
TritonSim/
├── BUILD.md                    # 构建指南
├── configs/
│   ├── ascend_910b.json       # 910B 硬件配置
│   ├── hardware_schema.json   # JSON Schema
│   └── README.md              # 配置说明
├── include/AscendModel/
│   ├── HardwareConfig.h       # 硬件配置接口
│   ├── IR/                    # Dialect 定义
│   ├── Transforms/            # Pass 定义
│   └── Analysis/              # 分析功能
├── lib/
│   ├── HardwareConfig.cpp     # 配置加载实现
│   ├── IR/                    # Dialect 实现
│   ├── Transforms/            # Pass 实现
│   └── Analysis/              # 分析实现
├── tools/                     # 主工具
└── test/                      # 测试用例
```

## Pass Pipeline

| Pass | 功能 |
|------|------|
| `-convert-triton-to-ascend` | Triton IR → AscendModel IR |
| `-insert-data-transfers` | 插入 Cube/Vector 之间的数据搬运 |
| `-assign-op-ids` | 分配操作 ID |
| `-estimate-cycles` | 估算执行周期 |
| `-analyze-pipeline` | 流水线调度分析 |
| `-perf-report` | 生成性能报告 |
| `-ascend-perf-model` | 运行完整 pipeline (上述所有 pass) |

## 扩展新硬件

1. 创建新的 JSON 配置文件 (参考 `configs/ascend_910b.json`)
2. 定义内存层次、计算单元、数据搬运路径
3. 运行时通过 `--hardware-config` 指定配置文件

```bash
./bin/tritonsim-opt input.mlir --hardware-config=my_new_hardware.json
```

## 许可证

Apache 2.0
