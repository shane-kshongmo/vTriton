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

```bash
# 构建 (详见 BUILD.md)
mkdir build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR=$LLVM_INSTALL_PREFIX/lib/cmake/mlir
ninja

# 方式1: 使用完整 pipeline (推荐)
./bin/tritonsim-opt test/softmax.ttir -ascend-perf-model

# 方式2: 分步执行各 pass
./bin/tritonsim-opt test/softmax.mlir \
  -convert-triton-to-ascend \
  -assign-op-ids \
  -estimate-cycles \
  -analyze-pipeline \
  -perf-report

# 使用自定义硬件配置
./bin/tritonsim-opt test/softmax.mlir \
  --hardware-config=configs/ascend_910b.json \
  -ascend-perf-model
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
