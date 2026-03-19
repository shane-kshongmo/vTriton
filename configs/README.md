# 硬件配置指南

TritonSim 使用 JSON 文件定义目标硬件的参数，使性能建模能够适配不同的硬件平台。

## 配置文件结构

```
configs/
├── hardware_schema.json    # JSON Schema (用于验证)
├── ascend_910b.json        # 昇腾 910B 配置
└── README.md               # 本文档
```

## 使用方法

```bash
# 使用默认配置 (910B)
./bin/tritonsim-opt input.mlir -analyze-pipeline

# 指定配置文件
./bin/tritonsim-opt input.mlir --hardware-config=configs/ascend_910b.json

# 使用自定义配置
./bin/tritonsim-opt input.mlir --hardware-config=my_custom_hw.json
```

## 配置文件格式

### 基本信息

```json
{
  "name": "Ascend 910B",
  "vendor": "Huawei",
  "version": "1.0",
  
  "clock": {
    "frequency_ghz": 1.85
  }
}
```

### 内存空间 (memory_spaces)

定义芯片的内存层次结构:

```json
"memory_spaces": {
  "hbm": {
    "type": "off_chip",           // off_chip | on_chip_shared | on_chip_local | register_file
    "size_gb": 32,
    "bandwidth_tbps": 1.6,
    "latency_cycles": 200
  },
  "l1": {
    "type": "on_chip_local",
    "size_kb": 1024,
    "bandwidth_gbps": 6400,
    "latency_cycles": 10
  },
  "l0a": {
    "type": "register_file",
    "size_kb": 64,
    "description": "Left matrix input for Cube"
  }
}
```

**910B 内存层次:**

```
┌─────────────────────────────────────────────────────────┐
│                        HBM (32GB)                       │
│                     1.6 TB/s bandwidth                  │
└───────────┬─────────────────────────────┬───────────────┘
            │                             │
            ▼                             ▼
    ┌───────────────┐             ┌───────────────┐
    │   L2 (192MB)  │             │   L2 (192MB)  │
    │    shared     │             │    shared     │
    └───────┬───────┘             └───────┬───────┘
            │                             │
            ▼                             ▼
    ┌───────────────┐             ┌───────────────┐
    │   L1 (1MB)    │             │   UB (256KB)  │
    │  Cube staging │             │ Vector buffer │
    └───────┬───────┘             └───────┬───────┘
            │                             │
     ┌──────┴──────┐                      │
     ▼             ▼                      ▼
┌─────────┐  ┌─────────┐          ┌─────────────┐
│  L0A    │  │  L0B    │          │   Vector    │
│ (64KB)  │  │ (64KB)  │          │   Compute   │
└────┬────┘  └────┬────┘          └─────────────┘
     │            │
     └─────┬──────┘
           ▼
     ┌─────────┐
     │  Cube   │
     │ 16×16×16│
     └────┬────┘
          ▼
     ┌─────────┐
     │  L0C    │
     │ (256KB) │
     └─────────┘
```

### 计算单元 (compute_units)

```json
"compute_units": {
  "cube": {
    "type": "matrix_engine",
    "tflops_fp16": 320,
    "tile_m": 16,
    "tile_n": 16,
    "tile_k": 16,
    "input_spaces": ["l0a", "l0b"],
    "output_space": "l0c"
  },
  "vector": {
    "type": "simd_engine",
    "width_elements": 128,      // 128 个 FP16 元素
    "width_bytes": 256,
    "tflops_fp32": 10,
    "compute_space": "ub"
  }
}
```

### 数据搬运单元 (data_movers)

```json
"data_movers": {
  "cube_mte2": {
    "description": "HBM → L1 (Cube input)",
    "src_space": "hbm",
    "dst_space": "l1",
    "bandwidth_gbps": 200
  },
  "mte1": {
    "description": "L1 → L0A/L0B",
    "src_space": "l1",
    "dst_spaces": ["l0a", "l0b"],
    "bandwidth_gbps": 400
  },
  "fixpipe": {
    "description": "L0C → HBM (Cube output)",
    "src_space": "l0c",
    "dst_space": "hbm",
    "bandwidth_gbps": 200
  },
  "vector_mte2": {
    "description": "HBM → UB (Vector input)",
    "src_space": "hbm",
    "dst_space": "ub",
    "bandwidth_gbps": 200
  },
  "mte3": {
    "description": "UB → HBM (Vector output)",
    "src_space": "ub",
    "dst_space": "hbm",
    "bandwidth_gbps": 200
  }
}
```

**910B 数据流:**

```
Cube 数据流:
  HBM ──MTE2──▶ L1 ──MTE1──▶ L0A ──┐
                             L0B ──┼──▶ Cube ──▶ L0C ──FixPipe──▶ HBM
                                   │
Vector 数据流:
  HBM ──MTE2──▶ UB ──▶ Vector ──▶ UB ──MTE3──▶ HBM
```

### 流水线定义 (pipeline)

```json
"pipeline": {
  "cube_path": {
    "stages": ["cube_mte2", "mte1", "cube", "fixpipe"],
    "description": "Matrix multiplication data flow"
  },
  "vector_path": {
    "stages": ["vector_mte2", "vector", "mte3"],
    "description": "Vector computation data flow"
  },
  "parallelism": {
    "cube_and_vector": true,
    "description": "Cube and Vector paths can execute in parallel"
  }
}
```

## 创建自定义配置

1. 复制 `ascend_910b.json` 作为模板
2. 修改参数以匹配目标硬件
3. 使用 `hardware_schema.json` 验证 (可选)

```bash
# 使用 ajv 或类似工具验证
npx ajv validate -s hardware_schema.json -d my_hardware.json
```

## 性能模型公式

`performance_model` 部分定义计算周期数的公式:

```json
"performance_model": {
  "cube_cycles_per_tile": {
    "formula": "ceil(M/16) * ceil(N/16) * ceil(K/16)",
    "base_cycles": 1
  },
  "vector_cycles_per_op": {
    "formula": "ceil(elements / 128)",
    "base_cycles": 1
  },
  "memory_cycles": {
    "formula": "bytes / bandwidth_per_cycle",
    "include_latency": true
  }
}
```

## 支持的硬件类型

| 硬件 | 配置文件 | 状态 |
|------|----------|------|
| Ascend 910B | `ascend_910b.json` | ✅ 完整支持 |
| Ascend 910C | `ascend_910c.json` | 🚧 规划中 |
| 自定义 | 用户提供 | ✅ 支持 |

## API 使用 (C++)

```cpp
#include "AscendModel/HardwareConfig.h"

// 加载配置
auto config = HardwareConfig::loadFromFile("configs/ascend_910b.json");

// 查询参数
double freq = config.getClockFrequencyGHz();      // 1.85
double hbmBW = config.getMemoryBandwidth("hbm");  // 1.6 TB/s
int vectorWidth = config.getVectorWidth();        // 128 elements

// 估算周期
int cubeCycles = config.estimateCubeCycles(M, N, K);
int memCycles = config.estimateMemoryCycles("hbm", bytes);
```
