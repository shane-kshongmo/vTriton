# vTriton (TritonSim) 项目架构深度分析报告

> 生成日期：2026-06-09 · 最近更新：2026-06-27（至 commit `4a63cc2`：含 US-SB-006 绝对 Gap-2 闭合、US-SB-008 双限硬件验证、occupancy-aware HBM 与 vector-floor soundness 修复）

---

## 目录

1. [项目概述](#1-项目概述)
2. [总体架构](#2-总体架构)
3. [C++ MLIR Pass 管道](#3-c-mlir-pass-管道)
4. [Python 性能边界模型 (perfbound)](#4-python-性能边界模型-perfbound)
5. [CLI 工具链](#5-cli-工具链)
6. [配置与校准体系](#6-配置与校准体系)
7. [测试体系](#7-测试体系)
8. [数据流与模块交互](#8-数据流与模块交互)
9. [项目状态与路线图](#9-项目状态与路线图)
10. [关键设计决策](#10-关键设计决策)

---

## 1. 项目概述

**vTriton**（又名 **TritonSim**）是一个基于 MLIR 的 **Ascend NPU 性能建模工具**，目标硬件为华为昇腾 910B / 910B3。项目的核心目标是为 Triton 语言编写的 GPU/NPU 内核提供一个**可证明保守的执行时间下界（performance lower bound）**。

项目采用**双层架构**：

- **C++ 层**：基于 MLIR/LLVM 框架的自定义方言（AscendModel Dialect）和变换通道（Transform Passes），负责 Triton IR 解析、操作分类、周期估算和管道调度分析。
- **Python 层**（`perfbound` 包）：一个零 MLIR 依赖的纯 Python 分析性能上界模型，通过两层级（Tier-1 网格级 + Tier-2 组件级）计算内核执行时间的严格下界。

### 核心公式

```
T_bound = max(T_grid_floor, T_core_floor + T_serial_irreducible)
```

- **T_grid_floor**：芯片网格级下界（基于占据率、负载均衡、带宽/算力瓶颈）
- **T_core_floor**：单核组件级下界（基于加权调和平均的 Roofline 各组件吞吐率）
- **T_serial_irreducible**：不可消除的串行开销（跨组件数据交换的强制握手开销）

> **实现与 spec 的有意分歧（soundness fix）**：spec 文本（`performance_bound_model.md` §4.1/§7）与
> `perfbound/__init__.py` 写成加性形式 `max(T_grid, T_core) + T_serial`。该形式 **非保守**：
> `max(a,b)+c ≥ max(a, b+c)`（c≥0），可能高估下界并导致 `T_bound > T_measured`，违反 spec §4.0 的
> 保守性定理。`combine/bound_combiner.py` 实际实现为 `max(grid, core+serial)` —— 最紧的可证明下界，
> 也与 spec §4.0 散文（“+T_serial 附着于 Tier-2 项”）一致。本文档以实现为准。

---

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                          vTriton / TritonSim                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐          ┌─────────────────────────────┐   │
│  │   C++ MLIR Pipeline  │          │   Python perfbound Package  │   │
│  │   (lib/AscendModel/) │◄────────►│   (perfbound/)              │   │
│  │                      │  JSON    │                              │   │
│  │  • AscendModel IR    │  桥接    │  M1: 校准 (Calibration)      │   │
│  │  • Transform Passes  │          │  M2: DSL 提取 (Grid)        │   │
│  │  • Analysis Modules  │          │  M3: HIVM 提取 (Component)  │   │
│  │  • Hardware Config   │          │  M4: 分析模型 (Models)      │   │
│  │                      │          │  M5: 边界合并 (Combine)     │   │
│  └─────────┬────────────┘          │  M6: 验证 (Validation)      │   │
│            │                        └─────────────────────────────┘   │
│            ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │                     CLI 工具层                                │     │
│  │  • tritonsim-opt    : MLIR Pass 管线入口                      │     │
│  │  • tritonsim-hivm   : HIVM 直接分析 + Triton DSL 编译转储     │     │
│  │  • ascend-tiling-opt: Tiling 参数优化入口 (待启用)             │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │                    外部依赖                                   │     │
│  │  • LLVM/MLIR (commit b5cc222d)                               │     │
│  │  • triton-ascend (可选，提供 Triton Dialect 支持)              │     │
│  │  • AscendNPU-IR / BiShengIR (HIVM 方言支持)                   │     │
│  └─────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. C++ MLIR Pass 管道

### 3.1 自定义方言：AscendModel Dialect

位置：[include/AscendModel/IR/](file:///d:/develop/vTriton/include/AscendModel/IR/) | [lib/AscendModel/IR/](file:///d:/develop/vTriton/lib/AscendModel/IR/)

基于 MLIR TableGen 定义的自定义方言，建模昇腾 NPU 的硬件执行单元：

| 操作类别                  | 操作名称                                                | 对应硬件单元      | 说明                             |
| ------------------------- | ------------------------------------------------------- | ----------------- | -------------------------------- |
| **矩阵计算**        | `ascend.matmul`                                       | Cube (达芬奇核心) | 矩阵乘法，16x16x16 分形大小      |
| **Cube 数据搬运**   | `ascend.cube_load`                                    | CubeMTE2          | 从内存加载数据到 Cube 本地缓冲区 |
|                           | `ascend.cube_store`                                   | FixPipe           | 从 Cube 输出存储到内存           |
| **Vector 数据搬运** | `ascend.vector_load`                                  | VecMTE2           | 加载数据到 Vector 核             |
|                           | `ascend.vector_store`                                 | MTE3              | 从 Vector 核存储数据             |
| **Vector 计算**     | `ascend.add`, `.sub`, `.mul`, `.div`            | Vector            | 逐元素算术运算                   |
|                           | `ascend.exp`, `.log`, `.sqrt`, `.rsqrt`         | Vector            | 超越函数                         |
|                           | `ascend.tanh`, `.sigmoid`, `.relu`, `.gelu`     | Vector            | 激活函数                         |
|                           | `ascend.reduce_sum`, `.reduce_max`, `.reduce_min` | Vector            | 归约操作                         |
|                           | `ascend.neg`, `.abs`, `.cast`, `.broadcast`     | Vector            | 辅助操作                         |
|                           | `ascend.select`, `.max`, `.min`                   | Vector            | 比较/选择操作                    |

**接口定义**：

- `EstimateCyclesOpInterface`：提供 `getFlops()` / `getTransferBytes()` 用于周期估算
- 定义于 [AscendModelInterfaces.td](file:///d:/develop/vTriton/include/AscendModel/IR/AscendModelInterfaces.td)

### 3.2 变换通道 (Transform Passes)

位置：[lib/AscendModel/Transforms/](file:///d:/develop/vTriton/lib/AscendModel/Transforms/)

| Pass                             | 文件                                                                                                        | 功能                                                                                                                                               |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ConvertTritonToAscend**  | [ConvertTritonToAscend.cpp](file:///d:/develop/vTriton/lib/AscendModel/Transforms/ConvertTritonToAscend.cpp)   | Triton IR → AscendModel 方言转换。将 Triton 指针算术链（tt.splat→tt.make_range→tt.addptr→tt.load/store）折叠为 ascend.vector_load/vector_store |
| **ExtractTTIRInfo**        | [ExtractTTIRInfo.cpp](file:///d:/develop/vTriton/lib/AscendModel/Transforms/ExtractTTIRInfo.cpp)               | 从 TTIR 提取结构信息（网格轴、持久循环、张量形状、Cube 检测）输出 JSON                                                                             |
| **AssignOpIDs**            | [AssignOpIDs.cpp](file:///d:/develop/vTriton/lib/AscendModel/Transforms/AssignOpIDs.cpp)                       | 为操作分配唯一 ID，标注硬件单元类型                                                                                                                |
| **EstimateCycles**         | [EstimateCycles.cpp](file:///d:/develop/vTriton/lib/AscendModel/Transforms/EstimateCycles.cpp)                 | 基于 Roofline 模型估算每个操作的执行周期数                                                                                                         |
| **HIVMAnalysisPass**       | [HIVMAnalysisPass.cpp](file:///d:/develop/vTriton/lib/AscendModel/Transforms/HIVMAnalysisPass.cpp)             | 原生 HIVM 调度分析，支持 Static 和 DES 两种调度器模式                                                                                              |
| **PipelineAnalysisPass**   | [PipelineAnalysisPass.cpp](file:///d:/develop/vTriton/lib/AscendModel/Transforms/PipelineAnalysisPass.cpp)     | 管道调度分析，生成依赖图和管道执行计划                                                                                                             |
| **PerfReportPass**         | [PerfReportPass.cpp](file:///d:/develop/vTriton/lib/AscendModel/Transforms/PerfReportPass.cpp)                 | 综合性能报告生成，统计各硬件单元利用率                                                                                                             |
| **TilingOptimizationPass** | [TilingOptimizationPass.cpp](file:///d:/develop/vTriton/lib/AscendModel/Transforms/TilingOptimizationPass.cpp) | Tiling 参数优化搜索（穷举/启发式/遗传算法）                                                                                                        |
| **InsertDataTransfers**    | [InsertDataTransfers.cpp](file:///d:/develop/vTriton/lib/AscendModel/Transforms/InsertDataTransfers.cpp)       | 插入显式数据传输操作                                                                                                                               |

**Pass 注册与管道**：[PassRegistration.cpp](file:///d:/develop/vTriton/lib/AscendModel/Transforms/PassRegistration.cpp)

主管道 `ascend-perf-model` 串联以上 Pass，支持以下选项：

- `arg-bindings`：函数参数和 program_id 绑定
- `loop-trip-counts`：直接循环迭代次数覆盖
- `hardware-config`：硬件配置 JSON 路径
- `optimize-tiling`：启用 Tiling 优化搜索
- `tiling-strategy`：Tiling 搜索策略

### 3.3 分析模块 (Analysis)

位置：[lib/AscendModel/Analysis/](file:///d:/develop/vTriton/lib/AscendModel/Analysis/)

| 模块                             | 文件                                                                                                      | 功能                                                                                                                                                |
| -------------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PipelineAnalysis**       | [PipelineAnalysis.cpp](file:///d:/develop/vTriton/lib/AscendModel/Analysis/PipelineAnalysis.cpp)             | 管道调度分析核心。管理 `PipelineOp`、`HWUnitPipeline`（每硬件单元独立管道）、`DependencyGraph`（操作依赖图）、`PipelineScheduler`（调度器） |
| **HIVMAnalysis**           | [HIVMAnalysis.cpp](file:///d:/develop/vTriton/lib/AscendModel/Analysis/HIVMAnalysis.cpp)                     | HIVM (.npuir.mlir) 性能分析。解析 HIVM 操作、分配到执行管道、进行 Static/DES 调度、生成 Perfetto Trace                                              |
| **RooflineAnalysis**       | [RooflineAnalysis.cpp](file:///d:/develop/vTriton/lib/AscendModel/Analysis/RooflineAnalysis.cpp)             | Roofline 模型分析。统计总 FLOPs 和总 Bytes，计算操作强度                                                                                            |
| **MemoryTilingOptimizer**  | [MemoryTilingOptimizer.cpp](file:///d:/develop/vTriton/lib/AscendModel/Analysis/MemoryTilingOptimizer.cpp)   | 基于 Orojenesis (ISCA 2024) 方法论的内存中心 Tiling 优化。建模张量访问模式、重用机会、缓冲区容量约束，生成 Pareto 最优 (buffer_size, accesses) 曲线 |
| **HardwareConfig**         | [HardwareConfig.cpp](file:///d:/develop/vTriton/lib/AscendModel/Analysis/HardwareConfig.cpp)                 | 硬件配置管理器。从 JSON 加载硬件参数（时钟频率、内存层级、计算单元吞吐率、数据搬运带宽）                                                            |
| **UnifiedTilingCostModel** | [UnifiedTilingCostModel.cpp](file:///d:/develop/vTriton/lib/AscendModel/Analysis/UnifiedTilingCostModel.cpp) | 综合 Tiling 成本模型。结合内存访问最小化、Tile 级管道重叠、依赖感知调度三个维度                                                                     |

---

## 4. Python 性能边界模型 (perfbound)

位置：[perfbound/](file:///d:/develop/vTriton/perfbound/)

**perfbound** 是一个纯 Python 包（零 MLIR 依赖），实现两层级分析性能上界模型。模型永远不会编译或运行内核——测量数据仅通过校准 (M1) 和验证 (M6) 进入系统。

### 4.1 模块架构

```
perfbound/
├── __init__.py                    # 包入口，版本 0.1.0（两层级 bound 公式）
├── calibration/                   # M1: 硬件常数校准（微基准 + 负载校准双层）
│   ├── constants.py               #   CalibrationDB / CoreConfig / Cube/Vector/Mem 数据类
│   ├── calib_loader.py            #   校准 JSON 加载与验证（含 occupancy-aware HBM peak）
│   ├── calib_reporter.py          #   校准报告生成
│   ├── model_trace_extractor.py   #   模型侧 trace 抽取（用于负载校准对比）
│   ├── msprof_trace_extractor.py  #   msprof 真实 trace 抽取 (RealTrace)
│   ├── per_core_calibrator.py     #   Layer 1: 单核时间校准（startup/scalar 系数）
│   ├── pipeline_calibrator.py     #   Layer 2: 流水重叠校准（handoff/barrier cycles）
│   ├── trace_calibrator.py        #   Layer 0–3 编排入口（grid→core, 收敛检查）
│   ├── data/calib_910b3_v1.json   #   微基准常数（16 个 P0，n=45，CI<2.5%）
│   ├── data/calib_910b3_v2.json   #   负载校准产出（startup/scalar/overlap 调整）
│   ├── microbench/                #   AscendC (.cce) 微基准源码 + bench_launcher
│   ├── scripts/                   #   cce_remote_bench / fit_constants / validate_vs_tilesim
│   └── bench_output/              #   基准测试 CSV（带宽/cube peak/handoff/MTE/scalar/vector）
├── extract/                       # M2/M3: 信息提取层
│   ├── dsl_extractor.py           #   M2: TTIR 网格信息提取 (GridInfo)
│   ├── grid_idioms.py             #   M2: 1D/2D 常见网格模式模板
│   ├── mlir_parser.py             #   M2: C++ ExtractTTIRInfo pass 子进程封装
│   ├── hivm_extractor.py          #   M3: HIVM 组件级操作提取 (OpRecord/HandoffRecord)
│   ├── op_classifier.py           #   M3: 操作→(组件, 精度) 分类器（6 组件）
│   ├── semantic_extractor.py      #   M3: Gap 1 语义资格分析
│   ├── eligibility_oracle.py      #   M3: 硬件资格判定 (Gap 1 输入)
│   └── hivm_runner.py             #   M3: HIVM CLI 运行器
├── model/                         # M4: 分析模型 (纯函数)
│   ├── bounds.py                  #   顶层 compute_bounds() 入口 + wave 缩放 (A.5 #2b)
│   ├── grid_model.py              #   Tier 1: 网格分析模型 (GridBound)
│   ├── component_model.py         #   Tier 2: 组件分析模型 (occupancy-aware, vector floor sound)
│   ├── bandwidth.py               #   持续带宽查找（含 occupancy-aware MTE cap）
│   └── serialization.py           #   强制/可避免串行化分裂
├── distribution/                  # 单 block DES → 全 launch wall-clock 桥接
│   └── core_mapper.py             #   固件 block→core 轮询分布，waves=ceil(blocks/cores)
├── analyze/                       # 诊断分析（非 bound 组成部分）
│   ├── hivm_bottleneck_diagnosis.py  # 已抽取 DES ops + 校准库的瓶颈归因
│   └── profile_utilization.py     #   真实 component 利用率 vs 理论 floor 对比
├── combine/                       # M5: 边界合并 + 双限
│   ├── bound_combiner.py          #   T_bound = max(T_grid, T_core + T_serial)（sound 形式）
│   ├── two_limit.py               #   A.7: 双限 (T_bound_DSL / T_bound_HIVM, 已实现)
│   ├── report.py                  #   每内核文本+JSON 报告 + 五维归因
│   └── run_report.py              #   端到端入口 (report_from_npuir / report_from_desgraph)
├── experiments/                   # Stage-A/B 实验装置
│   ├── registry.py                #   内核注册表 (KernelSpec, 组 I–V, build_inputs/Model 契约)
│   ├── stage_a.py                 #   chunk_kda 等 Stage-A 固定装置与加载器
│   └── artifacts.py               #   Stage-B 结果产物 schema (.omc/research/hw_runs/stageB)
├── validate/                      # M6: 验证（驱动 remote-bench-910b3）
│   ├── harness.py                 #   三态验证 (sound/tight/counterfactual)
│   ├── counterfactual.py          #   反事实：手改 HIVM → 重编译 → 正确性 + 计时 delta
│   ├── correctness.py             #   numpy.allclose 输出等价校验
│   ├── hivm_edits.py              #   结构化 HIVM JSON 编辑原语（非 regex）
│   └── msprof_parser.py           #   msprof op_summary CSV 解析（MIX_AIC/AIV/AI_VECTOR_CORE）
└── data/                          #   共享数据
    └── bandwidth_910b3.csv
```

### 4.2 六阶段模型详解 (M1–M6)

#### M1：校准 (Calibration)

**目标**：通过 AscendC 微基准测试测量所有持续硬件速率常数。

**关键数据**（`calib_910b3_v1.json`，16 个 P0 常数，每组 45 次运行，CI < 2.5%）：

| 类别                   | 指标   | 测量值                              |
| ---------------------- | ------ | ----------------------------------- |
| Cube FP16              | 吞吐率 | ~5.16 TFLOPS/core                   |
| Cube INT8              | 吞吐率 | ~5.16 TFLOPS/core                   |
| Cube BF16              | 吞吐率 | ~5.16 TFLOPS/core                   |
| 带宽 GM→UB            | MTE    | ~87 GB/s                            |
| 带宽 UB→GM            | MTE    | ~87 GB/s                            |
| 带宽 GM→L1            | MTE    | ~141 GB/s                           |
| 带宽 L1→L0A           | MTE    | ~452 GB/s                           |
| Vector Add/Mul/Max/Min | 吞吐率 | 14.6–16.2 GFLOPS                   |
| Vector 超越函数        | 吞吐率 | 3.3 GFLOPS                          |
| 强制握手开销           | 延迟   | 7621 ± 82 cycles (~4.1us @1.85GHz) |

**核心数据结构**：

- `CalibrationConstant`：值 ± CI，来源，运行次数
- `CalibrationDB`：包含 `CoreConfig`、`CubeConfig`、`VectorConfig`、`MemHierarchy`
- 完整 JSON 序列化/反序列化支持

**两层校准**：

1. **微基准层**（`calib_910b3_v1.json`，AscendC `.cce` 源码 → 远端 910B3 执行 → `fit_constants.py` 拟合）：得到上文 16 个 P0 持续速率常数。
2. **负载校准层**（`trace_calibrator.py` 编排 4 层，产出 `calib_910b3_v2.json`）：用真实内核的 msprof trace 与模型 trace 对比，调整 `startup_latency` / `scalar_overhead_factor` / 流水重叠系数。
   - Layer 0：核分布（grid→core，单 block→E2E 缩放）
   - Layer 1：单核时间校准
   - Layer 2：流水重叠校准（handoff/barrier cycles）
   - Layer 3：多内核收敛检查

**Occupancy-aware HBM peak（US-SB-006 / 绝对 Gap-2 闭合）**：实测峰值 HBM ≈ **1167 GB/s**。MTE 容量上限取 `min(single_core_rate, peak / active_cores)` —— 占据率越高，单核可分带宽越低。Vector floor 同步改为“可达成 per-op 速率”而非聚合峰值（修复了聚合速率偏低 15.6× 的非保守缺陷）。held-out 误差约 5.4%。

#### M2：DSL 提取器 (Grid)

**目标**：从 Triton DSL / TTIR 恢复 Tier 1 网格级数量。

**核心输出** (`GridInfo`)：

- `grid_dims`：启动网格维度 (G_x, G_y, G_z)
- `total_programs`：总程序数 G
- `tile_assignment`：程序 ID → Tile 坐标映射
- `work`：每个程序的工作量（元素/操作数）
- `occupancy`：min(G, n_cores) / n_cores
- `load_balance`：mean(work) / max(work)
- `redundancy`：GM 读取放大系数
- `buffer_pressure_ok` / `divisibility_ok`：硬件合法性检查

**提取方法**：

1. **TTIR 文件路径**（优先）：通过 C++ `ExtractTTIRInfoPass` 子进程解析 MLIR AST
2. **Triton Python 源码**：使用 `grid_idioms.py` 中的 1D/2D 网格模式模板

**C++ 提取信息**：

- 网格轴（`tt.get_program_id` 出现）
- 持久循环（`scf.for` 其中 `lb == program_id`）
- Tile 形状（`tt.make_tensor_ptr` 结果类型）
- Cube 检测（`tt.dot` 存在性）

#### M3：HIVM 提取器 (Component)

**目标**：从 HIVM 编译器输出提取 Tier 2 组件级操作细节。

**核心数据结构**：

- `OpRecord`：单个操作（op_id, op_name, component, precision, pipe, bytes/flops/持续时间, repeat, mask）
- `HandoffRecord`：组件间数据传递（生产者/消费者组件，数据量，传递周期）
- `HIVMExtract`：完整提取结果（操作列表、传递列表、每个组件的精度分解）

**提取路径**：

1. **优先路径**：消费 C++ `emitDESGraph()` / `emitDependencyGraphJSON()` 输出的结构化 JSON
2. **回退路径**：直接解析 `.npuir.mlir` 文件

**操作分类器** (`op_classifier.py`)：

- 6 个 Roofline 组件：`CUBE`, `VECTOR`, `SCALAR`, `MTE_GM`, `MTE_L1`, `MTE_UB`
- HIVM 管道 → 组件映射
- 操作名模式 → 组件分类

**语义资格判定** (`eligibility_oracle.py`)：

- 判定每个操作*可能*在哪些硬件单元上运行（与 HIVM 实际分配无关）
- Gap 1 输入：`realized_assignment(op) ∉ eligibility(op)`
- 保守策略：不确定时包括更多资格单元

#### M4：分析模型

**Tier 1：网格分析模型** (`grid_model.py`)

```
T_grid_floor = total_work × redundancy / (n_cores × occupancy × load_balance × I_binding)
```

- `I_binding` 由调用方以统一单位传入（B/us 或 FLOP/us）
- `total_work` 由调用方从 M3 提取结果聚合（Σ bytes 或 Σ FLOPs）

**Tier 2：组件分析模型** (`component_model.py`)

```
I_c = Σ_p O_{c,p} / Σ_p (O_{c,p} / P_p)   [加权调和平均，Eq.4]
T_core_floor = max_c(O_c / I_c)
```

- 加权调和平均正确建模混合精度指令流：整体速率受最慢精度按其工作份额支配
- 移植自 tilesim 的 `aicore_costmodel.py` 结构（非数值）
- **Wave 缩放（A.5 #2b，`bounds.py`）**：一次 extract 只覆盖一个 program；最忙核需运行 `waves = ceil(total_programs / n_cores)` 个 program。Tier-2 `T_core_floor` 按 waves 缩放（最忙核组件工作量）；Tier-1 `total_work = total_programs × per_program_work` 天然携带 waves 因子。
- **单 block → 全 launch 桥接（`distribution/core_mapper.py`）**：建模固件 `rtKernelLaunch` 的 block→core 轮询分布，`E2E wall = waves × per_block_span`（取瓶颈核类型），对齐 triton-ascend driver 行为。

#### M5：边界合并器

**核心公式**（实现的真实形式，见 §1 soundness 说明）：

```
T_bound = max(T_grid_floor, T_core_floor + T_serial_irreducible)
```

> `bound_combiner.py` 实现的是 `max(grid, core+serial)` 而非 spec 字面的加性形式 `max(grid,core)+serial`；
> 后者非保守（可违反 `T_bound ≤ T_measured`）。`T_serial` 附着于 Tier-2 项，因握手是核内（同核 Cube↔Vector）。

**五维归因分析**（诊断输出，非边界组成部分）：

| 维度                        | 含义                            | 对应优化           |
| --------------------------- | ------------------------------- | ------------------ |
| `grid`                    | 网格分区次优（占据率/负载均衡） | 改善网格分区       |
| `gap1` (wrong_unit)       | 操作运行在次优硬件单元          | 修正 DSL 类型      |
| `gap2` (coalescing)       | MTE 小包传输/对齐浪费           | 合并传输增大包大小 |
| `gap3` (avoidable_serial) | 可通过调度/ping-pong 消除的握手 | 添加乒乓缓冲区     |
| `gap4` (intra_unit_exec)  | SIMD repeat/mask 利用率低       | 提高 SIMD 利用率   |

**输出报告**：报告由 `report.py` 生成，端到端入口在 `run_report.py`：

- `report_from_npuir(npuir_path, grid, calib_db, hardware_config)` —— 运行 `tritonsim-hivm` → `des.json` → extract → 报告
- `report_from_desgraph(des_json, grid, calib_db)` —— 直接消费已有 `des.json`（`python -m perfbound.combine.run_report`）
- 每内核文本 + JSON 报告、绑定层级/组件识别、双限差距、单一推荐优化动作
- `scripts/run_bound.py` 在此之上串联完整 Stage-B 流水线（kernel → NPUIR dump → 清洗 → DES 图 → 报告）

#### M6：验证 (Validation)

**状态**：已实现（驱动远端 910B3 硬件，非模型本身）。`validate/harness.py` 提供三态验证：

1. **Soundness**：`T_bound ≤ T_measured`（二元，必须 100% 成立）
2. **Tightness**：`T_measured / T_bound`（记录中位数）
3. **Counterfactual**：手改 HIVM（提升 repeat / 插入 ping-pong）→ bishengir 重编译 → `correctness.py` 校验输出等价（`numpy.allclose`）→ `msprof_parser.py` 解析计时 delta，确认归因量与实测改进一致

`hivm_edits.py` 对 HIVM JSON 做结构化源到源编辑（**非 regex**，遵守项目规则）。`msprof_parser.py` 按集合精确匹配 `task_type` 枚举（`AI_CORE` / `MIX_AIC` / `MIX_AIV` / `AI_VECTOR_CORE`），避免子串误分类。

### 4.3 双限分析 (A.7) — 已实现 + 硬件验证

`combine/two_limit.py` 已实现（不再存根）。`T_bound_HIVM` 通过对理想化 extract 重新计算得到（**不是**从 `T_bound_DSL` 减去 gap 值）——理想化 extract 放宽 Gap-1（错位 op 重指派到合格单元）与可避免握手（移出串行集合），保留 Gap-2/Gap-4（硬件极限）。

```
T_bound_DSL  = 常规 T_bound（bishengir 实际生成的 HIVM 结构）
T_bound_HIVM = 分析性放宽后的理想 HIVM 结构（硬件合法约束下）

compiler_headroom = T_bound_DSL − T_bound_HIVM   (≥ 0，放宽只降 floor)
author_headroom   = T_measured − T_bound_DSL       (M6 提供 T_measured 前为 None)
```

**硬件验证（US-SB-008）**：在 910B3 上以 `seeded_serial` 内核验证了 compiler-headroom（结果见 `.omc/research/hw_runs/seeded_serial/seeded_serial_two_limit_result.json` 与 `RESULTS.md`）。

---

## 5. CLI 工具链

位置：[tools/](file:///d:/develop/vTriton/tools/)

### 5.1 tritonsim-opt

入口文件：[tritonsim-opt.cpp](file:///d:/develop/vTriton/tools/tritonsim-opt/tritonsim-opt.cpp)

基于 MLIR 的 `mlir-opt` 风格工具，注册 AscendModel 方言和 Pass。

**支持的输入格式**：

- AscendModel 方言 MLIR (`.mlir`)
- Triton IR (`.ttir`) — 需启用 Triton 支持
- 通用 MLIR Generic 格式（通过 `--allow-unregistered-dialect`）

**完整管道**：

```bash
tritonsim-opt input.mlir -ascend-perf-model
```

**分步执行**：

```bash
tritonsim-opt input.mlir -assign-op-ids -estimate-cycles -analyze-pipeline -perf-report
```

### 5.2 tritonsim-hivm

入口文件：[tritonsim-hivm.cpp](file:///d:/develop/vTriton/tools/tritonsim-hivm/tritonsim-hivm.cpp)

HIVM 原生分析工具。支持两种输入模式：

1. **直接模式**：分析已导出的 `.npuir.mlir` 文件

   ```bash
   tritonsim-hivm --npuir-file kernel.npuir.mlir
   ```
2. **Triton DSL 模式**：通过 triton-ascend 的 compile-only dump 路径生成 HIVM IR 后分析

   ```bash
   tritonsim-hivm --triton-script kernel.py
   ```

**关键特性**：

- `--scheduler static|des`：调度器后端选择
- `--perfetto-trace-file`：生成 Perfetto 兼容 JSON Trace
- `--des-graph-file`：导出操作图 JSON 用于外部 DES 模拟
- `--arg-bindings`：动态值绑定（如 `arg10=128,arg9=256`）
- `--triton-ascend-arch`：指定目标架构

---

## 6. 配置与校准体系

### 6.1 硬件配置

位置：[configs/](file:///d:/develop/vTriton/configs/)

| 文件                                                                         | 说明                         |
| ---------------------------------------------------------------------------- | ---------------------------- |
| [hardware_schema.json](file:///d:/develop/vTriton/configs/hardware_schema.json) | JSON Schema 定义硬件配置格式 |
| [ascend_910b.json](file:///d:/develop/vTriton/configs/ascend_910b.json)         | 昇腾 910B 默认配置           |
| [ascend_910b3.json](file:///d:/develop/vTriton/configs/ascend_910b3.json)       | 昇腾 910B3 详细配置          |

**910B3 关键硬件参数**：

| 参数             | 值                  |
| ---------------- | ------------------- |
| 时钟频率         | 1.85 GHz            |
| 周期/us          | 1850                |
| HBM 容量         | 32 GB               |
| HBM 带宽         | 1.6 TB/s            |
| L2 缓存          | 192 MB              |
| L1 缓冲区        | 1024 KB (每 AIC)    |
| UB (统一缓冲区)  | 256 KB              |
| L0A/L0B          | 各 64 KB            |
| L0C              | 256 KB              |
| Cube FP16 TFLOPS | 320 (全芯片)        |
| Vector 宽度      | 128 元素 / 256 字节 |

### 6.2 校准流程

1. **AscendC 微基准测试**：在 [perfbound/calibration/microbench/](file:///d:/develop/vTriton/perfbound/calibration/microbench/) 中使用 `.cce` 源码
2. **远程执行**：通过 [cce_remote_bench.py](file:///d:/develop/vTriton/perfbound/calibration/scripts/cce_remote_bench.py) 在 910B3 硬件上运行
3. **拟合常量**：通过 [fit_constants.py](file:///d:/develop/vTriton/perfbound/calibration/scripts/fit_constants.py) 从测量数据得到持续速率
4. **验证**：通过 [validate_vs_tilesim.py](file:///d:/develop/vTriton/perfbound/calibration/scripts/validate_vs_tilesim.py) 与 tilesim 模型对比

---

## 7. 测试体系

### 7.1 C++ 测试（MLIR 输入）

位置：[test/](file:///d:/develop/vTriton/test/)

| 文件                                                                                            | 内容                     | 说明                                          |
| ----------------------------------------------------------------------------------------------- | ------------------------ | --------------------------------------------- |
| [ascend_ops.mlir](file:///d:/develop/vTriton/test/ascend_ops.mlir)                                 | AscendModel 方言操作测试 | 包含 matmul、vector ops、softmax 完整测试用例 |
| [softmax_ascend.mlir](file:///d:/develop/vTriton/test/softmax_ascend.mlir)                         | Softmax 管道测试         | 验证 Softmax 算子各步骤的管道建模             |
| [layernorm_ascend.mlir](file:///d:/develop/vTriton/test/layernorm_ascend.mlir)                     | LayerNorm 管道测试       | 验证 LayerNorm 的管道分析                     |
| [hivm_add_kernel.npuir.mlir](file:///d:/develop/vTriton/test/hivm_add_kernel.npuir.mlir)           | HIVM 简单加法            | 基础 HIVM IR 文件测试                         |
| [hivm_mixed_cv_kernel.npuir.mlir](file:///d:/develop/vTriton/test/hivm_mixed_cv_kernel.npuir.mlir) | HIVM 混合 Cube-Vector    | 混合 CV 操作的 HIVM IR 测试                   |
| [flash_attention.ttir](file:///d:/develop/vTriton/test/flash_attention.ttir)                       | Flash Attention TTIR     | 验证 Triton IR 输入支持                       |
| [persistent_1.ttir](file:///d:/develop/vTriton/test/persistent_1.ttir)                             | 持久内核 TTIR (小)       | 持久化内核模式测试                            |
| [persistent_21.ttir](file:///d:/develop/vTriton/test/persistent_21.ttir)                           | 持久内核 TTIR (大)       | 较大持久化内核测试                            |

### 7.2 Python 测试（perfbound）

位置：[tests/perfbound/](file:///d:/develop/vTriton/tests/perfbound/)（`conftest.py` 自动把仓库根加入 `sys.path`）

**测试覆盖**：当前 **36 个测试文件**，覆盖全部六个阶段。下表为各阶段代表性用例（非完整列举）。新增覆盖包括双限 (`test_two_limit.py` / `test_seeded_serial_two_limit_hardware.py`)、绝对 Gap-2 (`test_gap2_absolute_validation.py`)、负载校准 (`test_calibration_*.py`)、Stage-B 故事 (`test_stage_b_*.py`)、远端基准 (`test_remote_bench.py`)、反事实 (`test_counterfactual*.py`) 与组件模型 (`test_component_model.py`)。

> 部分测试为**硬件门控**（需 910B3 + msprof/bishengir）：`test_*_hardware.py`、`test_remote_bench.py`、`test_hivm_cli_integration.py`、`test_stage_a_pipeline.py`（依赖 chunk_kda 固定装置）。无硬件时这些用例跳过，不应视为失败。运行：`python -m pytest tests/perfbound/`。

| 测试文件                           | 覆盖模块                           |
| ---------------------------------- | ---------------------------------- |
| `test_calibration_load.py`       | M1: 校准 JSON 加载/验证            |
| `test_calibration_extraction.py` | M1: 校准数据提取                   |
| `test_calibration_wiring.py`     | M1: 校准数据连线                   |
| `test_microbench_sources.py`     | M1: 微基准测试源文件存在性         |
| `test_dsl_extractor.py`          | M2: DSL 网格提取器 (10 个参考内核) |
| `test_grid_idioms.py`            | M2: 网格模式模板                   |
| `test_mlir_parser.py`            | M2: C++ MLIR 解析器封装            |
| `test_hivm_extractor.py`         | M3: HIVM 提取器                    |
| `test_eligibility_oracle.py`     | M3: 资格判定                       |
| `test_component_model.py`        | M4: 组件分析模型                   |
| `test_grid_model.py`             | M4: 网格分析模型                   |
| `test_bounds.py`                 | M4: 顶层边界计算                   |
| `test_serialization.py`          | M4: 强制/可避免握手分裂            |
| `test_hivm_cli_integration.py`   | HIVM CLI 集成                      |

### 7.3 冒烟测试

| 脚本                                                                                    | 说明                               |
| --------------------------------------------------------------------------------------- | ---------------------------------- |
| [triton_smoke.py](file:///d:/develop/vTriton/test/triton_smoke.py)                         | Python 冒烟测试（Triton DSL 流程） |
| [triton_hivm_launch_smoke.py](file:///d:/develop/vTriton/test/triton_hivm_launch_smoke.py) | HIVM 启动冒烟测试                  |

---

## 8. 数据流与模块交互

### 8.1 完整建模管道

```
                     ┌─────────────────┐
                     │  Triton Kernel   │
                     │  (.py / .ttir)   │
                     └────────┬────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │ tritonsim-opt│ │ tritonsim-   │ │perfbound DSL │
     │ (MLIR Passes)│ │ hivm         │ │ Extractor (M2)│
     └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
            │                │                │
            ▼                ▼                ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │Cycle Estimates│ │ HIVM Sched  │ │  GridInfo    │
     │+ Pipe Analysis│ │ + Perfetto  │ │(occupancy, LB)│
     └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
            │                │                │
            │                ▼                │
            │       ┌──────────────────┐      │
            │       │ DES Graph JSON   │      │
            │       │ Ops + Deps JSON  │      │
            │       └────────┬─────────┘      │
            │                │                │
            │                ▼                │
            │       ┌──────────────────┐      │
            │       │HIVM Extractor(M3)│◄─────┘
            │       │ OpRecord +       │
            │       │ HandoffRecord    │
            │       └────────┬─────────┘
            │                │
            │                ▼
            │       ┌──────────────────┐
            │       │   Calibration    │
            │       │   DB (M1)        │
            │       └────────┬─────────┘
            │                │
            │                ▼
            │       ┌──────────────────┐
            │       │ Models (M4)      │
            │       │ Grid + Component │
            │       └────────┬─────────┘
            │                │
            │                ▼
            │       ┌──────────────────┐
            │       │ Combiner (M5)    │
            │       │ T_bound +        │
            │       │ 5-way Attribution│
            │       └────────┬─────────┘
            │                │
            │                ▼
            └───────►┌──────────────────┐
                     │ Final Report     │
                     │ (Text + JSON)    │
                     └──────────────────┘
```

### 8.2 C++ → Python 桥接

C++ 和 Python 层通过 **JSON 序列化** 进行通信：

1. **TTIR 信息提取**：`ExtractTTIRInfoPass` → JSON → `mlir_parser.py` → `GridInfo`
2. **HIVM 操作图**：`emitDESGraph()` → JSON → `hivm_extractor.py` → `OpRecord[]`
3. **依赖图**：`emitDependencyGraphJSON()` → JSON → `hivm_extractor.py` → 依赖关系

### 8.3 硬件单元映射

昇腾 910B 的硬件执行单元与建模组件对应关系：

```
                    ┌──────────────────────────────────────┐
                    │          Ascend 910B DaVinci         │
                    │                                      │
                    │  ┌─────────┐     ┌─────────┐         │
                    │  │  Cube   │     │ Vector  │         │
                    │  │(Systolic│     │ (SIMD)  │         │
                    │  │ Array)  │     │         │         │
                    │  └────┬────┘     └────┬────┘         │
                    │       │              │               │
                    │  ┌────┴────┐    ┌────┴────┐          │
                    │  │ L0A L0B │    │   UB    │          │
                    │  │L0C(Acc) │    │ (256KB) │          │
                    │  └────┬────┘    └────┬────┘          │
                    │       │              │               │
                    │  ┌────┴──────────────┴────┐          │
                    │  │    L1 Buffer (1MB)     │          │
                    │  └───────────┬────────────┘          │
                    │              │                        │
                    │  ┌───────────┴───────────┐            │
                    │  │  MTE2 (Load)          │            │
                    │  │  MTE3 (Store)         │            │
                    │  │  FixPipe (Cube Store) │            │
                    │  └───────────┬───────────┘            │
                    │              │                        │
                    │  ┌───────────┴───────────┐            │
                    │  │  L2 Cache (192 MB)    │            │
                    │  └───────────┬───────────┘            │
                    │              │                        │
                    └──────────────┼────────────────────────┘
                                   │
                          ┌────────┴────────┐
                          │   HBM (32 GB)   │
                          │   1.6 TB/s      │
                          └─────────────────┘
```

---

## 9. 项目状态与路线图

### 9.1 当前完成度

| 阶段             | 范围                                | 状态                                          |
| ---------------- | ----------------------------------- | --------------------------------------------- |
| **A.0**    | Python `perfbound/` 包脚手架      | ✅ 完成                                       |
| **A.1**    | M1 校准 — 微基准 + 负载校准双层    | ✅ 完成 (16 P0 常数; `calib_910b3_v2.json` 负载校准; occupancy-aware HBM) |
| **A.2**    | M2 DSL 提取器 — TTIR 符号仿射恢复  | ✅ 完成 (C++ pass + 参考内核)                  |
| **A.3**    | M3 HIVM 提取器 — C++ JSON 往返     | ✅ 完成 (`run_report.py` 端到端消费 `des.json`) |
| **A.4**    | M4 模型 — wave 缩放 + soundness    | ✅ 完成 (occupancy-aware HBM、vector floor sound、Gap-2 闭合 US-SB-006) |
| **A.5**    | M5 合并器 — sound 公式 + 双限连线  | ✅ 完成 (`max(grid, core+serial)`; `core_mapper` E2E 桥接) |
| **A.6**    | M6 验证框架                         | ✅ 已实现 (harness/counterfactual/correctness/msprof_parser/hivm_edits) |
| **A.7**    | 双限计算                            | ✅ 完成 + 硬件验证 (`two_limit.py`, US-SB-008 seeded_serial) |
| **A.8**    | 端到端管道验证                      | ✅ 完成 (`run_bound.py` / `run_report.py`; seeded_serial、chunk_kda) |
| **Part B** | 实验、迭代校准、论文                | 🔶 进行中 (chunk_kda 分析、calib v2 迭代)     |

### 9.2 当前阻塞项

原阻塞项均已解决：C++ JSON → Python 端到端往返（`run_report.py`）、Cube `elements==0` 回退、Gap-4 repeat/mask 连线（`op_classifier.py` 已更新）、M6 验证框架（已实现并驱动远端硬件）。

**当前重点 / 已知局限**：

| 项                                    | 说明                                                                 | 优先级 |
| ------------------------------------- | -------------------------------------------------------------------- | ------ |
| spec ↔ 实现公式对齐                   | spec 仍写加性 `max(grid,core)+serial`；实现为 sound 的 `max(grid, core+serial)`。需回写 spec | P0     |
| AIV scalar-issue 项                   | chunk_kda 实测 ~104ms 为 AIV 标量发射瓶颈；模型缺 scalar-issue 项，致 55% “余量” 实为 bound-model artifact | P1     |
| Part B：扩展内核套件与迭代校准        | 更多 Group I–V 内核、负载校准 v2 迭代、论文写作                       | P2     |

---

## 10. 关键设计决策

### 10.1 双层建模架构

项目采用 C++ MLIR Pass + Python 分析模型的双层架构，实现了关注点分离：

- **C++ 层**负责与 MLIR/LLVM 生态集成、IR 解析和操作分类
- **Python 层**负责灵活的数学模型构建、校准数据管理和报告生成
- 两层通过 JSON 序列化进行松耦合通信

### 10.2 严格保守下界

性能边界模型的核心保证是**可证明保守**（provably conservative）：

- 使用持续（sustained）速率而非峰值（peak）速率
- 校准常数带 CI（置信区间）
- 使用下界逻辑（加权调和平均 I_c 受最慢精度支配）
- T_serial 分类中使用 "宁可漏报为强制" 策略避免低估

### 10.3 双限分析（Two-Limit）

区分两种性能边界：

- **T_bound_DSL**：基于编译器实际生成的 HIVM 结构
- **T_bound_HIVM**：分析性松弛的理想 HIVM 结构（硬件合法约束下）

两者差距揭示编译器优化空间，而实测时间与 T_bound_DSL 的差距揭示内核作者的优化空间。

### 10.4 零 MLIR Python 依赖

`perfbound` 包故意设计为零 MLIR Python 绑定依赖：

- 所有 MLIR 解析通过 C++ 子进程完成
- 纯 Python 数学模型可独立运行、测试和部署
- 降低了 Python 环境的复杂度

### 10.5 Triton 指针算术折叠

`ConvertTritonToAscend` Pass 的关键设计决策：

- Triton 的指针算术链（`tt.splat → tt.make_range → tt.addptr → tt.load`）在昇腾硬件上零成本（被吸收到 DMA 描述符中）
- 整链折叠为单个 `ascend.vector_load`/`vector_store` 操作
- 大幅简化了后续的周期估算和管道分析

### 10.6 Orojenesis 方法论

内存 Tiling 优化器基于 Orojenesis (ISCA 2024) 的核心理念：

- 最小化数据移动（内存访问）作为首要目标
- 考虑每级内存的缓冲区容量约束
- 建模 Tiling 带来的数据重用机会
- 生成 Pareto 最优 (buffer_size, accesses) 曲线
- 同时支持纯 Cube、纯 Vector 和 CV 融合模式

### 10.7 构建灵活性

支持两种构建模式：

- **带 Triton 支持**：从 `thirdparty/triton-ascend` 头文件编译 Triton Dialect，无需构建完整 triton wheel
- **不带 Triton 支持**：更快的构建，仅支持 AscendModel IR 输入
- 两个模式共享相同的 LLVM/MLIR 基础设施

---

## 附录

### A. 文件统计

| 目录                     | 文件数 | 语言                   | 说明                     |
| ------------------------ | ------ | ---------------------- | ------------------------ |
| `include/AscendModel/` | 13     | C++ Headers + TableGen | 公共头文件和方言定义     |
| `lib/AscendModel/`     | 14     | C++                    | 核心实现                 |
| `tools/`               | 3      | C++ + Python           | CLI 入口点               |
| `perfbound/`           | 23+    | Python                 | 性能边界模型包           |
| `tests/perfbound/`     | 14     | Python                 | Python 测试套件          |
| `test/`                | 9      | MLIR/TTIR/Python       | C++ 测试输入和冒烟测试   |
| `configs/`             | 3      | JSON                   | 硬件配置文件             |
| `scripts/`             | 4      | Shell/Python           | 构建和辅助脚本           |
| `patches/`             | 3      | Patch                  | triton-ascend 兼容性补丁 |

### B. 关键外部依赖

| 依赖          | 版本/Commit                      | 说明                         |
| ------------- | -------------------------------- | ---------------------------- |
| LLVM/MLIR     | `b5cc222d`                     | 从源码构建                   |
| triton-ascend | gitcode.com/Ascend/triton-ascend | 可选，提供 Triton Dialect    |
| AscendNPU-IR  | Ascend NPU IR submodule          | HIVM 方言支持                |
| CMake         | ≥ 3.20                          | 构建系统                     |
| Python        | 3.8+                             | perfbound 包                 |
| CANN          | 8.3.RC1+                         | 昇腾运行时（仅硬件执行需要） |

### C. 术语表

| 术语    | 全称                             | 说明                                         |
| ------- | -------------------------------- | -------------------------------------------- |
| AIC     | AI Core                          | 昇腾 NPU 的 Cube 计算核心 (910B3 有 20 个)   |
| AIV     | AI Vector Core                   | 昇腾 NPU 的 Vector 计算核心 (910B3 有 40 个) |
| MTE     | Memory Transfer Engine           | 数据搬运引擎 (MTE1/MTE2/MTE3)                |
| HIVM    | Heterogeneous IR Virtual Machine | 昇腾异构 IR 虚拟机 / BiShengIR               |
| TTIR    | Triton IR                        | Triton 编译器的中间表示                      |
| UB      | Unified Buffer                   | 统一缓冲区 (Vector 核片上内存)               |
| L1      | Level 1 Buffer                   | Cube 数据暂存缓冲区                          |
| L0A/B/C | Level 0 Buffer                   | Cube 寄存器文件 (A/B输入, C输出)             |
| HBM     | High Bandwidth Memory            | 高带宽内存 (片外)                            |
| DES     | Discrete Event Simulation        | 离散事件模拟                                 |
| CI      | Confidence Interval              | 置信区间                                     |
| CV      | Cube-Vector                      | Cube+Vector 融合模式                         |

---

> 本报告基于 2026-06-09 的 `HEAD` 初版生成，并于 2026-06-27 更新至 commit `4a63cc2`（US-SB-006 绝对 Gap-2 闭合、US-SB-008 双限硬件验证、负载校准模块、occupancy-aware HBM 与 vector-floor soundness 修复）。A.0–A.8 已完成，Part B（实验 / 迭代校准 / 论文）进行中。
