# vTriton 校准系统设计与使用指南

> 涵盖 M1 微基准校准 + Layer 0/1/2/3 工作负载校准的完整体系
>
> 更新 2026-06-23：新增 Layer 0 核心分配（CoreMapper）

---

## 一、校准系统全景

```
                             硬件真机 910B3
                                   │
                    ┌──────────────┼──────────────────┐
                    │              │                   │
                    ▼              ▼                   ▼
         ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
         │ M1 微基准校准  │  │ Layer 1/2 校准 │  │ 验证: harness.py │
         │ (孤立组件峰值)  │  │ (真实负载反馈)  │  │ T_bound vs T_real │
         └──────┬───────┘  └──────┬───────┘  └──────────────────┘
                │                 │
                ▼                 ▼
         calib_910b3_v1.json  calib_910b3_v2.json
         (16个P0常数,         (v1 + 校准后的
          n≥30, CI<5%)        交互参数)
                │                 │
                └────────┬────────┘
                         ▼
                   M4 模型层
              compute_bounds()
                   │
                   ▼
                 T_bound
```

---

## 二、M1 微基准校准（已有）

### 2.1 目的
测量 Ascend 910B3 每个硬件组件在**理想隔离条件**下的持续吞吐率上限。

### 2.2 组件清单

| 文件 | 被测组件 | 产出常数 | 单位 |
|------|----------|----------|------|
| `microbench/cube_peak_fp16.cce` | Cube FP16 | P_cube_fp16_sustained | TFLOPS/core |
| `microbench/cube_peak_int8.cce` | Cube INT8 | P_cube_int8_sustained | TFLOPS/core |
| `microbench/cube_peak_bf16.cce` | Cube BF16 | P_cube_bf16_sustained | TFLOPS/core |
| `microbench/vector_peak_elemwise_add.cce` | Vector Add | P_vector_add_sustained | GFLOPS |
| `microbench/vector_peak_elemwise_mul.cce` | Vector Mul | P_vector_mul_sustained | GFLOPS |
| `microbench/vector_peak_elemwise_max.cce` | Vector Max | P_vector_max_sustained | GFLOPS |
| `microbench/vector_peak_elemwise_min.cce` | Vector Min | P_vector_min_sustained | GFLOPS |
| `microbench/vector_peak_transcendental.cce` | Vector 超越函数 | P_vector_*_sustained | GFLOPS |
| `microbench/scalar_peak.cce` | Scalar ALU | P_scalar_add_sustained | GFLOPS |
| `microbench/mte_gm_to_ub.cce` | GM→UB | BW_gm_to_ub_sustained | GB/s |
| `microbench/mte_ub_to_gm.cce` | UB→GM | BW_ub_to_gm_sustained | GB/s |
| `microbench/mte_gm_to_l1.cce` | GM→L1 | BW_gm_to_l1_sustained | GB/s |
| `microbench/mte_l1_to_l0a.cce` | L1→L0A | BW_l1_to_l0a_sustained | GB/s |
| `microbench/mte_l0c_to_gm.cce` | L0C→GM (FixPipe) | BW_l0c_to_gm_sustained | GB/s |
| `microbench/mte_hbm_allcore.cce` | 全核 HBM | BW_hbm_allcore_sustained | GB/s |
| `microbench/mandatory_handoff.cce` | Cube→Vector 握手 | mandatory_handoff_cycles | cycles |

### 2.3 数据流

```
CCE 微核源码 (*.cce)
    │
    ▼ 编译 (ccec / ccecom)
CCE 二进制
    │
    ▼ 运行 + msprof (cce_remote_bench.py)
msprof op_summary CSV
    │
    ▼ 统计提取 (fit_constants.py)
calib_910b3_v1.json
    │
    ▼ 加载 (calib_loader.py)
CalibrationDB
    │
    ▼ 被 M4 模型层引用
compute_component_floor(extract, cube, vector, memory)
compute_bounds(grid, extract, calib_db)
```

### 2.4 使用

```bash
# 在真机 910B3 上跑全部微基准
python perfbound/calibration/scripts/cce_remote_bench.py \
  --host 910B3 \
  --n-repeat 45 \
  --output-dir perfbound/calibration/bench_output

# 提取常数
python perfbound/calibration/scripts/fit_constants.py \
  perfbound/calibration/bench_output \
  perfbound/calibration/data/calib_910b3_v1.json

# 验证
python perfbound/calibration/scripts/validate_vs_tilesim.py \
  perfbound/calibration/data/calib_910b3_v1.json
```



---

## 三、Layer 0 — 核心分布映射 (新增)

### 3.1 目的
DES 模型模拟 **一个 program block** 的管线调度。真机 msprof 上报的是
**整个 kernel launch** (如 4096 个 block) 的端到端结果。
Layer 0 按 NPU 固件的 block→核心 round-robin 分配逻辑
将 model per-block 时间缩放到 E2E, 从而使 model 和 real 的输入尺度对齐。

### 3.2 原理

```
真机 rtKernelLaunch:
  Grid = 4096 blocks,  AIC=20 cores,  AIV=40 cores
  MIX_AIC: 每 block 占用 1 AIC + 1 AIV
  bottleneck_cores = min(20, 40) = 20
  waves = ceil(4096 / 20) = 205
  E2E_wall = 205 x max(per_block_AIC, per_block_AIV)

DES 模型:
  输入: 1 block 的 HIVM MLIR (pid_x=0, pid_y=0)
  输出: AIC span (us), AIV span (us)  — from Perfetto trace

CoreMapper:
  e2e = mapper.map(grid=4096, per_block_aic, per_block_aiv, task_type="MIX_AIC")
  e2e.e2e_wall_us → 与 msprof Task Duration(us) 对比
```

### 3.3 模块

| 模块 | 文件 | 功能 |
|------|------|------|
| CoreMapper | `perfbound/distribution/core_mapper.py` | grid→core 映射 + E2E 缩放 |

### 3.4 CLI 参数

```bash
python -m perfbound.calibration.trace_calibrator \
    --aic-cores 20 --aiv-cores 40 \   # 物理核心数 (默认 20/40)
    --grid 128 32 \                    # 重写 grid (默认从 Block Num 推断)
    --real-csv ... --model-trace ...
```

### 3.5 约束与普适性

- `IS_VARLEN=False` 的 kernel: 所有 block 具有相同的 loop bounds 和操作数,
  per-block span 恒定——直接适用。
- `IS_VARLEN=True`: 不同 block 处理不同长度序列, 需多 block 采样后加权平均。
- `program_id` 只影响数据偏移量(内存地址), 不影响 DES 计算图或 op 数量。
- 对 `AI_CORE` / `AI_VECTOR_CORE` 单核模式同样适用 (CoreMapper 按 task_type 选择有效核心数)。
---

## 四、Layer 1 — 每核时间校准（新增）

### 4.1 目的
用真实 Triton 算子运行 msprof 数据，对比模型预测的 AIC/AIV 总时间和硬实测时间的差异，校准 `startup_latency` 和 `scalar_overhead_factor` 参数。

### 4.2 原理

```
真实侧:                             模型侧:
───────                             ───────
msprof op_summary.csv              Perfetto trace JSON
  │                                   │
  ▼                                   ▼
msprof_trace_extractor.py          model_trace_extractor.py
  │                                   │
  ├─ AIC tasks: Σ duration           ├─ AIC ops: Σ t_c_us
  └─ AIV tasks: Σ duration           └─ AIV ops: Σ t_c_us
                    │                         │
                    └──────────┬──────────────┘
                               ▼
                     per_core_calibrator.py
                               │
                               ▼
                     efficiency_aic = T_model / T_real
                     efficiency_aiv = T_model / T_real

调整规则:
  efficiency > 1.0 → 模型过快 → 增加 startup_latency
  efficiency < 1.0 → 模型过慢 → 减少 startup_latency (不低于 1 cycle)
```

### 4.3 模块

| 模块 | 文件 | 功能 |
|------|------|------|
| msprof_trace_extractor | `perfbound/calibration/msprof_trace_extractor.py` | 从 CSV 重建真实时间线 |
| model_trace_extractor | `perfbound/calibration/model_trace_extractor.py` | 从 Perfetto JSON 提取模型时间线 |
| per_core_calibrator | `perfbound/calibration/per_core_calibrator.py` | 计算效率比和建议调整 |

---

## 五、Layer 2 — 管道流水线校准（新增）

### 5.1 目的
对比真实硬件上 AIC 和 AIV 的实际并行执行程度和模型假设的完全并行执行，校准 `mandatory_handoff_cycles` 和 `pipe_barrier_cycles_per_iter`。

### 5.2 原理

```
从 msprof Start Time 重建时间线:
                                         时间轴 →
Core 0 (AIC):  [===Cube 60us===][FixPipe 15us][===Cube 58us===][FixPipe 14us]
Core 1 (AIV):     [=MTE3 12us=][Vector 42us][MTE2 10us]
                    ↑ 重叠区 = 并发执行 → real_overlap_ratio

模型假设:
Core 0 (AIC):  [===Cube===][FixPipe][===Cube===][FixPipe]
Core 1 (AIV):  [=MTE3=][Vector][MTE2]
               ↑ 完全并行 → model_overlap_ratio ≈ 1.0

pipeline_efficiency = real_overlap / model_overlap
  → < 1.0 表示真实硬件有同步气泡
  → 增加 handoff / barrier 周期以匹配真实行为
```

### 5.3 模块

| 模块 | 文件 | 功能 |
|------|------|------|
| pipeline_calibrator | `perfbound/calibration/pipeline_calibrator.py` | 计算重叠效率和建议调整 |

---

## 六、Layer 3 — 多算子收敛（新增）

### 6.1 目的
用多个不同负载的 Triton kernel 分别校准，检查所有内核的效率是否收敛在 [0.95, 1.05] 范围内。如果收敛，校准参数具有泛化性。

### 6.2 模块

| 模块 | 文件 | 功能 |
|------|------|------|
| trace_calibrator | `perfbound/calibration/trace_calibrator.py` | 编排整个校准流程 |
| calib_reporter | `perfbound/calibration/calib_reporter.py` | 生成报告 JSON + 文本 |

---

## 七、v1 → v2 验收矩阵

### 7.1 永远不变的 P0 组件峰值（M1 微基准产出）

这些值的 source = `"cce_microbench"`、n_runs ≥ 30、CI < 5%。校准Layer 1/2 **不改变**它们，保证 bound 的保守下界性质：

| 参数 | v1 值 | v2 值 | 原因 |
|------|-------|-------|------|
| `P_cube_fp16_sustained` | ~5.16 TFLOPS | **不变** | 硬件上限固定 |
| `P_cube_int8_sustained` | ~5.18 TFLOPS | **不变** | 硬件上限固定 |
| `P_cube_bf16_sustained` | ~5.16 TFLOPS | **不变** | 硬件上限固定 |
| `P_vector_add_sustained` | ~15.1 GFLOPS | **不变** | 硬件上限固定 |
| `P_vector_mul_sustained` | ~14.6 GFLOPS | **不变** | 硬件上限固定 |
| `P_vector_max_sustained` | ~16.2 GFLOPS | **不变** | 硬件上限固定 |
| `P_vector_min_sustained` | ~16.2 GFLOPS | **不变** | 硬件上限固定 |
| `P_vector_exp_sustained` 等 4 个超越函数 | ~3.33 GFLOPS | **不变** | 硬件上限固定 |
| `P_scalar_add_sustained` | ~0.6 GFLOPS | **不变** | 硬件上限固定 |
| `BW_gm_to_ub_sustained` | ~86.5 GB/s | **不变** | 硬件上限固定 |
| `BW_ub_to_gm_sustained` | ~86.5 GB/s | **不变** | 硬件上限固定 |
| `BW_gm_to_l1_sustained` | ~141.1 GB/s | **不变** | 硬件上限固定 |
| `BW_l1_to_l0a_sustained` | ~451.8 GB/s | **不变** | 硬件上限固定 |
| `BW_l0c_to_gm_sustained` | ~143.5 GB/s | **不变** | 硬件上限固定 |
| `BW_hbm_allcore_sustained` | 实测值 | **不变** | 硬件上限固定 |

### 7.2 会被改变的跨组件交互参数

这些值的 source = `"datasheet_seed"`（n_runs=0，未实测）。校准 Layer 1/2 将它们从"种子猜测值"升级为"基于真实负载反馈的计算值"：

| 参数 | v1 值 | v2 变化方式 | 校准来源 | 触发条件 |
|------|-------|------------|----------|----------|
| `startup_latency.cube` | 20 cycles | × AIC 效率因子 | Layer 1 | `\|AIC 效率 - 1\| > 2%` |
| `startup_latency.mte2` | 50 cycles | × AIC 效率因子 | Layer 1 | `\|AIC 效率 - 1\| > 2%` |
| `startup_latency.vector` | 35 cycles | × AIV 效率因子 | Layer 1 | `\|AIV 效率 - 1\| > 2%` |
| `startup_latency.mte3` | 40 cycles | × AIV 效率因子 | Layer 1 | `\|AIV 效率 - 1\| > 2%` |
| `scalar_overhead_factor` | 3.74 | × AIV 效率因子 | Layer 1 | `\|AIV 效率 - 1\| > 2%` |
| `mandatory_handoff_cycles` | 7621 (已有M1测量) | × (1 + gap×0.6) | Layer 2 | pipeline效率 < 0.98 |
| `pipe_barrier_cycles_per_iter` | 7500 (datasheet_seed) | × (1 + gap×0.4) | Layer 2 | pipeline效率 < 0.98 |

### 7.3 无校准数据因此不变的参数

| 参数 | v1 值 | 是否变化 | 原因 |
|------|-------|---------|------|
| `startup_latency.fixpipe` | 30 cycles | **不变** | Layer 1 无法单独拆出 FixPipe 时间 |
| `core` 所有字段 | aic=20, aiv=40, clock=1.85GHz | **不变** | 硬件拓扑不变 |
| `cube.fractal_sizes` | M16,K16,N16 等 | **不变** | 硬件特征不变 |
| `memory` 容量字段 | gm=32GB, l1=1024KB 等 | **不变** | 硬件容量不变 |

### 7.4 v2 新增的 provenance 标记

v2 的 `constants` 字段中，所有被校准的参数新增来源记录：

```json
{
  "startup_latency.cube": {
    "name": "startup_latency.cube",
    "value": 20.6,
    "unit": "cycles",
    "ci_95": 0.0,
    "source": "per_core_calibration",    // ← v1 是 "datasheet_seed"
    "n_runs": 1,                          // ← v1 是 0
    "notes": "Calibrated from chunk_kda_bwd; efficiency=0.968"
  },
  "mandatory_handoff_cycles": {
    "name": "mandatory_handoff_cycles",
    "value": 8550.0,
    "unit": "cycles",
    "ci_95": 0.0,
    "source": "pipeline_calibration",     // ← v1 是 "cce_microbench"
    "n_runs": 1,
    "notes": "Calibrated from chunk_kda_bwd pipeline overlap"
  }
}
```

v2 还会更新顶层字段：

| 字段 | v1 | v2 |
|------|----|----|
| `version` | `"v1"` | `"v2"` |
| `description` | `"Calibration database from CCE microbenchmarks"` | `"Calibrated from N kernel(s); avg AIC eff=X, avg AIV eff=Y"` |

### 7.5 效率因子计算方式

```
效率因子 = T_model / T_real        （模型效率, 由 Perfetto trace 和 msprof CSV 对比得出）

效率因子 > 1.0  → 模型过快（预测时间短于实际）→ 增加开销参数
效率因子 < 1.0  → 模型过慢（预测时间长于实际）→ 减少开销参数（但不低于 1 cycle）

校准公式: 新值 = 旧值 × 效率因子

AIC 效率影响: startup_latency.cube, startup_latency.mte2
AIV 效率影响: startup_latency.vector, startup_latency.mte3, scalar_overhead_factor

管道间隙 (pipeline_efficiency < 0.98):
  gap = 1.0 - pipeline_efficiency
  mandatory_handoff_cycles  ← 当前值 × (1 + gap × 0.6)   # handoff 是 Cube↔Vector 主瓶颈
  pipe_barrier_cycles_per_iter ← 当前值 × (1 + gap × 0.4) # barrier 是辅助同步
```

### 7.6 对比示例（假设校准）

```
假设 chunk_kda_bwd 校准结果: AIC eff=0.97, AIV eff=0.94, pipeline eff=0.86

v1                          →  v2
────────────────────────────────────────────────────────
startup_latency.cube: 20    →  19.4   (20 × 0.97)
startup_latency.mte2: 50    →  48.5   (50 × 0.97)
startup_latency.vector: 35  →  32.9   (35 × 0.94)
startup_latency.mte3: 40    →  37.6   (40 × 0.94)
scalar_overhead: 3.74       →  3.52   (3.74 × 0.94)
mandatory_handoff: 7621     →  8255   (7621 × 1.084)
pipe_barrier: 7500          →  7920   (7500 × 1.056)
startup_latency.fixpipe: 30 →  30     (不变, 无校准数据)
P_cube_fp16: 5.16           →  5.16   (不变, M1 P0)
BW_gm_to_ub: 86.5           →  86.5   (不变, M1 P0)
```

> **关键**: 效率 < 1 意味着模型比实际更慢 → 参数下调 → 变得更宽松 → **bound 仍然保守**（下界变小，但满足 ≤ 实测）。效率 > 1 意味着模型过快 → 参数上调 → bound 收紧但仍然由 M1 P0 上限保证。

---

## 八、完整使用流程

### 8.1 环境要求

- Linux x86_64 可 SSH 到 910B3 真机
- CANN 8.5.0+ 安装在 `/home/shane/Ascend` 或 `/usr/local/Ascend`
- triton-ascend Python 包已安装
- vTriton 项目已编译（`tritonsim-opt` 可用）

### 8.2 Step 1: 真机跑 chunk kernel + msprof

```bash
cd /home/triton_sim/vTriton

# 配置远程 910B3
cat > ~/.vtriton_remote << 'EOF'
[remote]
host = root@910b3-ip
path = ~/vTriton
EOF

# 跑 chunk_kda kernel 并收集 msprof CSV
python scripts/remote_bench.py \
  --kernel chunk_kda_bwd \
  --script test/chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2.py \
  --output temp/real_op_summary.csv

# 此时 temp/real_op_summary.csv 包含 msprof 输出
```

### 8.3 Step 2: 生成模型 Perfetto trace

```bash
# 先用 clean_npuir.py 清洗 HIVM MLIR
python scripts/clean_npuir.py temp/kernel.npuir.mlir temp/kernel_clean.npuir.mlir

# 用 tritonsim-opt 生成 DES + Perfetto trace
./build/bin/tritonsim-opt temp/kernel_clean.npuir.mlir \
  --allow-unregistered-dialect \
  --analyze-hivm="scheduler=des \
    hardware-config=$(pwd)/configs/ascend_910b3.json \
    des-graph-file=$(pwd)/temp/model_des.json \
    perfetto-trace-file=$(pwd)/temp/model_trace.json"
```

### 8.4 Step 3: 运行校准

```bash
# 单 kernel 校准
python perfbound/calibration/trace_calibrator.py \
  --real-csv temp/real_op_summary.csv \
  --model-trace temp/model_trace.json \
  --calib-in perfbound/calibration/data/calib_910b3_v1.json \
  --calib-out perfbound/calibration/data/calib_910b3_v2.json \
  --report temp/calibration_report.json
```

### 8.5 Step 4: 用新校准重跑验证

```bash
# 用 v2 校准重新运行分析
./build/bin/tritonsim-opt temp/kernel_clean.npuir.mlir \
  --allow-unregistered-dialect \
  --analyze-hivm="scheduler=des \
    hardware-config=$(pwd)/configs/ascend_910b3.json \
    des-graph-file=$(pwd)/temp/model_des_v2.json \
    perfetto-trace-file=$(pwd)/temp/model_trace_v2.json"

# 再次校准，检查是否收敛
python perfbound/calibration/trace_calibrator.py \
  --real-csv temp/real_op_summary.csv \
  --model-trace temp/model_trace_v2.json \
  --calib-in perfbound/calibration/data/calib_910b3_v2.json \
  --report temp/calibration_report_v2.json
```

### 8.6 多 kernel 校准

```bash
# 对多个不同的 Triton kernel 分别跑 msprof 和模型 trace，然后一起校准
python perfbound/calibration/trace_calibrator.py \
  --real-csv temp/kernel_a_msprof.csv \
  --real-csv temp/kernel_b_msprof.csv \
  --model-trace temp/kernel_a_trace.json \
  --model-trace temp/kernel_b_trace.json \
  --calib-in perfbound/calibration/data/calib_910b3_v1.json \
  --calib-out perfbound/calibration/data/calib_910b3_v3_multikernel.json \
  --report temp/calibration_report_multikernel.json
```

---

## 九、校准报告格式

`calibration_report.json` 示例：

```json
{
  "version": "v1",
  "description": "Calibration vs 1 kernel(s)",
  "kernels": ["real_op_summary"],
  "avg_aic_efficiency": 0.968,
  "avg_aiv_efficiency": 0.944,
  "avg_pipeline_efficiency": 0.863,
  "per_core_results": [
    {
      "kernel": "real_op_summary",
      "aic": {"model_us": 823.1, "real_us": 850.2, "efficiency": 0.968},
      "aiv": {"model_us": 387.5, "real_us": 410.7, "efficiency": 0.944},
      "notes": "AIC efficiency: 0.968; AIV efficiency: 0.944"
    }
  ],
  "pipeline_results": [
    {
      "kernel": "real_op_summary",
      "model_overlap": 0.95,
      "real_overlap": 0.82,
      "efficiency": 0.863
    }
  ],
  "adjustments": {
    "startup_latency.cube": 20.6,
    "startup_latency.vector": 37.1,
    "scalar_overhead_factor": 3.96,
    "mandatory_handoff_cycles": 8550.0
  },
  "converged": false
}
```

---

## 十、校准参数对照表

| 参数 | 类别 | 原始来源 | 单位 | Layer 1 校准 | Layer 2 校准 |
|------|------|----------|------|-------------|-------------|
| `P_cube_fp16_sustained` | P0 组件峰值 | M1 微基准 | TFLOPS | ❌ 不动 | ❌ 不动 |
| `P_vector_add_sustained` | P0 组件峰值 | M1 微基准 | GFLOPS | ❌ 不动 | ❌ 不动 |
| `BW_gm_to_ub_sustained` | P0 组件峰值 | M1 微基准 | GB/s | ❌ 不动 | ❌ 不动 |
| `startup_latency.cube` | 启动开销 | datasheet seed → **校准** | cycles | ✅ **调整** | ❌ |
| `startup_latency.vector` | 启动开销 | datasheet seed → **校准** | cycles | ✅ **调整** | ❌ |
| `startup_latency.mte2` | 启动开销 | datasheet seed → **校准** | cycles | ✅ **调整** | ❌ |
| `startup_latency.mte3` | 启动开销 | datasheet seed → **校准** | cycles | ✅ **调整** | ❌ |
| `scalar_overhead_factor` | 循环开销 | datasheet seed → **校准** | 倍数 | ✅ **调整** | ❌ |
| `mandatory_handoff_cycles` | 握手延迟 | M1 微基准 → **调整** | cycles | ❌ | ✅ **调整** |
| `pipe_barrier_cycles_per_iter` | 同步开销 | datasheet seed → **校准** | cycles | ❌ | ✅ **调整** |

**核心原则**：M1 微基准产出的 P0 组件峰值**永不改变**——保证 bound 的保守性。Layer 1/2 只调整"跨组件交互参数"（启动、同步、握手），这些参数在 M1 微基准中无法测量。
