# split_qkv E2E 管线

**split_qkv_rmsnorm_mrope** — vllm-ascend 算子 (AIV-only SIMD)

## 容器分工

| 容器 | CANN | 用途 |
|------|------|------|
| **triton_dev** | 9.0 | 主要开发容器：编译 dump、DES 模拟、校准比对 |
| **triton_profiling** | 8.5 | 真机 profiling 专用：采集 msprof 数据 |

## 流程

```
Step 0 (profiling)     →  Step 1 (dev)        →  Step 2 (dev)        →  Step 3 (dev)
run_profile.sh           run_dump_mlir.sh       run_des_trace.sh       run_calibration.sh
(triton_profiling)       (triton_dev)           (triton_dev)           (triton_dev)
     ↓                       ↓                     ↓                     ↓
 op_summary.csv         kernel.npuir.mlir      chunk_des*.json         calib_report*.txt
  (msprof 采集)          (574 lines HIVM)      chunk_trace*.json       calib_910b3_v2.json
```

## 快速开始

```bash
docker exec triton_profiling bash /home/triton_sim/vTriton/scripts/e2e_split_qkv/run_profile.sh    # Step 0
docker exec triton_dev bash /home/triton_sim/vTriton/scripts/e2e_split_qkv/run_dump_mlir.sh          # Step 1
docker exec triton_dev bash /home/triton_sim/vTriton/scripts/e2e_split_qkv/run_des_trace.sh          # Step 2
docker exec triton_dev bash /home/triton_sim/vTriton/scripts/e2e_split_qkv/run_calibration.sh        # Step 3
```

## 校准结果

```
Real E2E wall:    2,434 us
Model E2E wall:     340 us  (ratio = 0.14)
Model per-token:   2.8 us
Model per-block:   285 us  (103 tokens × 2.8 us/token)

Grid:    20 blocks, 40 AIV cores, 1 wave
Tokens:  103 per block, 2048 total
```

## 跨算子分析 (vs chunk_kda)

| | chunk_kda | split_qkv |
|---|---|---|
| 类型 | MIX_AIC (AIC+AIV) | AIV-only |
| Grid | 4096 (128×32) | 20 (1D) |
| E2E model/real | **0.50** | **0.14** |
| 模型/真机差距 | 2× | 7× |
| 瓶颈 | AIV scalar (84% wall) | token loop × per-op error |

**E2E ratio 差异来自 kernel 结构，不是硬件不一致**：
- chunk_kda: 并行度高 (4096 blocks)，scalar 瓶颈在 AIV wave 级暴露
- split_qkv: 仅 1 wave (20 blocks)，per-token 循环放大模型误差 103×
- 两台算子的 DES 建模逻辑正确，差异是 DES 每 op=1 cycle 的简化假设在不同 kernel 结构下表现不同

## 硬件不变校准参数

从 chunk_kda profiling 提取，跨 kernel 验证适用。详见 `perfbound/calibration/data/calib_910b3_hardware.json`。

| Pipe | 真实 cycles/DES cycle |
|------|:---:|
| AIC MAC | 21.0 |
| AIV Vector | 2.4 |
| AIV Scalar | 3.9 |
| AIV MTE3 | 1.6 |

## 分析

```
Kernel: AIV-only (func_core_type=AIV, mix_mode=aiv)
Grid:   20 blocks × 103 tokens/block
Cores:  40 AIV, waves=1
DES ops: 996/block/token, mostly scalar (67%)
```
