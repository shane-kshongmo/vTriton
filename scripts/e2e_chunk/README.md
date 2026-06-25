# chunk_kda E2E 校准管线

## 容器

| 容器 | CANN | 用途 |
|------|------|------|
| **triton_dev** | 9.0 | DES dump + 模拟 + 校准 |
| **triton_profiling** | 8.5 | 真机 profiling + CCE 基准 |

## 校准结果

| 版本 | 方法 | chunk_kda ratio | split_qkv ratio |
|------|------|:---:|:---:|
| v1 | CCE 微基准 (吞吐峰值) | — | — |
| v2 | per-pipe 平均 multiplier (chunk profiling) | **0.50** | **0.14** |
| **v3** | **per-opcode CCE 实测 cycle** | **0.63** | **0.17** |

v3 提升: chunk +25%, split +20%. Per-opcode cycle costs from `calib_910b3_v3_opcode.json`.

### v3 详情

```
chunk_kda (MIX_AIC, 4096 blocks, 205 AIC / 103 AIV waves):
  Model AIC:   9,656 us
  Model AIV:  70,241 us ← PIPE_S scalar bottleneck (88%)
  Model E2E:  70,241 us
  Real E2E:  111,892 us
  Ratio:      0.63

split_qkv (AIV-only, 20 blocks, 1 wave, 103 tokens/block):
  Model AIV:    408 us
  Real E2E:   2,434 us
  Ratio:       0.17
```

Gap 来源: PIPE_S sync ops (barrier/wait_flag) 仍用估计值 100-500 cyc，实际可能更高。

## 文件

```
scripts/e2e_chunk/
├── README.md
├── run_profile.sh        # triton_profiling
├── run_dump_mlir.sh      # triton_dev
├── run_des_trace.sh      # triton_dev
├── run_calibration.sh    # triton_dev
├── dump/kernel.npuir.mlir
└── output/
    ├── op_summary.csv                    (msprof profiling)
    ├── chunk_des_20260624.json           (DES: duration=1)
    ├── chunk_des_20260624_v3.json        (DES: v3 per-opcode cycles)
    ├── v3_final.json                     (v3 calibration report)
    └── v3_calib_report.json

perfbound/calibration/data/
└── calib_910b3_v3_opcode.json            (v3 per-opcode cycle table)
