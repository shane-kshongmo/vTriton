# profile_utilization 输入变量来源表

“当前是否可获得”按目前可用输入链路综合判断：`op_summary`、`tritonsim-hivm --des-graph-file` 导出的 DES JSON、`CalibrationDB` 和人工配置都算来源。当前 DES JSON 已导出顶层 `clock_ghz`，以及每个 op 的 `id`、`name`、`pipe`、`duration`、`depends_on`、`bytes`、`elements`、`loop_multiplier`、`src_space`、`dst_space`、`elem_type` 等字段；它不包含 profiling 真实耗时，也不包含 `flops` 字段。

| 基础变量 | 来源 | 当前是否可获得 |
| --- | --- | --- |
| `profile.kernel_name` | `op_summary` 的 `Op Name`；DES graph 目前没有 kernel/function 名，最多只能从文件名约定推断 | 是 |
| `profile.elapsed_time_us` | `op_summary` 的 `Task Duration(us)`；DES graph 有 `clock_ghz` 和 op `duration` cycles，只能估算模型时间，不能替代 profiling 真实 elapsed time | 是 |
| `profile.components[].component` | `HIVMExtract.operations[].component`；DES graph 的 `operations[].pipe` 可通过 `PIPE_TO_COMPONENT` 映射到 `cube`、`vector`、`scalar`、`mte_gm`、`mte_l1`、`mte_ub`；也可根据 `op_summary` 中哪些 component 时间字段非 0 粗略创建 | 是 |
| `profile.components[cube].active_time_us` | `op_summary` 的 `aic_mac_time(us)`；DES graph 可按 `pipe == "Cube"` 聚合模型 busy cycles 并用 `clock_ghz` 换算，但这不是 profiling active time | 是 |
| `profile.components[vector].active_time_us` | `op_summary` 的 `aiv_vec_time(us)`；DES graph 可按 `pipe == "Vector"` 聚合模型 busy cycles 并用 `clock_ghz` 换算，但这不是 profiling active time | 是 |
| `profile.components[scalar].active_time_us` | `op_summary` 的 `aic_scalar_time(us) + aiv_scalar_time(us)`；DES graph 可按 `pipe == "Scalar"` 聚合模型 busy cycles 并用 `clock_ghz` 换算，但这不是 profiling active time | 是 |
| `profile.components[mte_l1].active_time_us` | `op_summary` 的 `aic_mte1_time(us)`；DES graph 可按 `pipe == "MTE1"` 聚合模型 busy cycles 并用 `clock_ghz` 换算，但这不是 profiling active time | 是 |
| `profile.components[mte_gm].active_time_us` | `op_summary` 的 `aic_mte2_time(us) + aiv_mte2_time(us)`；DES graph 可按 `pipe == "CubeMTE2"` 或 `pipe == "VectorMTE2"` 聚合模型 busy cycles 并用 `clock_ghz` 换算，但这不是 profiling active time | 是 |
| `profile.components[mte_ub].active_time_us` | `op_summary` 的 `aic_fixpipe_time(us) + aiv_mte3_time(us)`；DES graph 可按 `pipe == "FixPipe"` 或 `pipe == "MTE3"` 聚合模型 busy cycles 并用 `clock_ghz` 换算，但这不是 profiling active time | 是 |
| `profile.components[compute].work_done` | `HIVMExtract.operations`，按 component 聚合 `op.elements * op.loop_multiplier`；DES graph 有 `elements` 和 `loop_multiplier`；如果后续 DES graph 提供 `flops`，也可改用 `op.flops * op.loop_multiplier`，但当前没有 `flops` 字段 | 是，按 `elements` 口径 |
| `profile.components[mte].work_done` | `HIVMExtract.operations`，按 component 聚合 `op.bytes_transferred * op.loop_multiplier`；DES graph 有 `bytes` 和 `loop_multiplier` | 是 |
| `profile.components[compute].work_breakdown[].label` | `HIVMExtract.operations[].precision`，例如 `fp16`、`bf16`、`fp32`、`int8`；DES graph 有 `elem_type`，可通过 `ELEM_TYPE_TO_PRECISION` 映射到 precision | 是；未知类型会落到 `unknown`/空 precision |
| `profile.components[compute].work_breakdown[].work` | `HIVMExtract.operations`，按 `component + precision` 聚合 `op.elements * op.loop_multiplier`；DES graph 有 `elements`、`elem_type` 和 `loop_multiplier`；如果后续 DES graph 提供 `flops`，也可改用 `op.flops * op.loop_multiplier` | 是，按 `elements` 口径 |
| `profile.components[compute].work_breakdown[].peak_rate` | 来自 `CalibrationDB`；Cube 用 `calib_910b3_v1.json` 的 `cube.throughput[precision] * 1e6`；Vector 用 `vector.throughput_*_tflops * 1e6` 或 `constants.P_vector_*_sustained * 1000` | 是 |
| `profile.components[mte].work_breakdown[].label` | `HIVMExtract.operations[].src_space + "->" + op.dst_space`；DES graph 有 `src_space` 和 `dst_space`；如果没有 path 信息，可先按 component 粗略映射 | 是；空 path 时需 fallback |
| `profile.components[mte].work_breakdown[].work` | `HIVMExtract.operations`，按 `component + transfer path` 聚合 `op.bytes_transferred * op.loop_multiplier`；DES graph 有 `bytes`、`src_space`、`dst_space` 和 `loop_multiplier` | 是 |
| `profile.components[mte].work_breakdown[].peak_rate` | 来自 `perfbound/calibration/data/bandwidth_910b3.csv`，按 path 查带宽，`GB/s * 1000 = bytes/us` | 是 |
| `component_bound` | 来自 `component_model.py`，由 `compute_component_floor_from_db(HIVMExtract, CalibrationDB)` 计算；DES graph 可提供 `HIVMExtract` 的 operation 输入，还需要 `CalibrationDB` | 是 |
| `work_tolerance` | 人工配置，默认 `0.10` | 是 |
| `u_threshold` | 人工配置，默认 `0.80` | 是 |
| `r_threshold` | 人工配置，默认 `0.50` | 是 |

## 当前 chunk kernel 可从 op_summary 直接得到的值

| 基础变量 | 来源 |
| --- | --- |
| `profile.kernel_name` | `chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2` |
| `profile.elapsed_time_us` | `104311.384` |
| `profile.components[cube].active_time_us` | `586.41` |
| `profile.components[vector].active_time_us` | `4598.194` |
| `profile.components[scalar].active_time_us` | `7619.85 + 87974.877 = 95594.727` |
| `profile.components[mte_l1].active_time_us` | `798.081` |
| `profile.components[mte_gm].active_time_us` | `2744.142 + 1975.392 = 4719.534` |
| `profile.components[mte_ub].active_time_us` | `2147.593 + 1099.592 = 3247.185` |
