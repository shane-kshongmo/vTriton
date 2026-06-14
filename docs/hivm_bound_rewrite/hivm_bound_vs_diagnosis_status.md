# 现状分析：基于 HIVM 的「bound」与「诊断」是否已具备

> 分支：`integrate/pr20-pr17`（已 cherry-pick PR#20 `HivmOpsEditor` + PR#17 `HIVMBottleneckDiagnosis`）。
> 问题：**目前基于 HIVM 分析出 bound 的逻辑是否已经有了？**
> 一句话结论：**「基于 HIVM 算解析 T_bound」的逻辑早就有了——在 Python `perfbound` 侧；
> 刚合入的 PR#17 加的是「瓶颈诊断 + 优化建议」，不是 bound。两者目前彼此独立、未打通。**

把三件事分清，因为它们在仓库里是**三套独立**的东西：

| 层 | 是什么 | 在哪 | 产出 | 状态 |
|----|--------|------|------|------|
| **A. 解析 T_bound** | 两层上界模型（spec） | Python `perfbound/` | `BoundResult.t_bound_us` + 5 轴归因 + 双下界 | ✅ 已有，端到端接通，有测试 |
| **B. 瓶颈诊断** | 从 HIVM 调度结果做根因分类 + 建议 | C++（PR#17） | `HIVMBottleneckReport`（6 类 + suggestions） | ✅ 新合入，已接 CLI；**非 bound** |
| **B′. 瓶颈诊断（实测口径）** | 同一套 5 类分类，从 realized A/I/U/R/E | Python `profile_utilization` | `OperatorBottleneckReport` | ✅ 已有（与 B 重复） |

---

## A. 真正的解析 T_bound —— ✅ 已存在（Python `perfbound`），已接通且有测试

消费的是 C++ 导出的 **HIVM DES-graph JSON**（schema `a3_hivm_des_v1`），不是裸 IR。

### 端到端链路（含 file:line 证据）

```
.npuir.mlir
 └─ extract_from_npuir()                         perfbound/extract/hivm_runner.py:41
      └─ subprocess: tritonsim-hivm --npuir-file … --des-graph-file des.json
      └─ extract_hivm(des.json) → HIVMExtract     perfbound/extract/hivm_extractor.py
 └─ compute_bounds(grid_info, extract, calib_db)  perfbound/model/bounds.py:44
      ├─ compute_component_floor → ComponentBound   (Tier2：调和均值 I_c、t_core_floor)
      │                                            perfbound/model/component_model.py:164
      ├─ compute_grid_floor      → GridBound        (Tier1：occupancy × load_balance)
      │                                            perfbound/model/grid_model.py:47
      └─ classify_handoffs       → SerializationSplit (强制/可避免 serial 拆分)
      │                                            perfbound/model/serialization.py
 └─ combine(grid, comp, serial)                   perfbound/combine/bound_combiner.py:114
      → BoundResult.t_bound_us = max(t_grid_floor, t_core_floor + t_serial_irreducible)
        + Attribution(grid + gap1..gap4)          perfbound/combine/bound_combiner.py:45
 └─ compute_two_limit                             perfbound/combine/two_limit.py:161
      → TwoLimitResult(t_bound_hivm, t_bound_dsl, t_measured) + headroom
                                                   perfbound/combine/two_limit.py:35
```

一键入口：`bound_from_extract()`（`perfbound/combine/bound_combiner.py:201`），自动加载
默认 910B3 校准（`load_default_calib_db`），从一个 `HIVMExtract` 直接得到 `BoundResult`。

### 它确实产出 bound 的证据

- `T_bound = max(T_grid_floor, T_core_floor + T_serial_irreducible)`：
  `bound_combiner.py:153-154`（注意是 `max(a, b+c)` 而非 `max(a,b)+c`，A.5 守恒性修正）。
- 两层来源：Tier1 grid floor + Tier2 component floor（调和均值 ideal `I_c`）+ 强制 serial。
- 双下界：`T_bound_HIVM` vs `T_bound_DSL`，差值 = 编译器 headroom（`two_limit.py:45-52`）。

### 覆盖测试

`tests/perfbound/test_bounds.py`、`test_combine.py`、`test_two_limit.py`、
`test_calibration_wiring.py`、`test_component_model.py`。

> 这部分是 master 原有的，**不是**本次 cherry-pick 带来的。

---

## B. PR#17 的 C++ `HIVMBottleneckDiagnoser` —— 是「诊断」，不是「bound」

### 接入点（已 wire 进 CLI）

```
HIVMAnalysis.cpp:3295   HIVMBottleneckDiagnoser diagnoser(config);
HIVMAnalysis.cpp:3296   diagnoser.diagnose(report, report.bottleneckReport);
tritonsim-hivm.cpp:606  report.bottleneckReport.print(llvm::outs(), getHardwareConfig());
```

### 它算什么

- **每 op 的 `theoreticalMinCycles`**（局部弱下界）+ `overheadRatio`：
  `HIVMBottleneckDiagnosis.cpp:52` `computeTheoreticalMin()`——
  传输 op = `ceil(bytes / 带宽)`；Vector = `startup + ceil(elements / 向量宽)`；Cube/FixPipe 类似。
- **6 类根因分类** `BottleneckType`（`HIVMBottleneckDiagnosis.h:27`）：
  `BandwidthBound, ComputeBound, StartupOverhead, SyncOverhead, PipelineImbalance, LowParallelism`。
  逐 op：`diagnoseOp`（`.cpp:94`）；流水：`diagnosePipeline`（`.cpp:267`，imbalanceRatio>3 判
  PipelineImbalance）；全局：`diagnoseGlobal`。
- **文字 suggestions** + `syncOverheadRatio` / `barrierOverheadRatio`（`HIVMBottleneckReport`，`.h:59`）。

### 它**不**算什么（= 它不是 bound 的原因）

- ❌ 无 kernel 级 `max(T_grid, T_core + T_serial)`；
- ❌ 无 Tier-1 grid（occupancy / load_balance）；
- ❌ 无 component 级调和均值 `I_c`、无按 component 聚合对比 sustained 速率
  （per-op `theoreticalMinCycles` 之和 ≠ component floor——没有重叠/并行模型）；
- ❌ 无 强制/可避免 serial 拆分；
- ❌ 无 HIVM-vs-DSL 双下界 / headroom。

一句话：**PR#17 = 「IR 时刻、从 HIVM 调度结果直接做根因分类 + 给优化建议」的诊断器。**

---

## C. 关键发现：重复 + 断裂

### C.1 重复（两套 5 类分类）

| | 输入口径 | 实现 |
|---|---------|------|
| **B（C++，PR#17）** | 分析调度得到的 theoMin / overhead | `HIVMBottleneckDiagnoser::diagnose` |
| **B′（Python）** | 实测 / realized 的 A/I/U/R/E | `analyze_operator_bottleneck`（`profile_utilization.py:235`） |

**同一套 5 类瓶颈分类的两个实现**（B 多一个 `LowParallelism`；B′ 另有「暴露控制/同步赤字」A.8 量化）。
两边各活各的，口径未统一。

### C.2 断裂（诊断 ↔ bound 未连）

- PR#17 的 C++ 诊断器**不**喂给 Python 的 bound；
- Python 的 bound **不**读 PR#17 的诊断；
- C++ 这边能出诊断但出不了 `T_bound`；Python 那边能出 `T_bound` 但跑在另一条 subprocess 链上。

### C.3 与优化手册的对应关系

PR#17 `diagnoseOp` 里的 suggestion 文字——
- "Replace PIPE_ALL barrier with per-pipe set_flag/wait_flag"（`.cpp:131`）
- "Increase tile size to amortize startup latency"（`.cpp:162/212`）
- "Increase K tile dimension to improve arithmetic intensity"（`.cpp:227`）
- "Consider Cube-Vector split to exploit dual-core parallelism"（`.cpp:323`）

**逐条就是** `hivm_optimization_playbook.md` §3–§7 的 21 条手段。三者的分工因此清晰：

```
PR#17  HIVMBottleneckDiagnoser   = 检测 + 建议引擎（C++，IR 时刻）
playbook hivm_optimization        = HIVM 重写规范（手段 → before/after IR）
perfbound bound_from_extract       = 量化 bound（改写前后各算一次，验证收敛）
            ▲ 缺：执行器（把建议真正改成优化后 HIVM 的 edit 原语） + 三者串联
```

---

## D. 结论与下一步

**结论**：
1. 「基于 HIVM 出解析 bound」——**已经有了**，是 Python `perfbound`（`bound_from_extract` /
   `compute_bounds` / `combine` / `two_limit`），master 原有、有测试。
2. PR#17 的价值是**补上了 C++ 侧 IR 时刻的根因诊断 + 优化建议**，对接优化目标，但**不产出 bound 数值**。
3. 现状是「诊断（B/B′）」与「bound（A）」两套独立体系，且 B 与 B′ 口径重复。

**若要做「诊断 → bound → 输出优化后 HIVM」闭环，建议顺序**：
1. **统一口径**：让 PR#17 的 `BottleneckType` 对齐 Python 五轴 gap（grid + gap1–4），避免两套标准。
2. **建执行器**：实现 `playbook` 里待建的 HIVM edit 原语（`barrier_to_p2p`、`reorder_independent`、
   `prefetch_to_l1`、`enlarge_tile`、`cube_vector_split`），对齐现有 `perfbound/validate/hivm_edits.py`
   的 source-to-source + no-op guard 风格。
3. **闭环验证**：改写前后各跑一次 `bound_from_extract` 比较 `T_bound`——
   (A) 缩小 gap 类应保持 `T_bound` 不变、关键路径下降；(B) 降低 bound 类应 `T_bound` 下移且守住
   `T_bound ≤ T_measured`（见 `playbook` §8 判读规则）。

---

## 附：核心证据索引（file:line）

| 主题 | 位置 |
|------|------|
| T_bound 公式 `max(grid, core+serial)` | `perfbound/combine/bound_combiner.py:153` |
| 一键入口 `bound_from_extract` | `perfbound/combine/bound_combiner.py:201` |
| 两层 bound 组装 `compute_bounds` | `perfbound/model/bounds.py:44` |
| Tier2 component floor / I_c | `perfbound/model/component_model.py:164` |
| Tier1 grid floor | `perfbound/model/grid_model.py:47` |
| 双下界 / headroom | `perfbound/combine/two_limit.py:35,161` |
| HIVM→extract runner | `perfbound/extract/hivm_runner.py:41` |
| PR#17 诊断接入 | `lib/AscendModel/Analysis/HIVMAnalysis.cpp:3295` |
| PR#17 诊断打印 | `tools/tritonsim-hivm/tritonsim-hivm.cpp:606` |
| PR#17 per-op theoMin | `lib/AscendModel/Analysis/HIVMBottleneckDiagnosis.cpp:52` |
| PR#17 BottleneckType（5+1） | `include/AscendModel/Analysis/HIVMBottleneckDiagnosis.h:27` |
| Python 同款诊断（重复） | `perfbound/analyze/profile_utilization.py:235` |
| 优化手段 ↔ HIVM 重写 | `hivm_optimization_playbook.md`（同目录） |
