# Bound-Guided HIVM 改写 — 文档集

本目录汇总「基于 bound 分析结论，对 HIVM 进行自动改写」这一特性的全部设计文档。
三份文档按「分析 → 现状 → 实现计划」递进，建议依此顺序阅读：

| # | 文档 | 作用 | 读者 |
|---|------|------|------|
| 1 | [hivm_optimization_playbook.md](hivm_optimization_playbook.md) | **优化手册**：5 类性能问题、21 条手段逐条落到 HIVM 层（检测信号 → HIVM 变换 → before/after IR → 对模型量的影响 → 落地与合法性），含手段 ↔ 五轴 gap ↔ `HivmOpsEditor` 原语映射 | 想知道「每条优化怎么改 HIVM」 |
| 2 | [hivm_bound_vs_diagnosis_status.md](hivm_bound_vs_diagnosis_status.md) | **现状分析**：解析 `T_bound`（Python `perfbound`）与瓶颈诊断（C++ `HIVMBottleneckDiagnoser`）的能力边界、重复与断裂，带 file:line 证据 | 想知道「现在有什么、缺什么」 |
| 3 | [feature_bound_guided_hivm_rewriter.md](feature_bound_guided_hivm_rewriter.md) | **特性设计/开发计划**：把优化循环建模成梯度下降（loss=`T_sched`、floor=`T_bound`、梯度=gap 归因、更新=`HivmOpsEditor` 改写），含端到端验证循环、模块接口、M0–M5 里程碑与验收标准 | 下一位基于 AI agent 做实现的开发者 |

## 一句话主线

```
检测 (诊断: 主导 gap)  →  改写 (HivmOpsEditor 应用对应手段)  →  重评 (perfbound 重算 T_bound/T_sched)
        ▲                                                                      │
        └──────────────────  迭代收敛 (贪心坐标下降 / 梯度下降类比)  ◀──────────┘
```

- **(A) 缩小 gap**：`T_bound` 不变，`T_sched` 向下界收敛。
- **(B) 降低 bound**：`T_bound` 自身下移，且守住 `T_bound ≤ T_measured`。

## 关联代码（已在本分支 `integrate/pr20-pr17`）

- 改写执行器：`include/AscendModel/Transforms/HivmOpsEditor.h`、`tools/hivm-crud/`
- 诊断：`lib/AscendModel/Analysis/HIVMBottleneckDiagnosis.cpp`
- 解析 bound：`perfbound/model/bounds.py`、`perfbound/combine/bound_combiner.py`
- 规格：`.omc/specs/performance_bound_model.md`
