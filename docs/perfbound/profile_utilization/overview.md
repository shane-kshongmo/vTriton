# Profiling 利用率分析模块输入输出说明

本文档说明 [profile_utilization.py](../../../perfbound/analyze/profile_utilization.py) 需要什么输入、输出什么结果，以及各个类字段的含义。

注意：文档中的类名、字段名、函数名保持 Python 代码里的英文名字；解释文字使用中文。

## 模块用途

`profile_utilization.py` 用于在 component-based Roofline 基准值已经算出之后，结合 profiling 统计数据判断算子的性能瓶颈。

分析流程如下：

```text
Profiling 数据
-> 计算 A / I / U
-> 判断是否达到 Roofline ceiling
-> 如果未达到 ceiling，则计算 R / E
-> 判断是并行不足还是组件效率不足
-> 输出主导 component，以及主导 precision 或 transfer path
```

核心指标：

```text
A = Actual Performance，实际性能
I = Ideal Performance，理想性能
U = Utilization = A / I，利用率
R = Active Time Ratio = T_component / T_total，活跃时间比例
E = Execution Efficiency = U / R，执行效率
```

## 入口函数

### compute_realized_utilization

```python
compute_realized_utilization(
    profile: KernelProfileStats,
    component_bound: ComponentBound,
    work_tolerance: float = 0.10,
) -> UtilizationReport
```

作用：只计算每个 component 的 `A/I/U/R/E`，不做最终瓶颈分类。

### analyze_operator_bottleneck

```python
analyze_operator_bottleneck(
    profile: KernelProfileStats,
    component_bound: ComponentBound,
    u_threshold: float = 0.80,
    r_threshold: float = 0.50,
    work_tolerance: float = 0.10,
) -> OperatorBottleneckReport
```

作用：先调用 `compute_realized_utilization()` 计算 component 级指标，然后按照论文第四章后半部分的流程输出最终诊断。

## 输入数据

两个入口函数都需要：

```text
profile: KernelProfileStats
component_bound: ComponentBound
```

其中 `profile` 来自 profiling 数据整理，`component_bound` 来自 `component_model.py` 的理论基准计算。

### KernelProfileStats

`KernelProfileStats` 表示一个算子的 profiling 汇总输入。

代码定义：

```python
@dataclass
class KernelProfileStats:
    kernel_name: str
    elapsed_time_us: float
    components: list[ProfileComponentStats]
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `kernel_name` | `str` | 算子名称，只用于报告展示 |
| `elapsed_time_us` | `float` | 算子总执行时间，单位是微秒 |
| `components` | `list[ProfileComponentStats]` | 每个 component 的 profiling 汇总 |

`elapsed_time_us` 是所有 component 计算 `R` 时共用的分母：

```text
R = active_time_us / elapsed_time_us
```

正常情况下，每个 component 不应该有自己的总时间；不同 component 只有自己的 `active_time_us`。

### ProfileComponentStats

`ProfileComponentStats` 表示单个 component 的 profiling 汇总。

代码定义：

```python
@dataclass
class ProfileComponentStats:
    component: Component
    work_done: float
    active_time_us: float
    work_breakdown: list[WorkBreakdownItem]
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `component` | `Component` | component 类型，例如 `Component.CUBE`、`Component.VECTOR`、`Component.MTE_GM` |
| `work_done` | `float` | 该 component 完成的总工作量 |
| `active_time_us` | `float` | 该 component 处于 active 状态的时间，单位是微秒 |
| `work_breakdown` | `list[WorkBreakdownItem]` | component 内部按 precision 或 transfer path 拆分的工作量 |

`work_done` 的单位取决于 component 类型：

| Component 类型 | `work_done` 单位 |
| --- | --- |
| `CUBE` / `VECTOR` / `SCALAR` | ops 或 FLOPs |
| `MTE_GM` / `MTE_L1` / `MTE_UB` | bytes |

`active_time_us` 对应论文里的 `T_component`。

### WorkBreakdownItem

`WorkBreakdownItem` 表示 component 内部的一项细分工作量。

代码定义：

```python
@dataclass
class WorkBreakdownItem:
    label: str
    work: float
    peak_rate: float
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `label` | `str` | 细分项名称 |
| `work` | `float` | 该细分项的工作量 |
| `peak_rate` | `float` | 该细分项对应的理想峰值或校准峰值 |

对于计算类 component：

```text
label 通常是 precision，例如 fp16 / bf16 / fp32 / int8
work 是该 precision 的 ops 或 FLOPs
peak_rate 是该 precision 的 ops/us 或 FLOPs/us
```

示例：

```python
WorkBreakdownItem("fp16", work=600.0, peak_rate=100.0)
WorkBreakdownItem("int8", work=400.0, peak_rate=200.0)
```

对于搬运类 component：

```text
label 通常是 transfer path，例如 gm->ub / l1->l0a / ub->gm
work 是该 path 搬运的 bytes
peak_rate 是该 path 的 bytes/us
```

示例：

```python
WorkBreakdownItem("gm->ub", work=600.0, peak_rate=100.0)
WorkBreakdownItem("gm->l1", work=300.0, peak_rate=150.0)
```

`work_breakdown` 用来计算 operator-aware 的理想性能：

```text
I = Sum(work_i) / Sum(work_i / peak_rate_i)
```

如果没有 `work_breakdown`，代码会回退到 `ComponentBound` 中已有的 component 级 `I_c`。

### ComponentBound

`ComponentBound` 是 `component_model.py` 的输出，不是 profiling 数据。

它由 `compute_component_floor()` 计算得到：

```python
compute_component_floor(
    extract,
    cube,
    vector,
    memory,
    core,
) -> ComponentBound
```

`profile_utilization.py` 主要使用这些字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `t_core_floor_us` | `float` | 单 core 的 component roofline 下界 |
| `binding_component` | `Component` | 理论上 binding 的 component |
| `total_ops` | `dict[str, float]` | 计算类 component 的理论 ops |
| `total_bytes` | `dict[str, float]` | 搬运类 component 的理论 bytes |
| `per_component_us` | `dict[str, float]` | 每个 component 的理论时间 `O_c / I_c` |

当 profiling 输入没有 `work_breakdown` 时，代码会用：

```text
I_c = bound_work / per_component_us[component]
```

其中：

```text
计算类 component: bound_work = total_ops[component]
搬运类 component: bound_work = total_bytes[component]
```

## 输出数据

### ComponentUtilization

`ComponentUtilization` 是单个 component 的分析结果。

代码定义：

```python
@dataclass
class ComponentUtilization:
    component: Component
    work_done: float
    bound_work: float
    elapsed_time_us: float
    active_time_us: float
    actual_performance: float
    ideal_performance: float
    e_efficiency: float
    r_residency: float
    u_utilization: float
    dominant_item: Optional[str]
    dominant_share: float
    warnings: list[str]
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `component` | component 类型 |
| `work_done` | profiling 输入中的实际工作量 |
| `bound_work` | `ComponentBound` 中对应 component 的理论工作量 |
| `elapsed_time_us` | 算子总时间 |
| `active_time_us` | component active 时间 |
| `actual_performance` | 实际性能，`A = work_done / elapsed_time_us` |
| `ideal_performance` | 理想性能，`I = Sum(work_i) / Sum(work_i / peak_rate_i)` |
| `u_utilization` | 利用率，`U = A / I` |
| `r_residency` | 活跃时间比例，`R = active_time_us / elapsed_time_us` |
| `e_efficiency` | 执行效率，`E = U / R` |
| `dominant_item` | 主导 precision 或 transfer path |
| `dominant_share` | 主导项占该 component 工作量的比例 |
| `warnings` | 该 component 的异常提示 |

### UtilizationReport

`UtilizationReport` 是 `compute_realized_utilization()` 的输出。

代码定义：

```python
@dataclass
class UtilizationReport:
    kernel_name: str
    elapsed_time_us: float
    t_core_floor_us: float
    binding_component: Component
    component_results: dict[str, ComponentUtilization]
    warnings: list[str]
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `kernel_name` | 算子名称 |
| `elapsed_time_us` | 算子总时间 |
| `t_core_floor_us` | `component_model.py` 给出的理论 core floor |
| `binding_component` | `component_model.py` 给出的理论 binding component |
| `component_results` | 每个 component 的 `ComponentUtilization` 结果 |
| `warnings` | 所有 component warning 的汇总 |

`component_results` 的 key 是 component 字符串，例如：

```text
cube
vector
mte_gm
mte_ub
```

### OperatorBottleneckReport

`OperatorBottleneckReport` 是 `analyze_operator_bottleneck()` 的输出。

代码定义：

```python
@dataclass
class OperatorBottleneckReport:
    kernel_name: str
    elapsed_time_us: float
    diagnosis: str
    bound_kind: Optional[str]
    dominant_component: Optional[Component]
    dominant_item: Optional[str]
    dominant_share: float
    component_results: dict[str, ComponentUtilization]
    warnings: list[str]
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `kernel_name` | 算子名称 |
| `elapsed_time_us` | 算子总时间 |
| `diagnosis` | 最终诊断结果 |
| `bound_kind` | 如果达到 roofline ceiling，记录是 `Compute Bound` 还是 `MTE Bound` |
| `dominant_component` | 主导瓶颈 component |
| `dominant_item` | 主导 precision 或 transfer path |
| `dominant_share` | 主导项占比 |
| `component_results` | 每个 component 的 `ComponentUtilization` 结果 |
| `warnings` | warning 汇总 |

`diagnosis` 可能的取值如下：

| 取值 | 中文含义 |
| --- | --- |
| `Compute Bound` | 计算受限，计算类 component 达到 roofline ceiling |
| `MTE Bound` | 搬运受限，搬运类 component 达到 roofline ceiling |
| `Insufficient Parallelism` | 并行不足，所有有效 component 的 active time ratio 都偏低 |
| `Inefficient Compute` | 计算效率不足，计算类 component 高 R 但低 E |
| `Inefficient MTE` | 搬运效率不足，搬运类 component 高 R 但低 E |
| `Insufficient Data` | 数据不足，无法分析 |

## 诊断规则

调用：

```python
analyze_operator_bottleneck(
    profile,
    component_bound,
    u_threshold=0.80,
    r_threshold=0.50,
)
```

规则：

1. 计算所有有效 component 的 `U`。
2. 如果存在 `U >= u_threshold`：
   - component 属于 `CUBE` / `VECTOR` / `SCALAR`，诊断为 `Compute Bound`
   - component 属于 `MTE_GM` / `MTE_L1` / `MTE_UB`，诊断为 `MTE Bound`
3. 如果所有 component 都未达到 ceiling，则进入利用率不足分析。
4. 如果所有有效 component 的 `R < r_threshold`，诊断为 `Insufficient Parallelism`。
5. 否则选择高 `R` 且低 `E` 最明显的 component：
   - 计算类 component 诊断为 `Inefficient Compute`
   - 搬运类 component 诊断为 `Inefficient MTE`

## 示例

```python
from perfbound.extract.op_classifier import Component
from perfbound.analyze.profile_utilization import (
    KernelProfileStats,
    ProfileComponentStats,
    WorkBreakdownItem,
    analyze_operator_bottleneck,
)

profile = KernelProfileStats(
    kernel_name="demo",
    elapsed_time_us=10.0,
    components=[
        ProfileComponentStats(
            component=Component.CUBE,
            work_done=1000.0,
            active_time_us=8.0,
            work_breakdown=[
                WorkBreakdownItem("fp16", work=600.0, peak_rate=100.0),
                WorkBreakdownItem("int8", work=400.0, peak_rate=200.0),
            ],
        ),
    ],
)

report = analyze_operator_bottleneck(profile, component_bound)
```

这个例子中，Cube component 的指标为：

```text
I = 1000 / (600/100 + 400/200) = 125
A = 1000 / 10 = 100
U = 100 / 125 = 0.8
R = 8 / 10 = 0.8
E = 0.8 / 0.8 = 1.0
```

如果 `u_threshold=0.80`，该 component 达到 roofline ceiling，最终诊断为 `Compute Bound`。

## 当前 CSV 数据能提供什么

仓库根目录下的 `data/ascend_kernel_details_*.csv` 是 profiling / op summary 类型数据。

它们通常能提供：

```text
elapsed_time_us: Duration(us) 或 aicore_time/aiv_time
active_time_us: aic_mac_time/aiv_vec_time/aiv_mte2_time/aiv_mte3_time 等
```

但它们通常还不直接提供完整的：

```text
Compute ops per precision
MTE bytes per transfer path
peak_rate / bandwidth per breakdown item
```

这些需要结合 shape、dtype、HIVM extract 和 calibration DB 推导后，才能填入 `ProfileComponentStats.work_done` 和 `work_breakdown`。
