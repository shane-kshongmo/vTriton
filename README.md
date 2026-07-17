# TritonSim

基于 MLIR 的 Ascend NPU 性能建模工具，目标硬件为昇腾 910B / 910B3。

仓库采用**双层架构**：

- **C++ / MLIR 层**（`lib/AscendModel/`、`tools/`）：自定义 AscendModel 方言与变换 pass，
  负责 IR 解析、操作分类、周期估算与调度分析。工具 `tritonsim-opt` / `tritonsim-hivm`。
- **Python 层**（`perfbound/`）：零 MLIR 依赖的纯 Python **分析性能下界模型**，
  不跑硬件就能回答"这个 kernel 还能再快吗、瓶颈在哪"。这是当前活跃开发重心。
  详见 [Python 性能边界模型 (perfbound)](#python-性能边界模型-perfbound)。

C++ 工具与 Python 模型之间通过 JSON（DES 图、依赖图）松耦合通信。当前主要覆盖两类输入：

- AscendModel MLIR：用于 pass 级性能分析与报告生成
- HIVM IR：用于调度、同步与 trace 分析

如需更详细的构建说明见 [BUILD.md](BUILD.md)，硬件配置见 [configs/README.md](configs/README.md)，
整体架构与 bound 模型深入说明见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 功能概览

| 工具 / 组件 | 用途 |
|------|------|
| `tritonsim-opt` | 运行 AscendModel 相关 pass pipeline |
| `tritonsim-hivm` | 直接分析 `.npuir.mlir`，也可从 Triton DSL 触发 compile-only dump |
| `ascend-tiling-opt` | 在构建中存在时提供 tiling 优化入口（当前默认未构建） |
| `perfbound/` | Python 性能下界模型：静态判断 kernel 是否已贴住硬件极限、差距归因到哪 |
| `scripts/run_bound.py` | 端到端 bound 流水线入口（NPUIR dump → DES 图 → bound 报告） |
| `configs/*.json` | 定义硬件参数，默认使用 `configs/ascend_910b.json` |
| `docs/` | 架构、校准、部署文档 |

## 前置要求

在开始构建前，请确认以下工具已安装：

| 依赖 | 最低版本 | 检查命令 |
|------|---------|---------|
| CMake | 3.20 | `cmake --version` |
| Ninja | 任意 | `ninja --version` |
| clang / lld | 15 | `clang --version` |
| Python | 3.8 | `python3 --version` |
| git | 任意 | `git --version` |

**磁盘空间**：至少 40 GB（LLVM 构建产物较大）

**内存**：推荐 16 GB（并行构建建议 32 GB）

**WSL 用户（Windows）**：所有二进制均为 Linux ELF 文件，请在 WSL 终端（Ubuntu 24.04）内执行全部命令。

## 快速开始

### 步骤 1：初始化子模块并应用补丁

```bash
git submodule update --init thirdparty/triton-ascend
git -C thirdparty/triton-ascend submodule update --init --depth 1 third_party/ascend/AscendNPU-IR
./scripts/apply_patches.sh
```

`thirdparty/triton-ascend` 指向官方 upstream（gitcode.com/Ascend/triton-ascend），本地补丁（`patches/`）在 submodule checkout 后自动应用，提供 compile-only mock 等功能。

> 如果拉取超时或失败，可改用浅克隆：
> ```bash
> git submodule update --init --depth 1 thirdparty/triton-ascend
> git -C thirdparty/triton-ascend submodule update --init --depth 1 third_party/ascend/AscendNPU-IR
> ./scripts/apply_patches.sh
> ```

---

### 步骤 2：构建 LLVM/MLIR

> **注意：首次构建耗时 30–60 分钟，需占用约 30 GB 磁盘空间，仅需执行一次。**

```bash
./scripts/build_llvm.sh
```

脚本会自动完成以下操作：
1. 使用全部 CPU 核心配置并构建 LLVM（启用 `mlir` 和 `llvm` 项目）
2. 将头文件和库安装到 `thirdparty/llvm-project/build/install`

如果脚本输出 `✅ LLVM/MLIR 已构建`，说明 LLVM 已安装，可直接跳到步骤 3。

> **构建失败？** 若出现内存不足错误，可限制并行度：
> ```bash
> cmake --build thirdparty/llvm-project/build --target install -- -j4
> ```

---

### 步骤 3：构建 triton-ascend（可选）

> 如果不需要 Triton DSL / `.ttir` 输入支持，可跳过此步骤，在步骤 4 中改用 `-DTRITONSIM_ENABLE_TRITON=OFF`。
> 注意：Triton DSL 模式依赖完整的 triton-ascend Python 构建，需要 CANN 环境。

```bash
cd thirdparty/triton-ascend
git submodule update --init --depth 1

cd python
LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
TRITON_PLUGIN_DIRS=$(pwd)/../ascend \
TRITON_BUILD_WITH_CCACHE=true \
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_BUILD_PROTON=OFF \
TRITON_WHEEL_NAME="triton-ascend" \
MAX_JOBS=$(nproc) \
python3 setup.py bdist_wheel

# 记录构建目录，供步骤 4 使用
export TRITON_BUILD_DIR=$(ls -d $PWD/build/cmake.* | head -1)
echo "Triton build dir: $TRITON_BUILD_DIR"

cd ../../..
```

---

### 步骤 4：构建 TritonSim

**方式 A：启用 Triton 支持**（推荐，支持 TTIR 建模）

Triton 支持从 `thirdparty/triton-ascend` 的头文件自动启用，无需构建 triton-ascend wheel。

```bash
mkdir -p build && cd build
cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=../thirdparty/llvm-project/build/install/lib/cmake/mlir \
  -DLLVM_DIR=../thirdparty/llvm-project/build/install/lib/cmake/llvm
ninja
cd ..
```

**方式 B：不启用 Triton 支持**（构建更快，无法处理 `.ttir` 输入）

```bash
mkdir -p build && cd build
cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=../thirdparty/llvm-project/build/install/lib/cmake/mlir \
  -DLLVM_DIR=../thirdparty/llvm-project/build/install/lib/cmake/llvm \
  -DTRITONSIM_ENABLE_TRITON=OFF
ninja
cd ..
```

构建成功后，二进制文件位于 `build/bin/`。

---

## 常用用法

### 分析 AscendModel MLIR

运行完整 pipeline：

```bash
./build/bin/tritonsim-opt test/ascend_ops.mlir -ascend-perf-model
```

分步运行常用 pass：

```bash
./build/bin/tritonsim-opt test/ascend_ops.mlir \
  -assign-op-ids \
  -estimate-cycles \
  -analyze-pipeline \
  -perf-report
```

指定硬件配置：

```bash
./build/bin/tritonsim-opt test/ascend_ops.mlir \
  -ascend-perf-model="hardware-config=configs/ascend_910b.json"
```

### 分析 HIVM IR

直接分析仓库内示例：

```bash
./build/bin/tritonsim-hivm --npuir-file test/hivm_add_kernel.npuir.mlir
./build/bin/tritonsim-hivm --npuir-file test/hivm_mixed_cv_kernel.npuir.mlir
```

导出 Perfetto trace：

```bash
./build/bin/tritonsim-hivm \
  --npuir-file test/hivm_add_kernel.npuir.mlir \
  --perfetto-trace-file /tmp/hivm_trace.json
```

也可在 `tritonsim-opt` 中直接对 HIVM IR 跑 pass：

```bash
./build/bin/tritonsim-opt --analyze-hivm test/hivm_mixed_cv_kernel.npuir.mlir
```

### 从 Triton DSL 触发 HIVM 分析

该模式依赖可用的 Python + `triton-ascend` 环境：

```bash
./build/bin/tritonsim-hivm \
  --triton-script test/triton_smoke.py \
  --python python3
```

如果脚本提供明确入口，优先使用 `--triton-entry` / `--entry-arg`：

```bash
./build/bin/tritonsim-hivm \
  --triton-script path/to/script.py \
  --triton-entry main \
  --entry-arg 1 \
  --python python3
```

`--triton-script` 会调用公共 launcher
`tools/common/triton_dsl_dump_launcher.py`，以 compile-only 模式执行 Triton
DSL。该 launcher 同时产出 TTIR 与 HIVM/NPUIR dump：

- TTIR dump 可由 `tritonsim-opt` 继续建模
- `.npuir.mlir` dump 可由 `tritonsim-hivm` 继续建模

因此 TTIR 与 HIVM 的 DSL 入口共享同一个 dump launcher，只是在 dump 之后进入不同的建模工具。

### 分析 Triton IR (`.ttir`)

需要 `triton-opt`（来自 triton-ascend 构建）将 TTIR 转为 generic MLIR，再由 `tritonsim-opt` 运行建模 pipeline：

```bash
TRITON_OPT=/path/to/triton-opt
HW_CONFIG=configs/ascend_910b.json

$TRITON_OPT kernel.ttir --allow-unregistered-dialect --mlir-print-op-generic | \
./build/bin/tritonsim-opt - \
  --allow-unregistered-dialect \
  -ascend-perf-model="hardware-config=${HW_CONFIG} arg-bindings=arg7=4096"
```

`arg-bindings` 用于绑定函数参数（如 `scf.for` 的动态上界），根据 `.mlir` 文件内容确定绑定值。

---

## Python 性能边界模型 (perfbound)

### 它解决什么问题？

你写完一个 Triton kernel，实测 450 µs。**这个数字是好是坏？还值得继续调吗？**
通常只能靠经验猜，或者把 tiling / fusion / 双缓冲挨个试一遍，试完才知道有没有用。

`perfbound` 就是来回答这个问题的。它对 kernel 做**纯静态分析**，算出一个**可证明保守的执行时间下界** `T_bound`：

> 在当前硬件和当前 kernel 结构下，**无论怎么调优都不可能快过 `T_bound`**。

有了这条地板线，优化就不再是猜谜：

| 实测 vs 下界 | 说明 | 该做什么 |
|------|------|---------|
| `T_measured ≈ T_bound` | kernel 已经贴住硬件极限 | **停止微调**——再快只能换算法（fusion / 降精度 / 减少数据搬运） |
| `T_measured ≫ T_bound` | 结构上还有差距 | 看报告的**五路归因**：它直接告诉你差距在哪、先做哪一件事 |

两个关键特性：

- **不编译、不运行 kernel，也不需要昇腾硬件**——只吃 IR，本地就能跑。实测数据只通过校准 (M1) 与验证 (M6) 进入模型。
- **保守，而不是"预测"**——它不预测 kernel 会跑多快，只保证**不会更快**。宁可报得偏低，也不会给出一个你根本达不到的乐观目标。

### 核心公式

```
T_bound = max(T_grid_floor, T_core_floor + T_serial_irreducible)
```

- `T_grid_floor`：芯片网格级下界（占据率、负载均衡、带宽 / 算力瓶颈）
- `T_core_floor`：单核组件级下界（基于加权调和平均的 Roofline 各组件吞吐率）
- `T_serial_irreducible`：不可消除的跨组件串行握手开销

> 这是实现的**保守**形式——`T_serial` 附着于 Tier-2 项（握手为核内 Cube↔Vector）。spec 散文与 `perfbound/__init__.py` 使用加性简写 `max(grid, core)+serial`，该形式非保守（可能违反 `T_bound ≤ T_measured`）；`bound_combiner.py` 实现的是上式。详见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §1。

### 运行端到端 bound 流水线

入口脚本 [`scripts/run_bound.py`](scripts/run_bound.py) 串联：

```
内核脚本 → Triton NPUIR dump → 清洗后的 NPUIR → DES 图 JSON → perfbound JSON 报告
```

```bash
# 按注册名运行（内置内核见 perfbound/experiments/registry.py：
# vector_add, vector_add_2x, softmax, layernorm, rmsnorm, seeded_gap1/gap2/serial）
python scripts/run_bound.py --kernel seeded_serial --grid 128,32

# 按脚本路径运行，并对比实测时间（给了 --measured-us，报告才会填入实测对比与 headroom 评估）
python scripts/run_bound.py --script path/to/kernel.py --grid 128,32 \
  --calibration <calib.json> --measured-us <T_measured>

# 只想看流水线会执行什么，不实际跑：
python scripts/run_bound.py --kernel vector_add --grid 128,32 --dry-run
```

`run_bound.py` 会调用 `build/bin/tritonsim-hivm` 生成 DES 图，因此真实（非 `--dry-run`）运行需要该二进制存在。
内核脚本需暴露 `build_inputs()` 与 `Model`（契约见 `perfbound/experiments/registry.py` 的 `KernelSpec.validate_interface`）。

### 读懂报告

流水线产出 `<kernel>.report.json`；`KernelReport.to_text()`（`perfbound/combine/report.py`）给出等价的可读版本。
以下为**示意输出**（数字仅用于说明格式），四块最值得看：

#### 1. 三级可达性层次 — 差距该归谁

```
Reachability Hierarchy:
  1. Hardware floor  (T_bound_HIVM):  120.00 us
  2. DSL bound       (T_bound_DSL):   180.00 us   [compiler headroom: 60.00 us]
  3. Measured        (T_measured):    450.00 us   [author residual, not proven attainable: 270.00 us]
```

- **compiler headroom** = `DSL − HIVM`：编译器（bishengir）实际生成的结构，比硬件本可允许的理想结构差多少。这部分你改 kernel 一般动不了。
- **author residual** = `Measured − DSL`：实测比你自己写的 kernel 结构所允许的下界差多少——**这是通常你能动的部分**。
- 若出现 `*** BOUND VIOLATION: T_bound > T_measured ***`，说明保守性被打破了：这是**模型 bug**，不是 kernel 变快了，请当作 issue 处理。

#### 2. 五路归因 — 差距在哪、先做哪件事

报告按占 `T_bound` 的比例列出五类差距，并给出**唯一一条**推荐动作（`recommended_action`）：

| 归因项 | 含义 | 对应建议动作 |
|--------|------|-------------|
| `grid` | 网格切分不好：占据率低或负载不均 | 调整 grid 划分，提高占据率 / 均衡负载 |
| `gap1_wrong_unit` | op 落在了错误的执行单元上 | 修正 DSL 类型，把 op 挪到合适的单元 |
| `gap2_coalescing` | 搬运太碎，摊销开销大 | 合并传输，增大单次传输尺寸 |
| `gap3_avoidable_serial` | 可避免的串行握手（Cube↔Vector 没重叠） | 加 ping-pong 双缓冲让握手重叠 |
| `gap4_intra_unit_exec` | 单元内执行效率低 | 提高 SIMD repeat / mask 利用率 |

若五类差距全部低于阈值，`recommended_action` 会直接告诉你 **"At component bound"**——
没有可做的软件层优化了，只剩算法层重设计。

#### 3. Attainable Headroom Assessment — 别把差距当成承诺的加速比

```
Attainable Headroom Assessment:
  status:     diagnostic_upper_bound
  confidence: low
  diagnostic range: 0.00..85.00 us
  point estimate: unavailable
```

这是最容易误读的一块，请注意：**`Measured − T_bound` 是上界，不是"能省下来的时间"。**
模型刻意不给点估计（`point estimate: unavailable`）——按 `headroom_method` 的说法，
在拿到**正确性验证过的反事实实测 (counterfactual)** 之前，不对"可兑现的加速"作任何声明。
差距里也可能有一部分根本兑现不了：它来自模型尚未建模的项（例如标量发射受限），
是 bound 偏低造成的假象，而非真实可回收的时间。
`status` / `confidence` 就是在标注这份不确定性，请照字面理解。

#### 4. 校准溯源

报告还会带上 `Calibration:` 段（来源、版本、目标硬件、P0 常量是否齐全、最大相对 CI）。
出现 `P0 violation` 或 `diagnostic fallback` 时，说明部分常量未实测、模型走了退化路径，
结论的可信度要相应打折。校准流程见 [docs/CALIBRATION_GUIDE.md](docs/CALIBRATION_GUIDE.md)。

### 完整示例：走查 seeded_serial

下面用仓库自带的 `test/seeded_serial_bench.py` 完整走一遍。
**本节所有数字均为真实输出**——来自 910B3 校准 (`calib_910b3_v1.json`) 与该 kernel 在 910B3 上 dump 出的 NPUIR，
不是示意值。

这个 fixture 是**故意造出来**的：kernel 内有两条**完全独立**的流——
Stream A 是 HBM 搬运，Stream B 是 Vector FMA 链——中间隔了一个 `tl.debug_barrier()`。
两条流不共享任何数据，所以这个 barrier **可证明是多余的**：它只是白白拆掉了 MTE↔Vector 的重叠。

#### 步骤 1：拿到 NPUIR

bound 模型吃的是 NPUIR。从 kernel 脚本 dump 需要昇腾环境
（`build_inputs()` 用 `device="npu"`），dump 后用 `scripts/clean_npuir.py` 清洗：

```bash
TRITON_KERNEL_DUMP=1 TRITON_ALWAYS_COMPILE=1 TRITON_DUMP_DIR=/tmp/ttdump \
  python test/seeded_serial_bench.py
python scripts/clean_npuir.py <dump>.npuir.mlir seeded_serial.clean.npuir.mlir
```

**只有这一步需要硬件。** 拿到 `.npuir.mlir` 之后，下面的分析纯本地、离线可跑。

#### 步骤 2：算 bound

```bash
python -m perfbound.combine.run_report \
  --npuir seeded_serial.clean.npuir.mlir \
  --grid 4096 --cores 20 \
  --kernel-name seeded_serial \
  --tritonsim-hivm build/bin/tritonsim-hivm \
  --calibration perfbound/calibration/data/calib_910b3_v1.json \
  --measured-us 3289.604
```

`--measured-us 3289.604` 是该 kernel 在 910B3 上的实测中位数（40 次，CV 0.06%）。
不给这个参数也能算 bound，只是不会有实测对比。

#### 步骤 3：读输出

```
=== Performance Bound Report: seeded_serial ===

T_bound:   1840.01 us
  Tier 1 (grid):      1838.21 us
  Tier 2 (component): 1840.01 us
  Serial irreducible: 0.00 us

Binding: component
  Component: vector

Calibration:
  source:   perfbound/calibration/data/calib_910b3_v1.json
  version:  v1
  hardware: Ascend 910B3
  P0 status: complete
  measured constants: 19 (max relative 95% CI: 1.07%)
  warning: P1 scalar_overhead_factor not calibrated — kernel-level bounds may be optimistic
  diagnostic fallback: Gap 4 startup latency uses hard-coded diagnostic defaults for vector, cube; attribution is not fully calibration-backed

Attribution (absolute and fraction of T_bound):
  gap3_avoidable_serial: 0.25 us (0.000)
  grid: 0.00 us (0.000)
  gap1_wrong_unit: 0.00 us (0.000)
  gap2_coalescing: 0.00 us (0.000)
  gap4_intra_unit_exec: 0.00 us (0.000)

Reachability Hierarchy:
  1. Hardware floor  (T_bound_HIVM):  1840.01 us
  2. DSL bound       (T_bound_DSL):   1840.01 us   [compiler headroom: 0.00 us]
  3. Measured        (T_measured):    3289.60 us   [author residual, not proven attainable: 1449.60 us]

Attainable Headroom Assessment:
  status:     unavailable
  confidence: none
  point estimate: unavailable
  method: No correctness-verified counterfactual measurement is available.

Recommended action: Add ping-pong buffer to overlap this handoff
```

#### 步骤 4：这份报告到底说了什么

**① 卡在 Vector 上。** `Binding: component / vector`，Tier-2 (1840.01) 略高于 Tier-1 (1838.21)，
两者几乎持平——说明网格切分没问题，瓶颈在单核 Vector 吞吐（就是 Stream B 那条 FMA 链）。

**② 实测比下界慢了 1449.60 µs（44%）。** 这是 `author residual`。**但先别急着高兴**——
注意 `Attainable Headroom Assessment` 明确写着 `status: unavailable` / `confidence: none`。
模型在告诉你：*我不认为这 1449 µs 是你能拿回来的时间*。再看校准段的那条 warning：

> `P1 scalar_overhead_factor not calibrated — kernel-level bounds may be optimistic`

标量开销尚未校准，**bound 本身可能偏低**——残差里有多少是"真实差距"、多少是"模型缺项造成的假象"，
这份报告没法区分。（这正是 `chunk_kda` 上踩过的坑：一个看似 55% 的 headroom 最后被证明是 bound 模型的伪影，
kernel 实际已经卡在标量发射上。）

**③ `Recommended action` 这次不要当真。** 报告建议 "Add ping-pong buffer"，但看归因的绝对值：
五类差距加起来只有 0.25 µs，占 `T_bound` 的 **0.014%**——对一个 3289 µs 的 kernel 完全是噪声。
它之所以还给建议、而不是报 "At component bound"，只是因为 `0.25/1840 ≈ 1.36e-4` 刚好越过了
at-bound 阈值 `1e-4`。**归因的绝对值 (µs) 比推荐动作更重要**：当五类 gap 都接近 0 时，
真正的信号是"那 1449 µs 没有被任何一类 gap 解释"，而不是那条推荐。

一句话总结这个 kernel：**网格没问题、结构上五类已知 gap 都已榨干，剩下的差距模型解释不了**——
下一步该做的是补标量校准 / 上 profile，而不是去加双缓冲。

#### 步骤 5：headroom 怎样才算数——一个已验证的反事实

前面反复强调"headroom 不等于承诺的加速比"。那什么时候才算数？答案是**做反事实实测**。
`seeded_serial` 恰好做过（US-SB-008，结果存于 `.omc/research/hw_runs/seeded_serial/`）：

把那条多余的 barrier 从编译器 IR 里摘掉，重新编译、实测、并逐位校验输出一致：

| 项 | 值 |
|----|-----|
| 模型预测的 compiler headroom | **20.94 µs** |
| 硬件实测去掉 barrier 后的提速 | **20.86 µs** |
| 量化误差 | **0.37%** |
| 输出逐位一致 (`output_verified`) | ✅ |

模型预测 20.94 µs，硬件实际给了 20.86 µs。**这才是一个可兑现的 headroom**——
因为它有一个正确性验证过的反事实撑着。步骤 4 里那 1449 µs 没有，所以模型拒绝为它背书。

> 注：这个 20.94 µs 的 compiler headroom 来自 US-SB-008 的专门实验（带 `pipe_barrier` 校准常量）；
> 上面步骤 3 的默认路径未建模该项，因此那里显示 `compiler headroom: 0.00 us`。

### 模型组织

六个阶段：`calibration/` (M1) → `extract/` (M2/M3) → `model/` (M4) → `combine/` (M5) → `validate/` (M6)。

| 阶段 | 目录 | 职责 |
|------|------|------|
| M1 | `calibration/` | 硬件常量校准数据库（实测在此进入） |
| M2/M3 | `extract/` | DSL 网格提取 + HIVM 组件提取 |
| M4 | `model/` | 网格 / 组件分析模型（纯函数） |
| M5 | `combine/` | 边界合并 + 五路归因 + 双限 (two-limit) |
| M6 | `validate/` | 保守性 / 紧致性验证 |

完整说明见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)（架构权威参考）。

---

## 测试与验证

常用本地检查：

```bash
./build/bin/tritonsim-opt test/ascend_ops.mlir -ascend-perf-model
./build/bin/tritonsim-hivm --npuir-file test/hivm_add_kernel.npuir.mlir
python3 test/triton_smoke.py
```

Python (perfbound) 测试套件位于 `tests/perfbound/`，`conftest.py` 会自动把仓库根加入 `sys.path`：

```bash
python -m pytest tests/perfbound/                       # 全部
python -m pytest tests/perfbound/test_bounds.py         # 单文件
python -m pytest tests/perfbound/ -k chunk_kda          # 按关键字筛选
python -m pytest tests/hivm/                            # HIVM 同步 / 组件验证
```

启用 C++ 测试后可运行：

```bash
ctest --test-dir build     # 需在 cmake 配置时加 -DASCEND_MODEL_ENABLE_TESTS=ON
```

---

## 端到端使用指南：Triton DSL → HIVM 分析

以 DeepSeek-V3 的稀疏注意力 prefill kernel (`prefill_a5_cvpipe.py`) 为例，演示从 Triton DSL 脚本到性能分析的完整流程。

### 示例 kernel 简介

`prefill_a5_cvpipe.py` 实现了面向 Ascend910D (A5) 的稀疏 Flash Attention prefill kernel，主要优化：

- **QK nope/rope 分裂**：将 K workspace 拆为 `K_nope[T1, N2, K, 512]` 和 `K_rope[T1, N2, K, 64]`，QK matmul 拆成两个子矩阵乘，SV matmul 直接复用 K_nope 数据（潜在 L1 命中）
- **Cube-Vector 混合流水 (cvpipe)**：通过 `enable_mixed_cv=True` 开启 Cube/Vector 双流水
- **Graph-based sync**：通过 `inject_barrier_all=False` 关闭全局 barrier，使用 GraphSyncSolver 进行最小化同步

kernel 入口函数 `test_dsa_prefill` 接收以下参数：

| 参数 | 含义 | 示例值 |
|------|------|--------|
| batch | batch size | 1 |
| q_seq_len | query 序列长度 | 2048 |
| k_seq_len | key 序列长度 | 1024 |
| head_num | 注意力头数 | 16 |
| kv_lora_rank | KV LoRA 维度 (D_v) | 512 |
| qk_rope_head_dim | RoPE head 维度 | 64 |
| dtype | 数据类型 | torch.bfloat16 |

### 步骤 1：准备 kernel 脚本

将 kernel 脚本放置在可访问的路径，例如 `/path/to/prefill_a5_cvpipe.py`。

脚本需要满足以下条件：
- 使用 `@triton.jit` 装饰 kernel 函数
- 提供一个 Python 可调用的入口函数（本例为 `test_dsa_prefill`），负责构造输入 tensor 并调用 kernel
- 入口函数的参数将通过 `--entry-arg` 逐一传入

### 步骤 2：运行 HIVM 分析

使用 `tritonsim-hivm` 的 `--triton-script` 模式，指定入口函数和参数：

```bash
./build/bin/tritonsim-hivm \
  --triton-script /path/to/prefill_a5_cvpipe.py \
  --triton-entry test_dsa_prefill \
  --entry-arg 1 \
  --entry-arg 2048 \
  --entry-arg 1024 \
  --entry-arg 16 \
  --entry-arg 512 \
  --entry-arg 64 \
  --entry-arg torch.bfloat16 \
  --python python3 \
  --scheduler des \
  --des-graph-file /tmp/prefill_a5_des_graph.json \
  --perfetto-trace-file /tmp/prefill_a5_trace.json \
  --keep-dump-dir \
  2>&1
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--triton-script` | Triton kernel 脚本路径 |
| `--triton-entry` | 脚本中的入口函数名 |
| `--entry-arg` | 入口函数的参数，按顺序传入（可多次指定） |
| `--python` | Python 解释器路径 |
| `--scheduler des` | 使用 DES (Discrete Event Simulation) 调度器 |
| `--des-graph-file` | 导出 DES 调度图（JSON 格式） |
| `--perfetto-trace-file` | 导出 Perfetto trace（可在 [ui.perfetto.dev](https://ui.perfetto.dev) 打开） |
| `--keep-dump-dir` | 保留中间编译产物目录，便于调试 |

保留的 dump 目录中通常包含：

| 产物 | 用途 |
|------|------|
| `*/add_kernel.ttir` | Triton TTIR，可用 `tritonsim-opt` 做 TTIR 建模 |
| `kernel_*.npuir.mlir` | HIVM/NPUIR，可用 `tritonsim-hivm --npuir-file` 做 HIVM 建模 |
| `tritonsim_hivm_bindings.jsonl` | launcher 捕获的动态参数绑定 |
| `tritonsim_hivm_compile_commands.jsonl` | launcher 捕获的 `bishengir-compile` 命令 |

### 步骤 3：查看分析结果

分析完成后会在终端输出性能报告，包括：

- 各 op 的周期数估计
- Cube/Vector 流水线利用率
- 内存搬运开销

同时可以：

1. **查看 Perfetto trace**：将 `/tmp/prefill_a5_trace.json` 拖入 [ui.perfetto.dev](https://ui.perfetto.dev)，可视化 Cube/Vector/MTE 各单元的时间线
2. **查看 DES 调度图**：`/tmp/prefill_a5_des_graph.json` 包含 op 间依赖关系和调度顺序

### 步骤 4（可选）：调整参数重新分析

修改 `--entry-arg` 即可测试不同 shape 配置下的性能，例如：

```bash
# batch=2, q_seq_len=4096
./build/bin/tritonsim-hivm \
  --triton-script /path/to/prefill_a5_cvpipe.py \
  --triton-entry test_dsa_prefill \
  --entry-arg 2 \
  --entry-arg 4096 \
  --entry-arg 1024 \
  --entry-arg 16 \
  --entry-arg 512 \
  --entry-arg 64 \
  --entry-arg torch.bfloat16 \
  --python python3 \
  --scheduler des \
  --perfetto-trace-file /tmp/prefill_a5_b2s4096_trace.json
```

### WSL 环境下的运行方式

如果在 Windows 环境下使用 WSL，需要将命令写入 shell 脚本再执行：

```bash
# 1. 将脚本写入 WSL 文件系统
cat > /tmp/run_prefill.sh << 'SCRIPT'
#!/bin/bash
set -e

TRITONSIM_HIVM=/mnt/d/work/git/vTriton/build/bin/tritonsim-hivm
PYTHON=/path/to/python3

# 可选：先重新编译
cd /mnt/d/work/git/vTriton/build && ninja -j$(nproc) 2>&1 | tail -5

${TRITONSIM_HIVM} \
  --triton-script /path/to/prefill_a5_cvpipe.py \
  --triton-entry test_dsa_prefill \
  --entry-arg 1 \
  --entry-arg 2048 \
  --entry-arg 1024 \
  --entry-arg 16 \
  --entry-arg 512 \
  --entry-arg 64 \
  --entry-arg torch.bfloat16 \
  --python ${PYTHON} \
  --scheduler des \
  --perfetto-trace-file /tmp/prefill_a5_trace.json \
  2>&1
SCRIPT

# 2. 执行
bash /tmp/run_prefill.sh
```

---

## 仓库结构

```text
include/AscendModel/   公共头文件（方言、接口、pass 声明）
lib/AscendModel/       分析、IR 与 transforms 的 C++ 实现
tools/                 命令行工具入口（tritonsim-opt / tritonsim-hivm）
perfbound/             Python 两层级分析性能下界模型（活跃开发重心）
tests/                 Python 测试套件（perfbound / hivm）
test/                  C++ 示例输入（.mlir/.ttir）与 smoke/bench 脚本
configs/               硬件配置与 schema（910B / 910B3）
scripts/               构建、补丁应用、bound 流水线与基准脚本
docs/                  架构、校准、部署文档
patches/               应用到 thirdparty 子模块的本地补丁
thirdparty/            外部依赖（triton-ascend、AscendNPU-IR 等）
```

## 说明

- 默认硬件配置为 Ascend 910B；校准与硬件实测针对 **910B3**（见 `configs/ascend_910b3.json`）
- `perfbound` 以源码内导入方式运行（仓库根加入 `sys.path`），无需 pip 安装
- 硬件实测 / 校准微基准 (M6) 在远端 910B3 机器上执行，而非本地
- 与具体本机路径绑定的示例、临时脚本路径和历史实现细节未保留在本 README 中
- 更深入的构建选项、Triton 集成方式和硬件配置格式请分别查看 [BUILD.md](BUILD.md) 与 [configs/README.md](configs/README.md)；
  整体架构与 bound 模型见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## License

Apache 2.0
