# TritonSim 构建指南

## 架构概览

```
┌──────────────────────────────────────────────────────────┐
│  LLVM/MLIR  (编译一次，共用)                               │
│  commit: b5cc222d  (由 triton-ascend 指定)                │
│  安装路径: $LLVM_INSTALL_PREFIX                            │
└──────────┬──────────────────────┬────────────────────────┘
           │                      │
           ▼                      ▼
┌─────────────────────┐  ┌─────────────────────────────────┐
│  triton-ascend      │  │  TritonSim (本项目)              │
│  (Triton 编译器)     │  │  (Ascend 910B 性能建模)          │
│                     │  │                                 │
│  产物:              │──▶│  输入:                          │
│  - TritonIR 库      │  │  - 复用同一 LLVM                 │
│  - triton headers   │  │  - 链接 TritonIR 库              │
│  - .ttir 文件       │  │  - 解析 .ttir → ascend dialect   │
└─────────────────────┘  └─────────────────────────────────┘
```

---

## 前置要求

- CMake >= 3.20
- Ninja (推荐)
- clang >= 15, lld >= 15
- Python 3.8+
- 约 40GB 磁盘空间

---

## Step 0: 获取子模块
 
所有外部依赖通过 git submodules 管理:
 
```bash
git submodule update --init thirdparty/triton-ascend
git -C thirdparty/triton-ascend submodule update --init --depth 1 third_party/ascend/AscendNPU-IR
```
 
这会拉取 `thirdparty/triton-ascend` 及其必要子模块。默认流程不需要递归初始化
`AscendNPU-IR/third-party/llvm-project`。
 
---
 
## Step 1: 编译 LLVM/MLIR

使用仓库内置的构建脚本:

```bash
export LLVM_INSTALL_PREFIX=$HOME/llvm-install  # 自定义安装路径
./scripts/build_llvm.sh
```
 
或者手动编译 (submodule 已包含正确的 LLVM commit):

```bash
export LLVM_INSTALL_PREFIX=$HOME/llvm-install

cd thirdparty/llvm-project
mkdir build && cd build
cmake -G Ninja ../llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
  -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX}

ninja install
cd ../../..
```

---

## Step 2: 编译 Triton-Ascend

triton-ascend 已作为子模块位于 `thirdparty/triton-ascend`。如果未做额外覆盖，
下面命令会直接使用该 mocked 子模块源码:

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
```

记下构建目录:

```bash
export TRITON_BUILD_DIR=$(ls -d $PWD/build/cmake.* | head -1)
echo "Triton build dir: $TRITON_BUILD_DIR"
cd ../../..
```

---

## Step 3: 编译 TritonSim

### 方式 A: 启用 Triton 支持 (子模块自动发现)

```bash
mkdir build && cd build

cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=${LLVM_INSTALL_PREFIX}/lib/cmake/mlir \
  -DLLVM_DIR=${LLVM_INSTALL_PREFIX}/lib/cmake/llvm \
  -DTRITON_BUILD_DIR=${TRITON_BUILD_DIR}

ninja
```

> CMake 会自动从 `thirdparty/triton-ascend` 发现 Triton 源码。如需指定外部路径，可传 `-DTRITON_SRC_DIR=<path>`。
 
### 方式 B: 不启用 Triton

```bash
cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=${LLVM_INSTALL_PREFIX}/lib/cmake/mlir \
  -DLLVM_DIR=${LLVM_INSTALL_PREFIX}/lib/cmake/llvm \
  -DTRITONSIM_ENABLE_TRITON=OFF

ninja
```

---

## Step 4: 配置硬件参数

TritonSim 支持通过 JSON 配置文件定义目标硬件:

```bash
# 使用默认 910B 配置
./bin/tritonsim-opt input.mlir --hardware-config=configs/ascend_910b.json

# 或自定义配置
./bin/tritonsim-opt input.mlir --hardware-config=my_hardware.json
```

详见 [configs/README.md](configs/README.md) 了解硬件配置格式。

---

## 使用示例

### AscendModel dialect (.mlir) — 方式A/B 均可

```bash
# 完整 pipeline (推荐, 使用内置 910B 配置)
./bin/tritonsim-opt test/ascend_ops.mlir -ascend-perf-model

# 自定义硬件配置 (选项传入 pipeline，不是全局 flag)
./bin/tritonsim-opt test/ascend_ops.mlir \
  -ascend-perf-model="hardware-config=configs/ascend_910b.json"

# 分步执行各 pass
./bin/tritonsim-opt test/ascend_ops.mlir \
  -assign-op-ids \
  -estimate-cycles \
  -analyze-pipeline \
  -perf-report
```

### Triton IR (.ttir) — 方式A (启用 Triton 支持)

```bash
# tritonsim-opt 直接接受 .ttir (tt dialect 已注册)
./bin/tritonsim-opt kernel.ttir -ascend-perf-model
```

### Triton IR (.ttir) — 方式B (未启用 Triton 支持)

方式B 构建不包含 `tt` dialect，需先用 triton-ascend 的 `triton-opt` 将 `.ttir`
转为通用 MLIR generic 格式，再传入 `tritonsim-opt`:

```bash
# 设置 triton-opt 路径 (来自 triton-ascend 构建)
TRITON_OPT=$(ls -d /path/to/triton-ascend/python/build/cmake.*/bin/triton-opt)

# 两阶段 pipeline
$TRITON_OPT kernel.ttir \
  --allow-unregistered-dialect --mlir-print-op-generic | \
./bin/tritonsim-opt - \
  --allow-unregistered-dialect \
  -ascend-perf-model="loop-trip-counts=<N>"
```

`loop-trip-counts=N` 指定 kernel 主循环的实际迭代次数 (运行时值)。

**示例: 分析 Flash Attention dump**

```bash
$TRITON_OPT triton_dumps_fa/<hash>/_attn_fwd.ttir \
  --allow-unregistered-dialect --mlir-print-op-generic | \
./bin/tritonsim-opt - \
  --allow-unregistered-dialect \
  -ascend-perf-model='loop-trip-counts=1'
```
