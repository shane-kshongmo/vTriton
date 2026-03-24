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
git submodule update --init --recursive
```
 
这会拉取 `thirdparty/llvm-project` (pinned at commit `b5cc222d`) 和 `thirdparty/triton-ascend`。
 
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

triton-ascend 已作为子模块位于 `thirdparty/triton-ascend`:

```bash
cd thirdparty/triton-ascend
git submodule update --init --depth 1

LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
TRITON_PLUGIN_DIRS=./ascend \
TRITON_BUILD_WITH_CCACHE=true \
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_BUILD_PROTON=OFF \
TRITON_WHEEL_NAME="triton-ascend" \
python3 setup.py develop
```

记下构建目录:

```bash
export TRITON_BUILD_DIR=$(ls -d $PWD/python/build/cmake.* | head -1)
echo "Triton build dir: $TRITON_BUILD_DIR"
cd ../..
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

```bash
# 完整 pipeline
./bin/tritonsim-opt test/softmax.ttir \
  --hardware-config=configs/ascend_910b.json \
  -convert-triton-to-ascend \
  -assign-op-ids \
  -estimate-cycles \
  -analyze-pipeline \
  -perf-report

# 仅分析 ascend dialect
./bin/tritonsim-opt test/ascend_ops.mlir \
  --hardware-config=configs/ascend_910b.json \
  -assign-op-ids \
  -estimate-cycles \
  -analyze-pipeline \
  -perf-report
```
