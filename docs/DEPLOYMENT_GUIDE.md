# vTriton 端到端部署与运行指南

> 从源码搭建含 BiShengIR 外部依赖的环境，到 tritonsim-opt 成功分析 HIVM MLIR 并输出调度数据的完整步骤。

---

## 一、依赖版本总览

| 组件 | 版本 | 来源/说明 |
|------|------|----------|
| LLVM / MLIR | **19.1.7** (`cd70802`) | AscendNPU-IR 的 submodule，位于 `thirdparty/triton-ascend/third_party/ascend/AscendNPU-IR/third-party/llvm-project` |
| BiShengIR | **v1.1.0** | 随 AscendNPU-IR 一并编译，与 LLVM 19 绑定 |
| triton-ascend | `865691e` (2026-05-18) | 单 commit 拉取，main 分支 HEAD |
| CMake | **≥ 3.22**（LLVM/MLIR），**≥ 3.28**（BiShengIR 顶层） | 系统自带 3.22 + pip 安装 3.28+ |
| Ninja | **≥ 1.11** | 构建系统 |
| GCC / G++ | **≥ 12** | C++17 编译器 |
| Python | **3.11+** | 运行 dump launcher 和 perfbound |
| CANN | **8.5.0** | 昇腾工具链，提供 bishengir-compile |
| OS | Linux (aarch64) | Ubuntu 20.04 / 22.04 |

---

## 二、端到端步骤

### Step 1: 获取源码并初始化 submodule

```bash
# 在 Linux 服务器上
cd /home/triton_sim
git clone <your-repo-url> vTriton
cd vTriton

# 拉取 triton-ascend 及其嵌套 submodule（含 LLVM 19.1.7）
git submodule update --init --recursive thirdparty/triton-ascend
```

### Step 2: 升级 CMake（BiShengIR 编译需要 3.28）

```bash
python3 -m pip install --upgrade cmake
export PATH=$HOME/.local/bin:$PATH
cmake --version   # 必须 ≥ 3.28
```

### Step 3: 编译 LLVM 19 + MLIR 框架

```bash
LLVM_SRC=$(realpath thirdparty/triton-ascend/third_party/ascend/AscendNPU-IR/third-party/llvm-project)

cd ${LLVM_SRC}

# 确保换行符为 LF（Windows 传输可能引入 CRLF）
git checkout -f llvmorg-19.1.7

mkdir -p build && cd build
cmake -G Ninja ../llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/install

ninja -j$(nproc) install

# 验证
ls install/lib/cmake/mlir/MLIRConfig.cmake && echo "✓ LLVM+MLIR ready"
```

> **预计耗时**：30-60 分钟（取决于 CPU 核心数）

### Step 4: 编译 BiShengIR（HIVM/HACC dialect 库）

BiShengIR 作为 LLVM 的外部项目编译。**注意**：BiShengIR 的 `bishengir-target-spec-tblgen` 工具与 LLVM 19 的 TableGen API 不完全兼容，需要删除工具构建目录使其跳过。

```bash
cd ${LLVM_SRC}/build

ASCNPU_DIR=$(realpath ../..)   # AscendNPU-IR 根目录

# 重新配置 cmake，加入 BiShengIR
cmake -G Ninja ../llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_EXTERNAL_PROJECTS="bishengir" \
  -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR=${ASCNPU_DIR} \
  -DBISHENGIR_BUILD_STANDALONE_IR_ONLY=ON \
  -DMLIR_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/install

# 删除不兼容的 tblgen 工具目录
rm -rf tools/bishengir/bishengir/tools

# 全量编译 BiShengIR（生成 .o 文件）
ninja -j$(nproc)
```

> 此时编译会碰到 `bishengir-target-spec-tblgen/TargetSpecGen.cpp` 的 API 不兼容报错（见七-1），删除工具目录后重新 cmake + ninja 即可。

### Step 5: 手动打包 BiShengIR 静态库

LLVM 的 `add_mlir_library` 默认只生成 `.o` object 文件（object library），需要用 `ar` 打包成 `.a`：

```bash
cd ${LLVM_SRC}/build

B_BUILD=tools/bishengir/bishengir
B_LIB=${B_BUILD}/lib/Dialect

# 为每个 dialect 目录打包
for dir in \
  HIVM/IR HIVM/Utils HACC/IR HACC/Utils HACC/Transforms \
  Annotation/IR HFusion/IR MemRef/IR MemRefExt/IR \
  Tensor/IR Scope/IR Symbol/IR Math/IR Utils; do
  OBJS=$(find ${B_LIB}/${dir}/CMakeFiles -name "*.cpp.o" 2>/dev/null)
  if [ -n "$OBJS" ]; then
    mkdir -p ${B_LIB}/${dir}
    ar crs ${B_LIB}/${dir}/$(basename ${dir}).a ${OBJS} 2>/dev/null
  fi
done

# 按 tritonsim 期望的库名重命名
mv ${B_LIB}/HIVM/IR/IR.a           ${B_LIB}/HIVM/IR/libBiShengIRHIVMDialect.a
mv ${B_LIB}/HIVM/Utils/Utils.a     ${B_LIB}/HIVM/Utils/libBiShengIRHIVMUtils.a
mv ${B_LIB}/HACC/IR/IR.a           ${B_LIB}/HACC/IR/libBiShengIRHACCDialect.a
mv ${B_LIB}/HACC/Utils/Utils.a     ${B_LIB}/HACC/Utils/libBiShengIRHACCUtils.a
mv ${B_LIB}/HACC/Transforms/Transforms.a  ${B_LIB}/HACC/Transforms/libBiShengIRHACCTransforms.a
mv ${B_LIB}/Annotation/IR/IR.a     ${B_LIB}/Annotation/IR/libBiShengIRAnnotationDialect.a
mv ${B_LIB}/HFusion/IR/IR.a        ${B_LIB}/HFusion/IR/libBiShengIRHFusionDialect.a
mv ${B_LIB}/MemRef/IR/IR.a         ${B_LIB}/MemRef/IR/libBiShengIRMemRefDialect.a
mv ${B_LIB}/MemRefExt/IR/IR.a      ${B_LIB}/MemRefExt/IR/libBiShengIRMemRefExtDialect.a
mv ${B_LIB}/Tensor/IR/IR.a         ${B_LIB}/Tensor/IR/libBiShengIRTensorDialect.a
mv ${B_LIB}/Scope/IR/IR.a          ${B_LIB}/Scope/IR/libBiShengIRScopeDialect.a
mv ${B_LIB}/Symbol/IR/IR.a         ${B_LIB}/Symbol/IR/libBiShengIRSymbolDialect.a
mv ${B_LIB}/Math/IR/IR.a           ${B_LIB}/Math/IR/libBiShengIRMathExtDialect.a
mv ${B_LIB}/Utils/Utils.a           ${B_LIB}/Utils/libBiShengIRDialectUtils.a

# 验证
ls ${B_LIB}/HIVM/IR/libBiShengIRHIVMDialect.a && echo "✓ BiShengIR libs ready"
```

### Step 6: 编译 tritonsim-opt（链接 BiShengIR）

```bash
cd /home/triton_sim/vTriton
rm -rf build && mkdir build && cd build

LLVM_BUILD=$(realpath ../thirdparty/triton-ascend/third_party/ascend/AscendNPU-IR/third-party/llvm-project/build)
ASCNPU_SRC=$(realpath ../thirdparty/triton-ascend/third_party/ascend/AscendNPU-IR)

cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=${LLVM_BUILD}/install/lib/cmake/mlir \
  -DLLVM_DIR=${LLVM_BUILD}/install/lib/cmake/llvm \
  -DTRITONSIM_ENABLE_TRITON=OFF \
  -DTRITONSIM_ENABLE_BISHENGIR_HIVM=ON \
  -DTRITONSIM_BISHENGIR_SRC_DIR=${ASCNPU_SRC}/bishengir \
  -DTRITONSIM_BISHENGIR_BUILD_DIR=${LLVM_BUILD}/tools/bishengir/bishengir

ninja
```

> 确认 cmake 输出中显示 `BiShengIR HIVM integration ENABLED` 和 `BiShengIR HIVM: ON`。

### Step 7: 准备 Triton kernel 源码并调整参数

```bash
cd /home/triton_sim/vTriton

# 删除编译器版本不兼容的 kwargs（这些是 A5/910D 专有参数）
sed -i '/enable_mixed_cv=False,/d'     prefill_a5_cvpipe.py
sed -i '/enable_auto_bind_sub_block=True,/d' prefill_a5_cvpipe.py
sed -i '/multibuffer=False,/d'         prefill_a5_cvpipe.py
sed -i '/inject_barrier_all=False,/d'  prefill_a5_cvpipe.py
sed -i '/tile_mix_cube_loop=4,/d'      prefill_a5_cvpipe.py
sed -i '/tile_mix_vector_loop=1,/d'    prefill_a5_cvpipe.py
sed -i '/enable_hivm_auto_cv_balance=True,/d'  prefill_a5_cvpipe.py
sed -i '/disable_auto_cv_workspace_manage=True,/d' prefill_a5_cvpipe.py
sed -i '/enable_code_motion=True,/d'   prefill_a5_cvpipe.py

# 避免 cbuf overflow
sed -i 's/BLOCK_SBS = 256/BLOCK_SBS = 128/' prefill_a5_cvpipe.py
```

### Step 8: Dump HIVM MLIR（含 AIC + AIV 两个核）

```bash
cd /home/triton_sim/vTriton

# 清缓存 + 重新 dump
rm -rf /root/.triton/cache/
rm -rf dumped_mlir && mkdir dumped_mlir

python3 tools/common/triton_dsl_dump_launcher.py \
  --script prefill_a5_cvpipe.py \
  --dump-dir ./dumped_mlir/

# 确认两个核都被 dump（自定义格式 MLIR）
grep "func.func" dumped_mlir/kernel_001.npuir.mlir
# 预期输出两行：_mix_aic (CUBE) 和 _mix_aiv (VECTOR)
```

### Step 9: 运行 tritonsim-opt 分析

```bash
cd /home/triton_sim/vTriton

./build/bin/tritonsim-opt dumped_mlir/kernel_001.npuir.mlir \
  --allow-unregistered-dialect \
  --analyze-hivm="scheduler=des hardware-config=/home/triton_sim/vTriton/configs/ascend_910b3.json des-graph-file=/home/triton_sim/vTriton/docs/prefill_des.json perfetto-trace-file=/home/triton_sim/vTriton/docs/prefill_trace.json"
```

> **注意**：`--allow-unregistered-dialect` 兜底 `arg_attrs` 中的 `tt.*`、`hacc.*` 等未注册属性。HIVM/HACC/Annotation 方言由 BiShengIR 原生解析。

### Step 10: 查看产物

| 文件 | 用途 |
|------|------|
| `docs/prefill_des.json` | JSON 格式的调度数据，可喂给 Python perfbound 算 T_bound |
| `docs/prefill_trace.json` | Perfetto trace 格式，拖入 https://ui.perfetto.dev 可视化 |

---

## 三、核心依赖关系图

```
vTriton
  ├── CMakeLists.txt                    (cmake ≥ 3.20)
  ├── tools/tritonsim-opt/
  │     └── tritonsim-opt.cpp           ← 注册 HIVM + HACC + Annotation dialect
  ├── lib/AscendModel/
  │     └── Analysis/HIVMAnalysis.cpp   ← populateTypedHivmOp() 强类型路径
  └── thirdparty/
        └── triton-ascend/
              └── third_party/ascend/AscendNPU-IR/
                    ├── CMakeLists.txt          (cmake ≥ 3.28)
                    ├── third-party/
                    │     └── llvm-project/     (LLVM 19.1.7)
                    │           └── build/install/
                    │                 ├── lib/cmake/mlir/
                    │                 └── lib/cmake/llvm/
                    └── bishengir/
                          ├── include/bishengir/Dialect/
                          │     ├── HIVM/IR/HIVM.h       (源码头文件)
                          │     ├── HACC/IR/HACC.h
                          │     └── Annotation/IR/Annotation.h
                          └── (生成的 .inc 头文件在 build/tools/bishengir/bishengir/include/)
```

---

## 四、vTriton 项目修改汇总

为支持 BiShengIR 原生解析，对项目做了以下改动：

| 文件 | 改动 | 目的 |
|------|------|------|
| [CMakeLists.txt:63](file:///d:/develop/vTriton/CMakeLists.txt#L63) | 添加 `TRITONSIM_ENABLE_BISHENGIR_HIVM` option 及对应的 include/link 逻辑 | 控制 BiShengIR 编译开关 |
| [tools/tritonsim-opt/tritonsim-opt.cpp:31-33](file:///d:/develop/vTriton/tools/tritonsim-opt/tritonsim-opt.cpp#L31-L33) | `#include` HIVM/HACC/Annotation 头文件 | 为 DialectRegistry 提供方言声明 |
| [tools/tritonsim-opt/tritonsim-opt.cpp:56-59](file:///d:/develop/vTriton/tools/tritonsim-opt/tritonsim-opt.cpp#L56-L59) | `registry.insert<HIVMDialect/HACCDialect/AnnotationDialect>()` | 注册 BiShengIR 方言到 MLIR parser |
| [tools/CMakeLists.txt:45-47](file:///d:/develop/vTriton/tools/CMakeLists.txt#L45-L47) | `target_link_libraries(tritonsim-opt PRIVATE ${TRITONSIM_BISHENGIR_HIVM_LIBS})` | 链接 BiShengIR 静态库 |
| [lib/AscendModel/Analysis/HIVMAnalysis.cpp:658-670](file:///d:/develop/vTriton/lib/AscendModel/Analysis/HIVMAnalysis.cpp#L658-L670) | 新增 `stringifyTypedEvent()` / `stringifyTypedFlag()` | 适配 BiShengIR 的 `EventAttr` / `IntegerAttr` 返回值类型 |
| [lib/AscendModel/Analysis/HIVMAnalysis.cpp:410-443](file:///d:/develop/vTriton/lib/AscendModel/Analysis/HIVMAnalysis.cpp#L410-L443) | `getElementTypeName` 等函数：优先 `MemRefType`/`TensorType` 具体路径，ShapedType 接口兜底 | 避免 unregistered dialect 类型的 interface dispatch 崩溃 |
| [tools/common/triton_dsl_dump_launcher.py:278](file:///d:/develop/vTriton/tools/common/triton_dsl_dump_launcher.py#L278) | 移除 `wrapped_run` 中的 `_exit_success()` | mix kernel 有多个编译变体（AIC+AIV），不能 dump 完第一个就退出 |
| [tools/common/triton_dsl_dump_launcher.py:117-142](file:///d:/develop/vTriton/tools/common/triton_dsl_dump_launcher.py#L117-L142) | `_extract_mlir_module` → `_extract_mlir_modules`，返回列表 | bishengir 输出多个 IR Dump 区块，全部提取 |
| `scripts/hivm_filter.py` | 新增文件 | 无 BiShengIR 时的降级方案（通过字符串替换消除 `#hivm.address_space<gm>` 等解析阻止器） |

---

## 五、BiShengIR 编译后的数据流

```
prefill_a5_cvpipe.py (Triton kernel)
    │
    ▼ bishengir-compile (CANN 8.5.0)
dumped_mlir/kernel_001.npuir.mlir (自定义格式 MLIR)
    │  func.func @..._mix_aic   (CUBE 核)
    │  func.func @..._mix_aiv   (VECTOR 核)
    │
    ▼ tritonsim-opt (--allow-unregistered-dialect)
    ├── HIVMDialect      → 解析 #hivm.address_space<gm>, hivm.hir.* ops
    ├── HACCDialect      → 解析 #hacc.arg_type, #hacc.function_kind
    ├── AnnotationDialect → 解析 annotation.mark
    └── --allow-unregistered-dialect  → 兜底 tt.*, func attrs
    │
    ▼ HIVMAnalysisPass
    ├── populateTypedHivmOp()  → 强类型匹配 load/store/copy/fixpipe 等
    ├── populateGenericHivmOp() → 字符串匹配兜底
    ├── DES / Static scheduler → 周期模拟
    │
    ▼
prefill_des.json  +  prefill_trace.json
```

---

## 六、常见问题速查

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `cmake_minimum_required(VERSION 3.28.0)` | 系统 cmake 太旧 | `pip install --upgrade cmake` |
| `config.guess: Syntax error` | CRLF 换行符 | `git checkout -f llvmorg-19.1.7` |
| `unsupported memory space Attribute` | HIVM 方言未注册 | 检查 `TRITONSIM_HAS_BISHENGIR_HIVM=1` 和 `registry.insert<HIVMDialect>()` |
| `Dialect 'annotation' not found` | Annotation 方言未注册 | `#include "bishengir/Dialect/Annotation/IR/Annotation.h"` + `registry.insert` |
| `cbuf overflow, requires 5013504 bits` | BLOCK_SBS 过大 | `sed -i 's/BLOCK_SBS = 256/BLOCK_SBS = 128/'` |
| `Keyword argument enable_mixed_cv was specified but unrecognised` | 安装的 triton 版本旧于 kernel 编写版本 | sed 删除不认识的 kwargs |
| `ninja: manifest build.ninja still dirty after 100 tries` | 系统时间不正常 | `date -s "2026-06-12 12:00:00"` + 完全重建 build |
| `D:\develop\vTriton_sync.tar.gz` 传到 Linux | Windows 打包，解压后可能有 CRLF | `tar xzf` 后对 `.sh`/`.cmake` 文件 `sed -i 's/\r$//'` |
| `collect2: undefined reference to computeElementwiseLimitation` | BiShengIR `.o` 未全部打包进 `.a` | 全量 ninja 后重新 `ar crs` 打包 |
| `Illegal instruction (core dumped) at _init` | `--unresolved-symbols=ignore-in-object-files` 导致虚函数表指向零地址 | **不要使用此 flag**，补齐缺失符号 |
| 只 dump 了一个核（AIV） | `_exit_success()` 在第一次成功后终止进程 | 删除 `wrapped_run` 中的 `_exit_success()` 调用 |
| mlir 文件末尾混入 `ld.lld: warning:` 垃圾文本 | bishengir-compile stderr 混入输出 | `sed -i '/^ld\.lld/d; /^warning:/d' kernel_001.npuir.mlir` |
| `failed to load hardware config` | 在 build 目录运行，相对路径解析错误 | 使用绝对路径：`hardware-config=/home/triton_sim/vTriton/configs/ascend_910b3.json` |

---

## 七、典型案例详解

### 7.1 BiShengIR tblgen 工具编译失败（LLVM 19 API 不兼容）

**现象**：`TargetSpecGen.cpp:112` `cannot convert SmallVector<const Record*> to SmallVectorImpl<Record*>&`

**根因**：BiShengIR 的 `bishengir-target-spec-tblgen` 工具使用了 `getDirectSuperClasses()` 等 TableGen API。LLVM 19 的签名是 `SmallVectorImpl<Record*>&`，LLVM 20+ 改为 `SmallVectorImpl<const Record*>&`。BiShengIR 源码通过 `#ifdef` 兼容多版本，但 cmake 未正确设置宏。

**方案**：`rm -rf tools/bishengir/bishengir/tools` 删除 tblgen 工具目录，ninja 跳过该工具编译。tritonsim 只需 dialect IR 库，不需要 tblgen 代码生成工具。

### 7.2 HIVMAnalysis.cpp 与 BiShengIR 类型系统对接

**现象**：`EventAttr has no member named getValue`；`getDynamicEventId() returns TypedValue<IntegerType>` 而非 `std::optional<EventAttr>`

**根因**：BiShengIR 生成的 API 中：
- `getStaticEventId()` → `std::optional<EventAttr>`（枚举属性）
- `getDynamicEventId()` → `TypedValue<IntegerType>`（SSA 动态值）
- `getStaticFlagId()` → `std::optional<IntegerAttr>`（整数属性）
- `getDynamicFlagId()` → `TypedValue<IntegerType>`（SSA 动态值）

`EventAttr` 的取值方法是 `getEvent()`（返回枚举），不是 `getValue()`。枚举转字符串用 `mlir::hivm::stringifyEVENT()`。

**方案**：新增两个 helper 函数，只接受静态属性部分：

```cpp
static std::string stringifyTypedEvent(std::optional<mlir::hivm::EventAttr> staticEvent) {
  if (staticEvent)
    return ("event_" + mlir::hivm::stringifyEVENT(staticEvent->getEvent()).str());
  return "";
}

static std::string stringifyTypedFlag(std::optional<mlir::IntegerAttr> staticFlag) {
  if (staticFlag)
    return ("flag_" + std::to_string(staticFlag->getInt()));
  return "";
}
```

动态 event/flag 的值由 `parsed.syncIdValue` 单独保存。

### 7.3 LLVM 19 unregistered dialect 类型导致的 Segfault

**现象**：`PLEASE submit a bug report ... #3 tritonsim-opt llvm::DefaultDoCastIfPossible<ShapedType, Type>` 在 `getElementTypeName()` 中崩溃

**根因**：LLVM 19 的 `--allow-unregistered-dialect` 为未注册操作的结果类型创建对象时，interface table 不完整。`dyn_cast<ShapedType>()` 走 interface dispatch 路径 → `TypeStorage::getAbstractType()` 返回 nullptr → segfault。

**方案**：在 `getElementTypeName()`、`getShapedTypeElementCount()`、`getShapedTypeBytes()` 三个函数中，优先使用 BuiltinDialect 的具体类型安全检查（TypeID 直接匹配，不走 interface dispatch）：

```cpp
// 安全路径：MemRefType / TensorType 的 TypeID 直接匹配
if (auto memref = llvm::dyn_cast<mlir::MemRefType>(type))
    return getElementTypeName(memref.getElementType());
if (auto tensor = llvm::dyn_cast<mlir::TensorType>(type))
    return getElementTypeName(tensor.getElementType());
// 兜底：ShapedType interface dispatch（可能崩溃但仅用于非标准类型）
auto shaped = llvm::dyn_cast<mlir::ShapedType>(type);
```

### 7.4 scf.if 空 else region 被清理步骤误删

**现象**：`'scf.if' op expected 2 regions`

**根因**：filter 清理步骤中的 `re.sub(r"\{\s*\}", "", inp)` 把 `scf.if(%77) ({ ... }, { })` 里的空 else `{ }` 删了，导致只剩一个 region。

**方案**：移除对 `{}` 的全局删除，改用更精确的模式仅在 attribute dict 上下文清理：`re.sub(r"\)\s*\{\s*\}\s*:", "):", inp)`

### 7.5 arg_attrs 括号匹配被 dense 内部的 `]` 截断

**现象**：`arg_attrs = [{...}, ..., {}]` 删除后残留 `%arg43)` 碎片，导致行被截断

**根因**：`arg_attrs` 包含了 `func_dyn_memref_args = dense<[false, true, ...]>`，其中 dense 内部的 `]` 比外层的 `arg_attrs = [...]` 的 `]` 更早出现，简单的字符扫描把 dense 的 `]` 当成 arg_attrs 的结尾。

**方案**：括号扫描时加入 `<>` 深度计数——仅在 `<>` 深度为 0 时才对 `[]` 做 +1/-1 计数。

### 7.6 mix kernel 只 dump 了一个核

**现象**：`dumped_mlir/` 里只有一个 `kernel_001.npuir.mlir`，且只含 `_mix_aiv`

**根因**：
1. `triton_dsl_dump_launcher.py` 的 `wrapped_run` 中 dump 成功后立即 `_exit_success()` → `os._exit(0)` 杀死 Python 进程
2. `_extract_mlir_module()` 只提取 bishengir 输出的第一个 IR Dump 区块就 `break`

**方案**：
1. 移除 `wrapped_run` 中的 `_exit_success()`，让进程自然执行直到 prefill 测试结束
2. 将 `_extract_mlir_module` 改为 `_extract_mlir_modules`（返回列表），用循环提取所有 `// IR Dump After` 区块
