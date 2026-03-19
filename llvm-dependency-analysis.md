# LLVM 依赖关系调查报告

## 调查问题
> triton-ascend 的 submodule AscendNPU-IR 是否已包含 LLVM，可以共用？

## 结论：前提不成立

**AscendNPU-IR 不是 triton-ascend 的 submodule。** 它们是 Ascend 组织下的两个独立仓库。

---

## 三个仓库的实际关系

```
GitHub: Ascend 组织
├── triton-ascend          ← Triton 编译框架（昇腾适配）
│   ├── triton @ 9641643   ← submodule: upstream triton-lang/triton
│   ├── ascend/            ← 昇腾 plugin 代码
│   └── .gitmodules        ← 仅引用 triton, 无 AscendNPU-IR
│
├── AscendNPU-IR           ← 独立仓库（毕昇编译器 IR）
│   ├── bishengir/         ← 源代码
│   ├── third-party/       ← 第三方依赖 (内容未知)
│   └── build-tools/
│
└── 其他项目 (torchair, vllm-ascend, ...)
```

### 关键发现

1. **triton-ascend 的唯一 submodule** 是 `triton`（来自 triton-lang/triton），**不是** AscendNPU-IR

2. **AscendNPU-IR** 是完全独立的项目，用于昇腾毕昇编译器（BishengIR），
   与 Triton 编译路径无关

3. **LLVM 必须单独构建**，triton-ascend 官方文档明确要求：
   ```bash
   git clone --no-checkout https://github.com/llvm/llvm-project.git
   cd llvm-project
   git checkout b5cc222d7429fe6f18c787f633d5262fac2e676f  # 指定版本
   # 构建 → 安装到 ${LLVM_INSTALL_PREFIX}
   ```
   然后通过 `LLVM_SYSPATH=${LLVM_INSTALL_PREFIX}` 传给 triton-ascend

---

## 当前 Session 10 方案已是最优

```
thirdparty/llvm-project/           ← LLVM 源码 (submodule, 只编译一次)
    build/install/                 ← LLVM 安装目录
        ↓ (共享)          ↓ (共享)
thirdparty/triton-ascend/  ←→  TritonSim (本项目)
    ├── triton/             ├── lib/
    ├── ascend/             ├── include/AscendModel/
    └── (源码)              └── thirdparty/triton-dialect/
```

**无法省掉 LLVM submodule。** 因为：
- triton-ascend 不捆绑 LLVM
- AscendNPU-IR 不是 triton-ascend 的一部分
- 即使 AscendNPU-IR 的 third-party/ 含有 LLVM，它也与我们的项目无关

## 信息来源
- triton-ascend 安装文档: github.com/Ascend/triton-ascend/blob/master/docs/sources/getting-started/installation.md
- AscendNPU-IR 仓库: github.com/Ascend/AscendNPU-IR
- Gitee 上 triton-ascend 的目录结构 (显示 .gitmodules 和 triton @ 9641643)
