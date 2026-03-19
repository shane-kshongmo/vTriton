# Ascend NPU 性能建模寻优设计

## 概述

基于我们的 AscendModel IR 和 Roofline 性能模型，可以在编译期预先探索多种优化配置，选择最优方案。

## 1. Tiling 寻优

### 1.1 MatMul Tiling

**搜索空间**:
```
tile_m ∈ {16, 32, 64, 128, 256}
tile_n ∈ {16, 32, 64, 128, 256}  
tile_k ∈ {16, 32, 64, 128, 256}
```

**约束条件**:
```cpp
// L0A: 存储 A 矩阵 tile (tile_m × tile_k)
size_t l0a_usage = tile_m * tile_k * sizeof(fp16);
assert(l0a_usage <= L0A_SIZE);  // 64KB

// L0B: 存储 B 矩阵 tile (tile_k × tile_n)  
size_t l0b_usage = tile_k * tile_n * sizeof(fp16);
assert(l0b_usage <= L0B_SIZE);  // 64KB

// L0C: 存储 C 矩阵 tile (tile_m × tile_n)
size_t l0c_usage = tile_m * tile_n * sizeof(fp32);
assert(l0c_usage <= L0C_SIZE);  // 256KB

// L1: 存储所有 tiles + double buffering
size_t l1_usage = 2 * (l0a_usage + l0b_usage) + l0c_usage;
assert(l1_usage <= L1_SIZE);  // 1MB
```

**代价模型**:
```cpp
int64_t estimateMatmulCycles(int M, int N, int K, 
                              int tile_m, int tile_n, int tile_k,
                              const HardwareConfig& config) {
  // 计算迭代次数
  int m_iters = (M + tile_m - 1) / tile_m;
  int n_iters = (N + tile_n - 1) / tile_n;
  int k_iters = (K + tile_k - 1) / tile_k;
  
  // 每个 tile 的计算量
  int64_t tile_flops = 2 * tile_m * tile_n * tile_k;
  int64_t compute_cycles = tile_flops / config.getCubeFLOPsPerCycle();
  
  // 数据搬运量
  // A: 每个 (m,k) tile 加载一次，被 n_iters 复用
  int64_t a_loads = m_iters * k_iters;
  int64_t a_bytes_per_load = tile_m * tile_k * 2;
  
  // B: 每个 (k,n) tile 加载一次，被 m_iters 复用
  int64_t b_loads = k_iters * n_iters;
  int64_t b_bytes_per_load = tile_k * tile_n * 2;
  
  // C: 每个 (m,n) tile 写回一次
  int64_t c_stores = m_iters * n_iters;
  int64_t c_bytes_per_store = tile_m * tile_n * 4;
  
  int64_t total_load_bytes = a_loads * a_bytes_per_load + 
                              b_loads * b_bytes_per_load;
  int64_t total_store_bytes = c_stores * c_bytes_per_store;
  
  int64_t load_cycles = total_load_bytes / config.getMTE2BytesPerCycle();
  int64_t store_cycles = total_store_bytes / config.getFixPipeBytesPerCycle();
  int64_t total_compute = m_iters * n_iters * k_iters * compute_cycles;
  
  // Roofline: max(compute, memory)
  return std::max(total_compute, load_cycles + store_cycles);
}
```

### 1.2 Vector Tiling

**搜索空间**:
```
vector_tile ∈ {128, 256, 512, 1024, 2048, 4096}
```

**约束**:
```cpp
// UB 限制
size_t ub_usage = vector_tile * element_size * num_operands;
assert(ub_usage <= UB_SIZE);  // 256KB
```

## 2. Double/Triple Buffering 寻优

**搜索空间**:
```
buffer_stages ∈ {1, 2, 3}  // Single, Double, Triple buffering
```

**代价模型**:
```cpp
int64_t estimateBufferingCycles(int64_t compute_cycles, 
                                 int64_t load_cycles,
                                 int64_t store_cycles,
                                 int buffer_stages,
                                 int num_iterations) {
  if (buffer_stages == 1) {
    // No overlap: sequential execution
    return num_iterations * (load_cycles + compute_cycles + store_cycles);
  } else if (buffer_stages == 2) {
    // Double buffering: overlap load[i+1] with compute[i]
    int64_t steady_state = std::max(compute_cycles, load_cycles);
    int64_t prologue = load_cycles;  // First load
    int64_t epilogue = compute_cycles + store_cycles;  // Last compute + store
    return prologue + (num_iterations - 1) * steady_state + epilogue;
  } else {
    // Triple buffering: full overlap
    int64_t steady_state = std::max({compute_cycles, load_cycles, store_cycles});
    int64_t prologue = 2 * load_cycles;
    int64_t epilogue = 2 * store_cycles;
    return prologue + (num_iterations - 2) * steady_state + epilogue;
  }
}

size_t estimateBufferMemory(int tile_size, int buffer_stages) {
  return tile_size * buffer_stages;
}
```

## 3. 算子融合寻优

### 3.1 融合模式识别

```cpp
// 可融合的模式
enum FusionPattern {
  MATMUL_BIAS_ACTIVATION,  // MatMul + Add + ReLU/GELU
  MATMUL_REDUCE,           // MatMul + ReduceSum/ReduceMax
  ELEMENTWISE_CHAIN,       // Add + Mul + Exp
  SOFTMAX,                 // Max + Sub + Exp + Sum + Div
};

struct FusionCandidate {
  SmallVector<Operation*> ops;
  FusionPattern pattern;
  int64_t unfused_cycles;
  int64_t fused_cycles;
  int64_t memory_savings;  // 减少的中间结果
};
```

### 3.2 融合收益估算

```cpp
int64_t estimateFusionBenefit(const FusionCandidate& candidate) {
  // 融合前: 每个 op 独立执行，中间结果写回
  int64_t unfused = 0;
  for (auto* op : candidate.ops) {
    unfused += estimateOpCycles(op);
    unfused += estimateIntermediateStoreCycles(op);
  }
  
  // 融合后: 单个 kernel，无中间结果
  int64_t fused = estimateFusedKernelCycles(candidate);
  
  return unfused - fused;
}
```

## 4. 流水线调度寻优

### 4.1 调度策略

```cpp
enum ScheduleStrategy {
  ASAP,           // As Soon As Possible
  ALAP,           // As Late As Possible
  LIST_SCHEDULE,  // 基于优先级的调度
  MODULO_SCHEDULE // 软件流水
};
```

### 4.2 流水线深度搜索

```cpp
struct PipelineConfig {
  int mte2_depth;   // MTE2 预取深度
  int cube_depth;   // Cube 流水深度
  int vector_depth; // Vector 流水深度
  int mte3_depth;   // MTE3 写回深度
};

int64_t estimatePipelineCycles(const PipelineConfig& config,
                                const SmallVector<Operation*>& ops) {
  // 模拟流水线执行
  PipelineSimulator sim(config);
  for (auto* op : ops) {
    sim.schedule(op);
  }
  return sim.getTotalCycles();
}
```

## 5. 寻优 Pass 设计

### 5.1 AutoTunePass

```cpp
struct AutoTunePass : public PassWrapper<AutoTunePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    const HardwareConfig& config = getHardwareConfig();
    
    // 1. 收集优化目标
    SmallVector<TuningTarget> targets;
    module.walk([&](Operation* op) {
      if (auto matmul = dyn_cast<MatmulOp>(op)) {
        targets.push_back(createMatmulTarget(matmul));
      }
    });
    
    // 2. 对每个目标进行寻优
    for (auto& target : targets) {
      auto bestConfig = searchBestConfig(target, config);
      applyConfig(target.op, bestConfig);
    }
  }
  
  TilingConfig searchBestConfig(const TuningTarget& target,
                                 const HardwareConfig& hw) {
    TilingConfig best;
    int64_t bestCycles = INT64_MAX;
    
    // Grid search or Bayesian optimization
    for (auto& config : generateSearchSpace(target)) {
      if (!checkConstraints(config, hw))
        continue;
      
      int64_t cycles = estimateCycles(target, config, hw);
      if (cycles < bestCycles) {
        bestCycles = cycles;
        best = config;
      }
    }
    return best;
  }
};
```

### 5.2 输出格式

寻优结果可以：
1. 作为 IR 属性附加到操作上
2. 生成配置文件供后端使用
3. 输出可视化报告

```mlir
ascend.matmul %a, %b {
  M = 1024, N = 1024, K = 512,
  tile_m = 128, tile_n = 128, tile_k = 64,
  buffer_stages = 2,
  estimated_cycles = 12345,
  memory_usage = 524288,
  arithmetic_intensity = 85.3
} : (tensor<1024x512xf16>, tensor<512x1024xf16>) -> tensor<1024x1024xf32>
```

## 6. 实现路线图

### Phase 1: 基础框架
- [ ] TilingConfig 数据结构
- [ ] 约束检查器
- [ ] 基础代价模型

### Phase 2: MatMul 寻优
- [ ] Tiling 搜索空间生成
- [ ] Buffering 策略选择
- [ ] 性能估算验证

### Phase 3: 算子融合
- [ ] 融合模式识别
- [ ] 融合收益估算
- [ ] 融合变换实现

### Phase 4: 全局优化
- [ ] 流水线调度
- [ ] 内存分配优化
- [ ] 多核并行策略
