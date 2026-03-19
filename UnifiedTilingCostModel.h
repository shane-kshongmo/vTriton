//===----------------------------------------------------------------------===//
// UnifiedTilingCostModel.h - Comprehensive tiling cost model
//
// Combines three key aspects:
// 1. Memory Access Minimization (Orojenesis-inspired)
// 2. Tile-level Pipeline Overlap (Cube-Vector parallelism)
// 3. Dependency-aware Scheduling (producer-consumer constraints)
//
// The unified cost function balances:
// - Data movement cost (dominant for memory-bound ops)
// - Compute throughput (dominant for compute-bound ops)
// - Pipeline efficiency (critical for CV fusion)
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MODEL_UNIFIED_TILING_COST_MODEL_H
#define ASCEND_MODEL_UNIFIED_TILING_COST_MODEL_H

#include "AscendModel/HardwareConfig.h"
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <functional>

namespace mlir {
namespace ascend {

//===----------------------------------------------------------------------===//
// Hardware Unit Types
//===----------------------------------------------------------------------===//

enum class HWUnit {
  Cube,       // Matrix engine (systolic array)
  Vector,     // Vector processing unit
  MTE2,       // Memory transfer engine (load)
  MTE3,       // Memory transfer engine (store)
  Scalar      // Scalar unit
};

//===----------------------------------------------------------------------===//
// Operation Types (generalized)
//===----------------------------------------------------------------------===//

enum class OpType {
  // Cube operations
  Matmul,           // C[m,n] += A[m,k] * B[k,n]
  BatchedMatmul,    // C[b,m,n] += A[b,m,k] * B[b,k,n]
  Conv2D,           // Convolution (lowered to matmul)
  
  // Vector operations
  Softmax,          // Row-wise: exp(x-max) / sum(exp(x-max))
  LayerNorm,        // (x - mean) / std * gamma + beta
  Elementwise,      // Add, Mul, Activation, etc.
  Reduction,        // Sum, Max, etc.
  Cast,             // Type conversion
  
  // Memory operations
  Load,             // HBM -> L1/UB
  Store,            // L1/UB -> HBM
  
  // Fusion patterns
  FlashAttention,   // Q@K -> Softmax -> @V
  FusedMLP,         // Linear -> Activation -> Linear
  
  Unknown
};

//===----------------------------------------------------------------------===//
// Dimension Descriptor
//===----------------------------------------------------------------------===//

struct DimDesc {
  char label;           // 'm', 'n', 'k', 'b', etc.
  int64_t size;         // Full problem size
  int64_t tileSize;     // Tile size
  bool isReduction;     // Is this a contracted dimension?
  bool isBatch;         // Is this a batch dimension?
  
  int64_t getNumTiles() const {
    return (size + tileSize - 1) / tileSize;
  }
};

//===----------------------------------------------------------------------===//
// Tensor Descriptor with Access Pattern
//===----------------------------------------------------------------------===//

struct TensorDesc {
  std::string name;
  std::vector<char> dims;           // Dimension labels
  std::map<char, int64_t> shape;    // Full shape
  int64_t elementBytes = 2;
  
  // Access pattern for reuse analysis
  // reuseDims: dimensions that DON'T affect this tensor, enabling reuse
  // e.g., for A[m,k] in matmul, reuseDims = {'n'}
  std::set<char> reuseDims;
  
  int64_t getFullSize() const;
  int64_t getTileSize(const std::map<char, int64_t> &tileSizes) const;
  
  // Calculate backing store accesses with given tiling
  int64_t calculateAccesses(const std::map<char, int64_t> &tileSizes,
                            const std::map<char, int64_t> &problemSizes) const;
};

//===----------------------------------------------------------------------===//
// Operation Descriptor
//===----------------------------------------------------------------------===//

struct OpDesc {
  std::string name;
  OpType type;
  HWUnit hwUnit;                    // Primary execution unit
  
  std::vector<TensorDesc> inputs;
  TensorDesc output;
  
  // Dimension info
  std::map<char, DimDesc> dims;
  
  // Compute characteristics
  int64_t flopsPerOutputElement = 0;
  
  // Dependencies
  std::vector<std::string> producerOps;  // Ops that produce inputs
  std::vector<std::string> consumerOps;  // Ops that consume output
  
  int64_t getTotalFlops() const;
  int64_t getOutputElements() const;
};

//===----------------------------------------------------------------------===//
// Tiling Configuration
//===----------------------------------------------------------------------===//

struct UnifiedTilingConfig {
  // Tile sizes per dimension
  std::map<char, int64_t> tileSizes;
  
  // For CV fusion: can have different granularity
  std::map<char, int64_t> cubeTileSizes;
  std::map<char, int64_t> vectorTileSizes;
  bool independentCV = false;
  
  // Double buffering configuration
  bool doubleBufferInputs = true;
  bool doubleBufferOutputs = false;
  
  int64_t getTile(char dim, int64_t defaultVal = 1) const {
    auto it = tileSizes.find(dim);
    return (it != tileSizes.end()) ? it->second : defaultVal;
  }
};

//===----------------------------------------------------------------------===//
// Cost Breakdown Structure
//===----------------------------------------------------------------------===//

struct CostBreakdown {
  //=== Memory Costs ===//
  int64_t totalMemoryAccesses = 0;     // Bytes moved (primary for memory-bound)
  int64_t hbmAccessCycles = 0;         // Cycles for HBM access
  int64_t l1AccessCycles = 0;          // Cycles for L1 access
  
  // Per-tensor breakdown
  std::map<std::string, int64_t> tensorAccesses;
  
  //=== Compute Costs ===//
  int64_t cubeCycles = 0;              // Cube compute cycles
  int64_t vectorCycles = 0;            // Vector compute cycles
  int64_t totalComputeCycles = 0;
  
  //=== Buffer Requirements ===//
  int64_t l0aRequired = 0;             // L0A (Cube input A)
  int64_t l0bRequired = 0;             // L0B (Cube input B)
  int64_t l0cRequired = 0;             // L0C (Cube output)
  int64_t ubRequired = 0;              // UB (Vector buffer)
  int64_t l1Required = 0;              // L1 (shared cache)
  
  //=== Pipeline Metrics ===//
  int64_t serialCycles = 0;            // If no overlap
  int64_t pipelinedCycles = 0;         // With overlap
  double pipelineEfficiency = 0.0;     // serialCycles / (2 * pipelinedCycles)
  
  //=== Tile-level Metrics ===//
  int64_t numTiles = 0;
  int64_t cyclesPerTile = 0;
  double cvBalanceRatio = 0.0;         // min(C,V) / max(C,V), 1.0 = perfect
  double memComputeRatio = 0.0;        // mem_cycles / compute_cycles
  
  //=== Final Cost ===//
  int64_t totalCycles = 0;             // Final estimated cycles
  double operationalIntensity = 0.0;   // FLOPs / byte
  double hardwareUtilization = 0.0;    // Achieved / peak throughput
};

//===----------------------------------------------------------------------===//
// Unified Cost Model
//===----------------------------------------------------------------------===//

class UnifiedTilingCostModel {
public:
  explicit UnifiedTilingCostModel(const HardwareConfig &config);
  
  //==========================================================================//
  // Core Cost Evaluation
  //==========================================================================//
  
  /// Evaluate cost for a single operation
  CostBreakdown evaluate(const OpDesc &op, 
                         const UnifiedTilingConfig &tiling);
  
  /// Evaluate cost for a sequence of operations (with fusion)
  CostBreakdown evaluateFused(const std::vector<OpDesc> &ops,
                               const UnifiedTilingConfig &tiling);
  
  //==========================================================================//
  // Component Cost Models
  //==========================================================================//
  
  /// 1. Memory Access Cost (Orojenesis model)
  struct MemoryCost {
    int64_t totalAccesses;           // Bytes
    int64_t hbmCycles;               // Bandwidth-limited cycles
    std::map<std::string, int64_t> perTensorAccesses;
    double dataReuseRatio;           // algoMin / actual
  };
  MemoryCost computeMemoryCost(const OpDesc &op,
                                const UnifiedTilingConfig &tiling);
  
  /// 2. Compute Cost (fractal-based for Cube, element-based for Vector)
  struct ComputeCost {
    int64_t cubeCycles;
    int64_t vectorCycles;
    int64_t cubeOccupancy;           // Fractal utilization
    int64_t vectorOccupancy;         // Vector lane utilization
  };
  ComputeCost computeComputeCost(const OpDesc &op,
                                  const UnifiedTilingConfig &tiling);
  
  /// 3. Pipeline Overlap Cost
  struct PipelineCost {
    int64_t serialCycles;            // No overlap
    int64_t pipelinedCycles;         // With overlap
    double overlapRatio;             // Fraction of shorter path hidden
    double cvBalance;                // Balance between Cube and Vector
    int64_t effectivePipelineDepth;  // Number of overlapped stages
  };
  PipelineCost computePipelineCost(const MemoryCost &mem,
                                    const ComputeCost &compute,
                                    const UnifiedTilingConfig &tiling,
                                    int64_t numTiles);
  
  /// 4. Buffer Constraint Check
  struct BufferCheck {
    bool valid;
    std::string reason;
    int64_t l0aUsed, l0bUsed, l0cUsed, ubUsed, l1Used;
  };
  BufferCheck checkBufferConstraints(const OpDesc &op,
                                      const UnifiedTilingConfig &tiling);
  
  //==========================================================================//
  // Fusion-specific Analysis
  //==========================================================================//
  
  /// Analyze producer-consumer dependencies
  struct DependencyInfo {
    std::string producer;
    std::string consumer;
    std::string intermediateTensor;
    int64_t intermediateSize;
    bool canFuse;                    // Can intermediate stay on-chip?
    int64_t savedAccesses;           // Bytes saved by fusion
  };
  std::vector<DependencyInfo> analyzeDependencies(
      const std::vector<OpDesc> &ops,
      const UnifiedTilingConfig &tiling);
  
  /// Check fusion compatibility
  bool canFuseOps(const OpDesc &producer, 
                  const OpDesc &consumer,
                  const UnifiedTilingConfig &tiling);
  
  //==========================================================================//
  // Utility Methods
  //==========================================================================//
  
  const HardwareConfig& getConfig() const { return hwConfig; }
  
  // Get hardware limits
  int64_t getL0ASize() const { return l0aSize; }
  int64_t getL0BSize() const { return l0bSize; }
  int64_t getL0CSize() const { return l0cSize; }
  int64_t getUBSize() const { return ubSize; }
  int64_t getL1Size() const { return l1Size; }
  double getHBMBandwidth() const { return hbmBandwidth; }
  int64_t getCubePeakThroughput() const { return cubePeakThroughput; }
  int64_t getVectorWidth() const { return vectorWidth; }

private:
  const HardwareConfig &hwConfig;
  
  // Cached hardware parameters
  int64_t l0aSize, l0bSize, l0cSize, ubSize, l1Size;
  double hbmBandwidth;              // GB/s
  int64_t cubePeakThroughput;       // MACs/cycle
  int64_t vectorWidth;              // Elements/cycle
  int64_t mte2Bandwidth;            // Bytes/cycle (load)
  int64_t mte3Bandwidth;            // Bytes/cycle (store)
  double clockFreqGHz;
  
  // Fractal sizes per precision
  std::map<int, std::tuple<int,int,int>> fractalSizes;  // bits -> (M,K,N)
  
  // Startup latencies
  int64_t cubeStartupLatency;
  int64_t vectorStartupLatency;
  int64_t mte2StartupLatency;
  int64_t mte3StartupLatency;
};

//===----------------------------------------------------------------------===//
// Cost Function for Optimization
//===----------------------------------------------------------------------===//

/// Unified cost function that can be used for tiling search
/// Returns a scalar cost value (lower is better)
class TilingCostFunction {
public:
  enum class Objective {
    MinMemoryAccess,      // Minimize data movement
    MinCycles,            // Minimize total cycles
    MaxThroughput,        // Maximize FLOPs/second
    Balanced              // Balance memory and compute
  };
  
  TilingCostFunction(UnifiedTilingCostModel &model, 
                     Objective obj = Objective::MinCycles)
      : costModel(model), objective(obj) {}
  
  /// Compute cost for single op
  double operator()(const OpDesc &op,
                    const UnifiedTilingConfig &tiling) const;
  
  /// Compute cost for fused ops
  double operator()(const std::vector<OpDesc> &ops,
                    const UnifiedTilingConfig &tiling) const;
  
  /// Set weights for balanced objective
  void setWeights(double memWeight, double computeWeight, double pipelineWeight) {
    this->memWeight = memWeight;
    this->computeWeight = computeWeight;
    this->pipelineWeight = pipelineWeight;
  }
  
  /// Get detailed breakdown for the cost
  CostBreakdown getBreakdown(const OpDesc &op,
                              const UnifiedTilingConfig &tiling) const {
    return costModel.evaluate(op, tiling);
  }

private:
  UnifiedTilingCostModel &costModel;
  Objective objective;
  
  // Weights for balanced objective
  double memWeight = 1.0;
  double computeWeight = 1.0;
  double pipelineWeight = 0.5;
};

//===----------------------------------------------------------------------===//
// Tiling Search with Unified Cost
//===----------------------------------------------------------------------===//

class UnifiedTilingOptimizer {
public:
  UnifiedTilingOptimizer(const HardwareConfig &config);
  
  /// Search for optimal tiling
  struct SearchResult {
    UnifiedTilingConfig bestConfig;
    CostBreakdown bestCost;
    std::vector<std::pair<UnifiedTilingConfig, CostBreakdown>> paretoFrontier;
    int64_t configurationsExplored;
  };
  
  /// Optimize for single operation
  SearchResult optimize(const OpDesc &op,
                        TilingCostFunction::Objective objective =
                            TilingCostFunction::Objective::MinCycles);
  
  /// Optimize for fused operations
  SearchResult optimizeFused(const std::vector<OpDesc> &ops,
                              TilingCostFunction::Objective objective =
                                  TilingCostFunction::Objective::MinCycles);
  
  /// Set search parameters
  void setSearchRange(char dim, int64_t minTile, int64_t maxTile, int64_t step);
  void setDefaultRange(int64_t minTile, int64_t maxTile, int64_t step);
  void setVerbose(bool v) { verbose = v; }
  
  /// Get cost model for direct evaluation
  UnifiedTilingCostModel& getCostModel() { return costModel; }

private:
  UnifiedTilingCostModel costModel;
  
  // Search ranges per dimension
  std::map<char, std::tuple<int64_t, int64_t, int64_t>> searchRanges;
  int64_t defaultMinTile = 16;
  int64_t defaultMaxTile = 256;
  int64_t defaultStep = 16;
  
  bool verbose = false;
  
  // Search strategies
  SearchResult exhaustiveSearch(const OpDesc &op,
                                 const TilingCostFunction &costFn);
  SearchResult heuristicSearch(const OpDesc &op,
                                const TilingCostFunction &costFn);
};

} // namespace ascend
} // namespace mlir

#endif // ASCEND_MODEL_UNIFIED_TILING_COST_MODEL_H
