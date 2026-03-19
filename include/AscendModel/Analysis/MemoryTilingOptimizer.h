//===----------------------------------------------------------------------===//
// MemoryTilingOptimizer.h - Memory-centric tiling optimization
//
// Based on Orojenesis methodology (ISCA 2024):
// "Mind the Gap: Attainable Data Movement and Operational Intensity Bounds"
//
// This module provides:
// 1. Memory access modeling for tensor operations
// 2. Pareto frontier analysis (buffer size vs memory accesses)
// 3. Tiling optimization targeting minimum data movement
// 4. Support for Cube, Vector, and CV fusion operators
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MODEL_MEMORY_TILING_OPTIMIZER_H
#define ASCEND_MODEL_MEMORY_TILING_OPTIMIZER_H

#include "AscendModel/HardwareConfig.h"
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace mlir {
namespace ascend {

//===----------------------------------------------------------------------===//
// Operator Categories
//===----------------------------------------------------------------------===//

enum class OpCategory {
  Matmul,       // C[m,n] += A[m,k] * B[k,n]
  BMM,          // Batched matmul
  Softmax,      // Row-wise softmax
  LayerNorm,    // Layer normalization
  Elementwise,  // Element-wise ops (add, mul, etc.)
  Reduction,    // Sum, max along dimension
  CVFusion,     // Fused Cube + Vector
  Unknown
};

//===----------------------------------------------------------------------===//
// Tiling Configuration
//===----------------------------------------------------------------------===//

struct MemoryTilingConfig {
  std::map<char, int64_t> tileSizes;  // Tile size per dimension
  
  // For CV fusion with independent tiling
  std::map<char, int64_t> cubeTileSizes;
  std::map<char, int64_t> vectorTileSizes;
  bool independentCV = false;
  
  int64_t getTile(char dim, int64_t defaultSize = 1) const {
    auto it = tileSizes.find(dim);
    return (it != tileSizes.end()) ? it->second : defaultSize;
  }
  
  void setTile(char dim, int64_t size) {
    tileSizes[dim] = size;
  }
};

//===----------------------------------------------------------------------===//
// Tiling Result with Memory-Centric Metrics
//===----------------------------------------------------------------------===//

struct MemoryTilingResult {
  MemoryTilingConfig config;
  
  //=== Primary optimization metrics ===//
  int64_t totalMemoryAccesses = 0;    // Bytes moved from/to backing store
  int64_t bufferSizeRequired = 0;     // On-chip buffer requirement (bytes)
  
  // Per-tensor breakdown
  std::map<std::string, int64_t> tensorAccesses;
  std::map<std::string, int64_t> tensorBufferSizes;
  
  //=== Derived metrics ===//
  double operationalIntensity = 0.0;  // FLOPs / byte moved
  double dataReuseRatio = 0.0;        // Algo min / actual accesses (higher = better)
  
  //=== Compute metrics (secondary) ===//
  int64_t totalCycles = 0;
  int64_t cubeCycles = 0;
  int64_t vectorCycles = 0;
  
  //=== Validity ===//
  bool valid = false;
  std::string invalidReason;
  
  // Pareto dominance
  bool dominates(const MemoryTilingResult &other) const;
};

//===----------------------------------------------------------------------===//
// Operation Descriptor (Einsum-style)
//===----------------------------------------------------------------------===//

struct OpDescriptor {
  std::string name;
  OpCategory category;
  
  // Problem dimensions
  std::map<char, int64_t> dimSizes;  // e.g., {'m': 1024, 'n': 1024, 'k': 512}
  
  // Dimension classification
  std::set<char> outputDims;      // Dimensions in output
  std::set<char> reductionDims;   // Contracted dimensions
  std::set<char> batchDims;       // Batch dimensions
  
  // Compute characteristics
  int64_t flopsPerOutputElement = 0;
  int64_t elementBytes = 2;
  
  int64_t getDimSize(char dim) const {
    auto it = dimSizes.find(dim);
    return (it != dimSizes.end()) ? it->second : 1;
  }
  
  int64_t getTotalFlops() const;
  int64_t getAlgorithmicMinAccesses() const;
};

//===----------------------------------------------------------------------===//
// Pareto Frontier
//===----------------------------------------------------------------------===//

class ParetoFrontier {
public:
  void addPoint(const MemoryTilingResult &result);
  
  // Get best tiling within buffer constraint
  MemoryTilingResult getBestForBuffer(int64_t maxBuffer) const;
  
  // Get all Pareto-optimal points
  const std::vector<MemoryTilingResult>& getPoints() const { return points; }
  
  // Print ski-slope diagram
  void printSkiSlope(llvm::raw_ostream &os) const;
  
  // Get maximal effectual buffer size (where curve flattens)
  int64_t getMaxEffectualBufferSize() const;

private:
  std::vector<MemoryTilingResult> points;
};

//===----------------------------------------------------------------------===//
// Memory Access Model
//===----------------------------------------------------------------------===//

class MemoryAccessModel {
public:
  explicit MemoryAccessModel(const HardwareConfig &config);
  
  // Evaluate tiling for different operation types
  MemoryTilingResult evaluateMatmul(int64_t M, int64_t N, int64_t K,
                                     const MemoryTilingConfig &tiling,
                                     int64_t elementBytes);
  
  MemoryTilingResult evaluateBMM(int64_t B, int64_t M, int64_t N, int64_t K,
                                  const MemoryTilingConfig &tiling,
                                  int64_t elementBytes);
  
  MemoryTilingResult evaluateSoftmax(int64_t M, int64_t N,
                                      const MemoryTilingConfig &tiling,
                                      int64_t elementBytes);
  
  MemoryTilingResult evaluateElementwise(const std::vector<int64_t> &shape,
                                          const std::vector<int64_t> &tileShape,
                                          int64_t elementBytes,
                                          int64_t opsPerElement = 1);
  
  MemoryTilingResult evaluateCVFusion(const std::vector<OpDescriptor> &ops,
                                       const MemoryTilingConfig &tiling,
                                       int64_t elementBytes);
  
  // Get memory hierarchy sizes
  int64_t getL1Size() const { return l1Size; }
  int64_t getUBSize() const { return ubSize; }
  int64_t getL0ASize() const { return l0aSize; }
  int64_t getL0BSize() const { return l0bSize; }
  int64_t getL0CSize() const { return l0cSize; }

private:
  const HardwareConfig &hwConfig;
  int64_t l1Size, ubSize, l0aSize, l0bSize, l0cSize;
  double hbmBandwidth;
};

//===----------------------------------------------------------------------===//
// Tiling Optimizer (Main API)
//===----------------------------------------------------------------------===//

class MemoryTilingOptimizer {
public:
  explicit MemoryTilingOptimizer(const HardwareConfig &config);
  
  // Build Pareto frontier for given operation
  ParetoFrontier buildParetoFrontier(const OpDescriptor &op);
  
  // Build frontier for fused operations
  ParetoFrontier buildFusionParetoFrontier(const std::vector<OpDescriptor> &ops);
  
  // Get optimal tiling given buffer budget
  MemoryTilingResult optimizeForBuffer(const OpDescriptor &op, 
                                        int64_t bufferBudget);
  
  // Get optimal tiling for minimum memory accesses (unconstrained buffer)
  MemoryTilingResult optimizeForMinAccesses(const OpDescriptor &op);
  
  // Configure search parameters
  void setSearchRange(int64_t minTile, int64_t maxTile, int64_t step);
  void setVerbose(bool v) { verbose = v; }

private:
  std::unique_ptr<MemoryAccessModel> model;
  const HardwareConfig &hwConfig;
  
  int64_t minTile = 16;
  int64_t maxTile = 256;
  int64_t tileStep = 16;
  bool verbose = false;
};

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Calculate algorithmic minimum accesses for matmul
inline int64_t matmulAlgoMinAccesses(int64_t M, int64_t N, int64_t K, int64_t elemBytes) {
  return (M * K + K * N + M * N) * elemBytes;
}

/// Calculate peak operational intensity for matmul
inline double matmulPeakOI(int64_t M, int64_t N, int64_t K) {
  // OI_peak = 2MNK / (MK + KN + MN) ≈ min(M, N, K) for balanced shapes
  int64_t flops = 2L * M * N * K;
  int64_t minAccesses = M * K + K * N + M * N;
  return static_cast<double>(flops) / minAccesses;
}

/// Calculate maximal effectual buffer size for matmul
/// From Orojenesis: approximately size of smallest operand + smallest dim + 1
inline int64_t matmulMaxEffectualBuffer(int64_t M, int64_t N, int64_t K, int64_t elemBytes) {
  int64_t sizeA = M * K * elemBytes;
  int64_t sizeB = K * N * elemBytes;
  int64_t sizeC = M * N * elemBytes;
  return std::min({sizeA, sizeB, sizeC}) + std::min({M, N, K}) * elemBytes + elemBytes;
}

} // namespace ascend
} // namespace mlir

#endif // ASCEND_MODEL_MEMORY_TILING_OPTIMIZER_H
