//===----------------------------------------------------------------------===//
// MemoryTilingOptimizer.cpp - Memory-centric tiling optimization
//
// Based on Orojenesis methodology (ISCA 2024):
// "Mind the Gap: Attainable Data Movement and Operational Intensity Bounds
//  for Tensor Algorithms"
//
// Key principles:
// 1. Minimize data movement (memory accesses) as primary objective
// 2. Consider buffer capacity constraints at each memory level
// 3. Model data reuse opportunities from tiling
// 4. Support fusion to exploit producer-consumer locality
// 5. Generate Pareto-optimal (buffer size, accesses) curves
//
// Generalized to support:
// - Pure Cube (matmul) operations
// - Pure Vector operations (softmax, layernorm, elementwise)
// - CV Fusion operations (FlashAttention, fused MLP)
//===----------------------------------------------------------------------===//

#include "AscendModel/IR/AscendModelDialect.h"
#include "AscendModel/IR/AscendModelInterfaces.h"
#include "AscendModel/Transforms/Passes.h"
#include "AscendModel/HardwareConfig.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace mlir {
namespace ascend {
namespace {

//===----------------------------------------------------------------------===//
// Operator Type Classification
//===----------------------------------------------------------------------===//

enum class OpCategory {
  Matmul,       // Cube-only: C[m,n] += A[m,k] * B[k,n]
  BMM,          // Batched matmul: C[b,m,n] += A[b,m,k] * B[b,k,n]
  Softmax,      // Vector: row-wise softmax
  LayerNorm,    // Vector: layer normalization
  Elementwise,  // Vector: add, mul, etc.
  Reduction,    // Vector: sum, max along dimension
  CVFusion,     // Fused: matmul + vector ops
  Unknown
};

//===----------------------------------------------------------------------===//
// Einsum-style Tensor Descriptor
//===----------------------------------------------------------------------===//

struct EinsumTensor {
  std::string name;
  std::vector<char> dims;           // Dimension labels (e.g., {'m', 'k'} for A in matmul)
  std::map<char, int64_t> shape;    // Full shape per dimension
  int64_t elementBytes = 2;         // FP16 default
  
  // Access pattern: how many times this tensor is accessed in full computation
  // For A[m,k] in matmul with output C[m,n]: A is accessed N times (once per output column)
  std::map<char, int64_t> reuseFactors;  // Reuse along each dimension
  
  int64_t getFullSize() const {
    int64_t size = elementBytes;
    for (char d : dims) {
      auto it = shape.find(d);
      if (it != shape.end()) size *= it->second;
    }
    return size;
  }
  
  int64_t getTileSize(const std::map<char, int64_t> &tileSizes) const {
    int64_t size = elementBytes;
    for (char d : dims) {
      auto it = tileSizes.find(d);
      if (it != tileSizes.end()) {
        size *= it->second;
      } else {
        // If dimension not tiled, use full size
        auto shapeIt = shape.find(d);
        if (shapeIt != shape.end()) size *= shapeIt->second;
      }
    }
    return size;
  }
  
  // Calculate number of accesses to backing store given tiling
  // Accesses = TileIterations * TileSize / ReuseWithinBuffer
  int64_t calculateAccesses(const std::map<char, int64_t> &tileSizes) const {
    int64_t tileSize = getTileSize(tileSizes);
    
    // Calculate number of tile iterations
    int64_t iterations = 1;
    for (auto &[dim, fullSize] : shape) {
      auto tileIt = tileSizes.find(dim);
      int64_t tileD = (tileIt != tileSizes.end()) ? tileIt->second : fullSize;
      iterations *= (fullSize + tileD - 1) / tileD;
    }
    
    // Factor in reuse: some dimensions don't cause re-access
    // For matmul A[m,k]: when iterating over N, A can be reused
    // The outer loops that don't touch A's dimensions allow reuse
    int64_t reuseIterations = 1;
    for (auto &[dim, reuseFactor] : reuseFactors) {
      // If this dimension is not in tensor's dims, it's a reuse opportunity
      if (std::find(dims.begin(), dims.end(), dim) == dims.end()) {
        auto tileIt = tileSizes.find(dim);
        auto shapeIt = shape.find(dim);
        if (tileIt != tileSizes.end() && shapeIt != shape.end()) {
          // Number of outer iterations over this dimension
          reuseIterations *= (shapeIt->second + tileIt->second - 1) / tileIt->second;
        }
      }
    }
    
    // Total accesses = iterations * tileSize, considering reuse pattern
    return iterations * tileSize;
  }
};

//===----------------------------------------------------------------------===//
// Operation Descriptor - describes a single tensor operation
//===----------------------------------------------------------------------===//

struct OperationDesc {
  std::string name;
  OpCategory category;
  
  std::vector<EinsumTensor> inputs;
  EinsumTensor output;
  
  // Dimensions of the operation
  std::set<char> allDims;           // All dimensions
  std::set<char> reductionDims;     // Contracted/reduced dimensions
  std::set<char> batchDims;         // Batch dimensions
  std::set<char> spatialDims;       // Output spatial dimensions
  
  // Compute characteristics
  int64_t flopsPerElement = 2;      // For matmul: 2 (mul + add)
  
  int64_t getTotalFlops() const {
    int64_t flops = flopsPerElement;
    for (char d : allDims) {
      // Find the size from any tensor that has this dim
      for (const auto &t : inputs) {
        auto it = t.shape.find(d);
        if (it != t.shape.end()) {
          flops *= it->second;
          break;
        }
      }
      auto outIt = output.shape.find(d);
      if (outIt != output.shape.end()) {
        flops *= outIt->second;
      }
    }
    return flops;
  }
  
  // Get problem size for a dimension
  int64_t getDimSize(char dim) const {
    for (const auto &t : inputs) {
      auto it = t.shape.find(dim);
      if (it != t.shape.end()) return it->second;
    }
    auto outIt = output.shape.find(dim);
    if (outIt != output.shape.end()) return outIt->second;
    return 1;
  }
};

//===----------------------------------------------------------------------===//
// Tiling Configuration
//===----------------------------------------------------------------------===//

struct TilingConfig {
  std::map<char, int64_t> tileSizes;  // Tile size per dimension
  
  // For CV fusion with independent tiling
  std::map<char, int64_t> cubeTileSizes;
  std::map<char, int64_t> vectorTileSizes;
  bool independentCV = false;
  
  int64_t getTile(char dim, int64_t defaultSize = 1) const {
    auto it = tileSizes.find(dim);
    return (it != tileSizes.end()) ? it->second : defaultSize;
  }
};

//===----------------------------------------------------------------------===//
// Tiling Result - memory-centric metrics
//===----------------------------------------------------------------------===//

struct TilingResult {
  TilingConfig config;
  
  // === Primary metrics (memory-centric optimization) ===
  int64_t totalMemoryAccesses = 0;    // Total bytes moved from/to backing store
  int64_t bufferSizeRequired = 0;     // On-chip buffer requirement
  
  // Per-tensor breakdown
  std::map<std::string, int64_t> tensorAccesses;
  std::map<std::string, int64_t> tensorBufferSizes;
  
  // === Secondary metrics ===
  int64_t totalCycles = 0;
  int64_t cubeCycles = 0;
  int64_t vectorCycles = 0;
  
  // === Derived metrics ===
  double operationalIntensity = 0.0;   // FLOPs / byte moved
  double memoryEfficiency = 0.0;       // Achieved OI / Algorithmic max OI
  double dataReuseRatio = 0.0;         // Total tensor size / Actual accesses
  
  bool valid = false;
  std::string invalidReason;
  
  // Pareto dominance for (buffer_size, accesses) frontier
  bool dominates(const TilingResult &other) const {
    bool betterAccesses = totalMemoryAccesses < other.totalMemoryAccesses;
    bool betterBuffer = bufferSizeRequired < other.bufferSizeRequired;
    bool sameOrBetterAccesses = totalMemoryAccesses <= other.totalMemoryAccesses;
    bool sameOrBetterBuffer = bufferSizeRequired <= other.bufferSizeRequired;
    
    return (betterAccesses && sameOrBetterBuffer) ||
           (betterBuffer && sameOrBetterAccesses);
  }
};

//===----------------------------------------------------------------------===//
// Memory Access Model (Orojenesis-inspired)
//===----------------------------------------------------------------------===//

class MemoryAccessModel {
public:
  MemoryAccessModel(const HardwareConfig &config) : hwConfig(config) {
    // Get memory hierarchy sizes
    l1Size = config.getMemorySizeBytes("l1");
    ubSize = config.getMemorySizeBytes("ub");
    l0aSize = config.getMemorySizeBytes("l0a");
    l0bSize = config.getMemorySizeBytes("l0b");
    l0cSize = config.getMemorySizeBytes("l0c");
    
    // Defaults if not specified
    if (l1Size == 0) l1Size = 1024 * 1024;    // 1MB
    if (ubSize == 0) ubSize = 256 * 1024;     // 256KB
    if (l0aSize == 0) l0aSize = 64 * 1024;    // 64KB
    if (l0bSize == 0) l0bSize = 64 * 1024;
    if (l0cSize == 0) l0cSize = 256 * 1024;
    
    hbmBandwidth = config.getHBMBandwidthGBs();
  }
  
  //=== Matmul: C[m,n] += A[m,k] * B[k,n] ===//
  
  TilingResult evaluateMatmul(int64_t M, int64_t N, int64_t K,
                               const TilingConfig &tiling,
                               int64_t elementBytes) {
    TilingResult result;
    result.config = tiling;
    
    int64_t tileM = tiling.getTile('m', M);
    int64_t tileN = tiling.getTile('n', N);
    int64_t tileK = tiling.getTile('k', K);
    
    // Validate buffer constraints
    int64_t bufA = tileM * tileK * elementBytes;  // A tile in L0A
    int64_t bufB = tileK * tileN * elementBytes;  // B tile in L0B
    int64_t bufC = tileM * tileN * elementBytes;  // C tile in L0C
    
    if (bufA > static_cast<int64_t>(l0aSize)) {
      result.valid = false;
      result.invalidReason = "A tile exceeds L0A";
      return result;
    }
    if (bufB > static_cast<int64_t>(l0bSize)) {
      result.valid = false;
      result.invalidReason = "B tile exceeds L0B";
      return result;
    }
    if (bufC > static_cast<int64_t>(l0cSize)) {
      result.valid = false;
      result.invalidReason = "C tile exceeds L0C";
      return result;
    }
    if (bufA + bufB > static_cast<int64_t>(l1Size)) {
      result.valid = false;
      result.invalidReason = "Input tiles exceed L1";
      return result;
    }
    
    result.valid = true;
    result.bufferSizeRequired = bufA + bufB + bufC;
    
    // Calculate memory accesses using Orojenesis model
    // 
    // For matmul with tiling (tileM, tileN, tileK):
    // - Number of output tiles: (M/tileM) * (N/tileN)
    // - Each output tile requires K/tileK iterations over K
    //
    // Access pattern:
    // for m1 in M/tileM:
    //   for n1 in N/tileN:
    //     for k1 in K/tileK:
    //       C[m1,n1] += A[m1,k1] * B[k1,n1]
    //
    // A[m,k] accesses: Each A tile is loaded N/tileN times (once per n1)
    //   Total A accesses = (M/tileM) * (N/tileN) * (K/tileK) * tileM * tileK * elemBytes
    //                    = M * K * (N/tileN) * elemBytes
    //
    // B[k,n] accesses: Each B tile is loaded M/tileM times (once per m1)
    //   Total B accesses = (M/tileM) * (N/tileN) * (K/tileK) * tileK * tileN * elemBytes
    //                    = K * N * (M/tileM) * elemBytes
    //
    // C[m,n] accesses: Write once per output tile, but read back for K accumulation
    //   Total C accesses = M * N * elemBytes (write only, assuming accumulator holds partials)
    
    int64_t tilesM = (M + tileM - 1) / tileM;
    int64_t tilesN = (N + tileN - 1) / tileN;
    int64_t tilesK = (K + tileK - 1) / tileK;
    
    // A is reused across N dimension
    int64_t accessA = M * K * elementBytes * tilesN;
    // Actually, with proper loop ordering, A can be reused:
    // If we loop m1 outermost, then k1, then n1: A[m1,k1] loaded once per (m1,k1) pair
    // Best case: A loaded once = M * K * elemBytes
    // Worst case: A loaded tilesN times
    
    // Using output-stationary dataflow (common for Ascend):
    // for m1: for n1: for k1: ... 
    // A is accessed tilesN times, B is accessed tilesM times
    
    accessA = M * K * elementBytes;  // Each A element loaded tilesN times, but A has M*K elements
    int64_t reuseA = tilesN;          // A is reused across N tiles
    accessA = (M * K * elementBytes) * reuseA / std::max(reuseA, 1L);
    
    // With double buffering and proper scheduling:
    // Minimum accesses = tensor size (each element loaded once)
    // Actual accesses depend on buffer capacity
    
    // Key insight from Orojenesis: buffer capacity determines achievable reuse
    // If buffer can hold all A tiles for a row, we get full reuse
    
    // Compute actual accesses with given tiling
    accessA = static_cast<int64_t>(M * K * elementBytes * 
              std::ceil(static_cast<double>(N) / tileN));
    int64_t accessB = static_cast<int64_t>(K * N * elementBytes * 
              std::ceil(static_cast<double>(M) / tileM));
    int64_t accessC = M * N * elementBytes;  // Write output once
    
    // If we use output-stationary with k-reduction in registers:
    // C is only written once, not read back
    
    result.totalMemoryAccesses = accessA + accessB + accessC;
    result.tensorAccesses["A"] = accessA;
    result.tensorAccesses["B"] = accessB;
    result.tensorAccesses["C"] = accessC;
    
    result.tensorBufferSizes["A"] = bufA;
    result.tensorBufferSizes["B"] = bufB;
    result.tensorBufferSizes["C"] = bufC;
    
    // Compute operational intensity
    int64_t flops = 2L * M * N * K;  // 2 ops per MAC
    result.operationalIntensity = static_cast<double>(flops) / result.totalMemoryAccesses;
    
    // Algorithmic minimum accesses (compulsory misses)
    int64_t algoMinAccesses = (M*K + K*N + M*N) * elementBytes;
    result.dataReuseRatio = static_cast<double>(algoMinAccesses) / result.totalMemoryAccesses;
    
    // Compute cycles (simplified)
    int fracM, fracK, fracN;
    hwConfig.getCubeFractalSize(elementBytes * 8, fracM, fracK, fracN);
    int64_t fractalOps = ((tileM + fracM - 1) / fracM) *
                         ((tileK + fracK - 1) / fracK) *
                         ((tileN + fracN - 1) / fracN);
    result.cubeCycles = fractalOps * tilesM * tilesN * tilesK;
    result.totalCycles = result.cubeCycles;
    
    return result;
  }
  
  //=== Vector operation: element-wise on tensor of shape [dims...] ===//
  
  TilingResult evaluateVector(const std::vector<int64_t> &shape,
                               const std::vector<int64_t> &tileShape,
                               int64_t elementBytes,
                               int64_t opsPerElement = 1) {
    TilingResult result;
    
    // Calculate total elements and tile elements
    int64_t totalElements = 1;
    int64_t tileElements = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      totalElements *= shape[i];
      tileElements *= (i < tileShape.size()) ? tileShape[i] : shape[i];
    }
    
    // Buffer requirement
    int64_t bufferReq = tileElements * elementBytes * 2;  // Input + output
    if (bufferReq > static_cast<int64_t>(ubSize)) {
      result.valid = false;
      result.invalidReason = "Tile exceeds UB size";
      return result;
    }
    
    result.valid = true;
    result.bufferSizeRequired = bufferReq;
    
    // Number of tiles
    int64_t numTiles = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      int64_t tileD = (i < tileShape.size()) ? tileShape[i] : shape[i];
      numTiles *= (shape[i] + tileD - 1) / tileD;
    }
    
    // Memory accesses: each element read once, written once
    // Assuming no data reuse opportunity for element-wise ops
    int64_t totalBytes = totalElements * elementBytes;
    result.totalMemoryAccesses = totalBytes * 2;  // Read + write
    
    result.tensorAccesses["input"] = totalBytes;
    result.tensorAccesses["output"] = totalBytes;
    
    // Compute
    int64_t vectorWidth = hwConfig.getVectorWidthElements();
    if (vectorWidth == 0) vectorWidth = 128;
    
    result.vectorCycles = (totalElements * opsPerElement + vectorWidth - 1) / vectorWidth;
    result.totalCycles = result.vectorCycles;
    
    // OI for vector ops is typically very low
    result.operationalIntensity = static_cast<double>(totalElements * opsPerElement) / 
                                   result.totalMemoryAccesses;
    
    return result;
  }
  
  //=== Softmax: row-wise softmax on [M, N] tensor ===//
  
  TilingResult evaluateSoftmax(int64_t M, int64_t N,
                                const TilingConfig &tiling,
                                int64_t elementBytes) {
    TilingResult result;
    result.config = tiling;
    
    int64_t tileM = tiling.getTile('m', M);
    // N must be processed fully for row-wise softmax (reduction)
    int64_t tileN = N;
    
    // Buffer: need full row for reduction
    int64_t bufferReq = tileM * N * elementBytes * 2;  // Input row + output row
    if (bufferReq > static_cast<int64_t>(ubSize)) {
      result.valid = false;
      result.invalidReason = "Softmax row exceeds UB";
      return result;
    }
    
    result.valid = true;
    result.bufferSizeRequired = bufferReq;
    
    // Memory accesses: each element read once, written once
    int64_t totalBytes = M * N * elementBytes;
    result.totalMemoryAccesses = totalBytes * 2;
    
    // Compute: ~5 ops per element (sub max, exp, sum, div)
    int64_t vectorWidth = hwConfig.getVectorWidthElements();
    if (vectorWidth == 0) vectorWidth = 128;
    
    result.vectorCycles = (M * N * 5 + vectorWidth - 1) / vectorWidth;
    result.totalCycles = result.vectorCycles;
    
    result.operationalIntensity = static_cast<double>(M * N * 5) / result.totalMemoryAccesses;
    
    return result;
  }
  
  //=== CV Fusion: Matmul followed by Vector ops ===//
  //
  // FlashAttention-style: QK = Q @ K^T, Softmax(QK), O = QK @ V
  // The intermediate QK tensor can stay on-chip if buffer is large enough
  //
  
  TilingResult evaluateCVFusion(
      const std::vector<OperationDesc> &ops,
      const TilingConfig &tiling,
      int64_t elementBytes) {
    TilingResult result;
    result.config = tiling;
    
    // For fusion, we need to analyze producer-consumer relationships
    // Key: intermediate tensors stay on-chip, only load inputs and store final output
    
    int64_t totalAccesses = 0;
    int64_t maxBufferReq = 0;
    
    for (size_t i = 0; i < ops.size(); ++i) {
      const auto &op = ops[i];
      
      if (op.category == OpCategory::Matmul || op.category == OpCategory::BMM) {
        // For matmul in fusion chain
        int64_t M = op.getDimSize('m');
        int64_t N = op.getDimSize('n');
        int64_t K = op.getDimSize('k');
        
        auto matmulResult = evaluateMatmul(M, N, K, tiling, elementBytes);
        if (!matmulResult.valid) {
          result.valid = false;
          result.invalidReason = "Matmul in fusion: " + matmulResult.invalidReason;
          return result;
        }
        
        // Only count external tensor accesses
        // First matmul: load A, B (inputs)
        // Intermediate: C stays on-chip for next op
        // Last matmul: store C (output)
        
        if (i == 0) {
          // First op: load inputs, intermediate stays on-chip
          totalAccesses += matmulResult.tensorAccesses["A"];
          totalAccesses += matmulResult.tensorAccesses["B"];
        } else if (i == ops.size() - 1) {
          // Last op: store output
          totalAccesses += matmulResult.tensorAccesses["C"];
          // Still need to load B (weights)
          totalAccesses += matmulResult.tensorAccesses["B"];
        } else {
          // Middle op: only load B (weights), intermediate on-chip
          totalAccesses += matmulResult.tensorAccesses["B"];
        }
        
        maxBufferReq = std::max(maxBufferReq, matmulResult.bufferSizeRequired);
        result.cubeCycles += matmulResult.cubeCycles;
        
      } else {
        // Vector op in fusion chain
        // Intermediate tensor already on-chip from producer
        // Only contribute to buffer requirement, not memory accesses
        
        int64_t M = op.getDimSize('m');
        int64_t N = op.getDimSize('n');
        
        if (op.category == OpCategory::Softmax) {
          auto softmaxResult = evaluateSoftmax(M, N, tiling, elementBytes);
          if (!softmaxResult.valid) {
            result.valid = false;
            result.invalidReason = "Softmax in fusion: " + softmaxResult.invalidReason;
            return result;
          }
          // Softmax operates on intermediate - no external memory access
          maxBufferReq = std::max(maxBufferReq, softmaxResult.bufferSizeRequired);
          result.vectorCycles += softmaxResult.vectorCycles;
        }
      }
    }
    
    result.valid = true;
    result.totalMemoryAccesses = totalAccesses;
    result.bufferSizeRequired = maxBufferReq;
    result.totalCycles = std::max(result.cubeCycles, result.vectorCycles);
    
    return result;
  }

private:
  const HardwareConfig &hwConfig;
  size_t l1Size, ubSize, l0aSize, l0bSize, l0cSize;
  double hbmBandwidth;
};

//===----------------------------------------------------------------------===//
// Pareto Frontier Builder
//===----------------------------------------------------------------------===//

class ParetoFrontier {
public:
  void addPoint(const TilingResult &result) {
    if (!result.valid) return;
    
    // Check if this point is dominated by any existing point
    for (const auto &existing : points) {
      if (existing.dominates(result)) {
        return;  // Dominated, don't add
      }
    }
    
    // Remove any points dominated by the new point
    points.erase(
      std::remove_if(points.begin(), points.end(),
        [&result](const TilingResult &p) { return result.dominates(p); }),
      points.end());
    
    points.push_back(result);
  }
  
  const std::vector<TilingResult>& getPoints() const { return points; }
  
  // Get the best tiling for a given buffer constraint
  TilingResult getBestForBuffer(int64_t maxBuffer) const {
    TilingResult best;
    best.totalMemoryAccesses = INT64_MAX;
    
    for (const auto &p : points) {
      if (p.bufferSizeRequired <= maxBuffer &&
          p.totalMemoryAccesses < best.totalMemoryAccesses) {
        best = p;
      }
    }
    return best;
  }
  
  void print(llvm::raw_ostream &os) const {
    os << "\n===== Pareto Frontier (Buffer Size vs Memory Accesses) =====\n";
    os << llvm::format("%12s %15s %12s %8s\n", 
                       "Buffer(KB)", "Accesses(MB)", "OI", "Reuse");
    os << "-------------------------------------------------------------\n";
    
    // Sort by buffer size
    auto sorted = points;
    std::sort(sorted.begin(), sorted.end(),
      [](const TilingResult &a, const TilingResult &b) {
        return a.bufferSizeRequired < b.bufferSizeRequired;
      });
    
    for (const auto &p : sorted) {
      os << llvm::format("%12.1f %15.2f %12.1f %8.2f\n",
                         p.bufferSizeRequired / 1024.0,
                         p.totalMemoryAccesses / (1024.0 * 1024.0),
                         p.operationalIntensity,
                         p.dataReuseRatio);
    }
  }

private:
  std::vector<TilingResult> points;
};

//===----------------------------------------------------------------------===//
// Tiling Search (Exhaustive for Pareto frontier)
//===----------------------------------------------------------------------===//

class TilingSearch {
public:
  TilingSearch(MemoryAccessModel &model) : accessModel(model) {}
  
  // Search for Matmul
  ParetoFrontier searchMatmul(int64_t M, int64_t N, int64_t K,
                               int64_t elementBytes,
                               int64_t minTile, int64_t maxTile, int64_t step) {
    ParetoFrontier frontier;
    
    // Exhaustive search over tiling space
    for (int64_t tileM = minTile; tileM <= std::min(M, maxTile); tileM += step) {
      for (int64_t tileN = minTile; tileN <= std::min(N, maxTile); tileN += step) {
        for (int64_t tileK = minTile; tileK <= std::min(K, maxTile); tileK += step) {
          TilingConfig config;
          config.tileSizes['m'] = tileM;
          config.tileSizes['n'] = tileN;
          config.tileSizes['k'] = tileK;
          
          auto result = accessModel.evaluateMatmul(M, N, K, config, elementBytes);
          frontier.addPoint(result);
        }
      }
    }
    
    return frontier;
  }
  
  // Search for CV Fusion
  ParetoFrontier searchCVFusion(const std::vector<OperationDesc> &ops,
                                 int64_t elementBytes,
                                 int64_t minTile, int64_t maxTile, int64_t step) {
    ParetoFrontier frontier;
    
    // Collect all dimensions
    std::set<char> allDims;
    std::map<char, int64_t> maxSizes;
    for (const auto &op : ops) {
      for (char d : op.allDims) {
        allDims.insert(d);
        maxSizes[d] = std::max(maxSizes[d], op.getDimSize(d));
      }
    }
    
    // Generate tiling configurations (simplified: just M, N, K)
    int64_t M = maxSizes.count('m') ? maxSizes['m'] : 1;
    int64_t N = maxSizes.count('n') ? maxSizes['n'] : 1;
    int64_t K = maxSizes.count('k') ? maxSizes['k'] : 1;
    
    for (int64_t tileM = minTile; tileM <= std::min(M, maxTile); tileM += step) {
      for (int64_t tileN = minTile; tileN <= std::min(N, maxTile); tileN += step) {
        for (int64_t tileK = minTile; tileK <= std::min(K, maxTile); tileK += step) {
          TilingConfig config;
          config.tileSizes['m'] = tileM;
          config.tileSizes['n'] = tileN;
          config.tileSizes['k'] = tileK;
          
          auto result = accessModel.evaluateCVFusion(ops, config, elementBytes);
          frontier.addPoint(result);
        }
      }
    }
    
    return frontier;
  }

private:
  MemoryAccessModel &accessModel;
};

//===----------------------------------------------------------------------===//
// Main API
//===----------------------------------------------------------------------===//

/// Optimize tiling for minimum memory accesses given buffer constraint
TilingResult optimizeTilingForMemory(
    const HardwareConfig &config,
    OpCategory opType,
    int64_t M, int64_t N, int64_t K,
    int64_t elementBytes,
    int64_t bufferBudget,
    bool verbose = false) {
  
  MemoryAccessModel model(config);
  TilingSearch search(model);
  
  // Get fractal alignment
  int fracM, fracK, fracN;
  config.getCubeFractalSize(elementBytes * 8, fracM, fracK, fracN);
  
  int64_t minTile = std::max({fracM, fracK, fracN});
  int64_t maxTile = 256;
  int64_t step = minTile;
  
  ParetoFrontier frontier;
  
  if (opType == OpCategory::Matmul) {
    frontier = search.searchMatmul(M, N, K, elementBytes, minTile, maxTile, step);
  } else {
    // TODO: Handle other op types
    TilingResult empty;
    empty.valid = false;
    empty.invalidReason = "Unsupported operation type";
    return empty;
  }
  
  if (verbose) {
    frontier.print(llvm::outs());
  }
  
  // Get best configuration within budget
  return frontier.getBestForBuffer(bufferBudget);
}

/// Get full Pareto frontier for analysis
ParetoFrontier getMemoryParetoFrontier(
    const HardwareConfig &config,
    OpCategory opType,
    int64_t M, int64_t N, int64_t K,
    int64_t elementBytes) {
  
  MemoryAccessModel model(config);
  TilingSearch search(model);
  
  int fracM, fracK, fracN;
  config.getCubeFractalSize(elementBytes * 8, fracM, fracK, fracN);
  
  int64_t minTile = std::max({fracM, fracK, fracN});
  int64_t maxTile = 256;
  int64_t step = minTile;
  
  return search.searchMatmul(M, N, K, elementBytes, minTile, maxTile, step);
}

} // namespace
} // namespace ascend
} // namespace mlir
