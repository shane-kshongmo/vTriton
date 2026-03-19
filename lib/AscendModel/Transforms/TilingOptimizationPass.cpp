//===- TilingOptimizationPass.cpp - Tiling parameter optimization ---------===//
//
// This pass searches for optimal tiling parameters to maximize pipeline
// parallelism between Cube and Vector paths on Ascend hardware.
//
// The key insight is that different tiling configurations affect:
// 1. The ratio of Cube vs Vector work per iteration
// 2. Memory transfer sizes and opportunities for hiding latency
// 3. Buffer utilization (L1, UB, L0A/B/C constraints)
//
//===----------------------------------------------------------------------===//

#include "AscendModel/IR/AscendModelDialect.h"
#include "AscendModel/IR/AscendModelInterfaces.h"
#include "AscendModel/Transforms/Passes.h"
#include "AscendModel/HardwareConfig.h"
#include "AscendModel/Utils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

#include <algorithm>
#include <random>
#include <vector>

namespace mlir {
namespace ascend {

#define GEN_PASS_DEF_TILINGOPTIMIZATIONPASS
#include "AscendModel/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Tiling Configuration
//===----------------------------------------------------------------------===//

/// Tiling configuration for Cube path
struct CubeTilingConfig {
  int64_t tileM = 128;
  int64_t tileN = 128;
  int64_t tileK = 64;
  
  bool operator==(const CubeTilingConfig &other) const {
    return tileM == other.tileM && tileN == other.tileN && tileK == other.tileK;
  }
};

/// Tiling configuration for Vector path
struct VectorTilingConfig {
  int64_t tileM = 128;  // Vector processes output in tiles
  int64_t tileN = 128;
  
  bool operator==(const VectorTilingConfig &other) const {
    return tileM == other.tileM && tileN == other.tileN;
  }
};

/// Combined tiling configuration for CV fusion
struct TilingConfig {
  CubeTilingConfig cube;
  VectorTilingConfig vector;
  bool independentCV = false;  // Whether Cube and Vector use different tiling
  
  // Legacy constructor for unified tiling
  TilingConfig() = default;
  TilingConfig(int64_t m, int64_t n, int64_t k) 
      : cube{m, n, k}, vector{m, n}, independentCV(false) {}
  
  // Constructor for independent CV tiling
  TilingConfig(const CubeTilingConfig &c, const VectorTilingConfig &v)
      : cube(c), vector(v), independentCV(true) {}
  
  bool operator==(const TilingConfig &other) const {
    return cube == other.cube && vector == other.vector && 
           independentCV == other.independentCV;
  }
};

struct TilingResult {
  TilingConfig config;
  int64_t totalCycles = 0;
  int64_t cubeCycles = 0;
  int64_t vectorCycles = 0;
  int64_t memoryCycles = 0;
  double pipelineEfficiency = 0.0;  // 1.0 = perfect overlap
  bool valid = false;
  std::string invalidReason;
  
  // Detailed breakdown for analysis
  int64_t numCubeTiles = 0;
  int64_t numVectorTiles = 0;
  int64_t cubeComputePerTile = 0;
  int64_t vectorComputePerTile = 0;
  double overlapRatio = 0.0;
  
  // CV balance metric (closer to 1.0 = better balance)
  double cvBalanceRatio = 0.0;
};

//===----------------------------------------------------------------------===//
// Search Range
//===----------------------------------------------------------------------===//

struct SearchRange {
  int64_t min = 64;
  int64_t max = 256;
  int64_t step = 64;
  
  std::vector<int64_t> enumerate() const {
    std::vector<int64_t> values;
    for (int64_t v = min; v <= max; v += step) {
      values.push_back(v);
    }
    return values;
  }
  
  static bool parse(llvm::StringRef str, SearchRange &range) {
    if (str.empty()) return false;
    
    llvm::SmallVector<llvm::StringRef, 3> parts;
    str.split(parts, ',');
    
    if (parts.size() != 3) return false;
    
    if (parts[0].getAsInteger(10, range.min)) return false;
    if (parts[1].getAsInteger(10, range.max)) return false;
    if (parts[2].getAsInteger(10, range.step)) return false;
    
    return range.min > 0 && range.max >= range.min && range.step > 0;
  }
};

//===----------------------------------------------------------------------===//
// Tiling Evaluator - Uses existing model at tile granularity
//===----------------------------------------------------------------------===//

class TilingEvaluator {
public:
  TilingEvaluator(const HardwareConfig &config, 
                  int64_t problemM, int64_t problemN, int64_t problemK,
                  int elementBits)
      : config(config), problemM(problemM), problemN(problemN), 
        problemK(problemK), elementBits(elementBits) {
    // Get fractal dimensions
    config.getCubeFractalSize(elementBits, fracM, fracK, fracN);
    
    // Get memory constraints
    l1Size = config.getMemorySizeBytes("l1");
    ubSize = config.getMemorySizeBytes("ub");
    l0aSize = config.getMemorySizeBytes("l0a");
    l0bSize = config.getMemorySizeBytes("l0b");
    l0cSize = config.getMemorySizeBytes("l0c");
    
    if (l1Size == 0) l1Size = 1024 * 1024;      // 1MB default
    if (ubSize == 0) ubSize = 256 * 1024;       // 256KB default
    if (l0aSize == 0) l0aSize = 64 * 1024;      // 64KB default
    if (l0bSize == 0) l0bSize = 64 * 1024;
    if (l0cSize == 0) l0cSize = 256 * 1024;
    
    bytesPerElement = elementBits / 8;
    
    // Get bandwidth (bytes per cycle)
    hbmBwBytesPerCycle = config.getHBMBandwidthGBs() * 1e9 / 
                         (config.getClockFrequencyGHz() * 1e9);
    
    vectorWidth = config.getVectorWidthElements();
    if (vectorWidth == 0) vectorWidth = 128;
  }
  
  TilingResult evaluate(const TilingConfig &tiling) {
    TilingResult result;
    result.config = tiling;
    
    // Check validity
    if (!checkValidity(tiling, result.invalidReason)) {
      result.valid = false;
      return result;
    }
    result.valid = true;
    
    //=== Cube path analysis ===//
    const auto &cubeTile = tiling.cube;
    
    int64_t numCubeTilesM = (problemM + cubeTile.tileM - 1) / cubeTile.tileM;
    int64_t numCubeTilesN = (problemN + cubeTile.tileN - 1) / cubeTile.tileN;
    int64_t numCubeTilesK = (problemK + cubeTile.tileK - 1) / cubeTile.tileK;
    int64_t numCubeOutputTiles = numCubeTilesM * numCubeTilesN;
    
    // Cube compute cycles (fractal-based)
    int64_t fracTilesM = (cubeTile.tileM + fracM - 1) / fracM;
    int64_t fracTilesK = (cubeTile.tileK + fracK - 1) / fracK;
    int64_t fracTilesN = (cubeTile.tileN + fracN - 1) / fracN;
    int64_t cubeComputePerTile = fracTilesM * fracTilesK * fracTilesN;
    
    // Cube memory transfer cycles
    int64_t cubeLoadBytesA = cubeTile.tileM * cubeTile.tileK * bytesPerElement;
    int64_t cubeLoadBytesB = cubeTile.tileK * cubeTile.tileN * bytesPerElement;
    int64_t cubeStoreBytesC = cubeTile.tileM * cubeTile.tileN * bytesPerElement;
    
    int64_t cubeLoadPerTile = static_cast<int64_t>(
        (cubeLoadBytesA + cubeLoadBytesB) / hbmBwBytesPerCycle);
    int64_t cubeStorePerTile = static_cast<int64_t>(
        cubeStoreBytesC / hbmBwBytesPerCycle);
    
    // Cube path per-tile (intra-path overlap)
    int64_t cubePathPerTile = std::max({
        cubeComputePerTile,
        cubeLoadPerTile,
        cubeStorePerTile
    });
    
    // Total Cube path cycles
    int64_t totalCubeCompute = cubeComputePerTile * numCubeTilesK * numCubeOutputTiles;
    int64_t totalCubeLoad = cubeLoadPerTile * numCubeTilesK * numCubeOutputTiles;
    int64_t totalCubeStore = cubeStorePerTile * numCubeOutputTiles;
    int64_t cubePathTotal = std::max({totalCubeCompute, totalCubeLoad, totalCubeStore});
    
    //=== Vector path analysis ===//
    const auto &vectorTile = tiling.vector;
    
    int64_t numVectorTilesM = (problemM + vectorTile.tileM - 1) / vectorTile.tileM;
    int64_t numVectorTilesN = (problemN + vectorTile.tileN - 1) / vectorTile.tileN;
    int64_t numVectorTiles = numVectorTilesM * numVectorTilesN;
    
    // Vector compute cycles
    int64_t vectorOpsPerTile = vectorTile.tileM * vectorTile.tileN;
    int64_t vectorComputePerTile = (vectorOpsPerTile + vectorWidth - 1) / vectorWidth;
    
    // Vector memory transfer cycles
    int64_t vectorBytesPerTile = vectorTile.tileM * vectorTile.tileN * bytesPerElement;
    int64_t vectorLoadPerTile = static_cast<int64_t>(vectorBytesPerTile / hbmBwBytesPerCycle);
    int64_t vectorStorePerTile = static_cast<int64_t>(vectorBytesPerTile / hbmBwBytesPerCycle);
    
    // Vector path per-tile (intra-path overlap)
    int64_t vectorPathPerTile = std::max({
        vectorComputePerTile,
        vectorLoadPerTile,
        vectorStorePerTile
    });
    
    // Total Vector path cycles
    int64_t totalVectorCompute = vectorComputePerTile * numVectorTiles;
    int64_t totalVectorLoad = vectorLoadPerTile * numVectorTiles;
    int64_t totalVectorStore = vectorStorePerTile * numVectorTiles;
    int64_t vectorPathTotal = std::max({totalVectorCompute, totalVectorLoad, totalVectorStore});
    
    //=== CV Pipeline overlap analysis ===//
    //
    // For CV fusion with independent tiling:
    // - Cube produces output tiles at rate: cubePathPerTile cycles/tile
    // - Vector consumes tiles at rate: vectorPathPerTile cycles/tile
    //
    // Optimal overlap when production rate matches consumption rate.
    // If Cube tile != Vector tile, we need to consider tile mapping.
    //
    // Key insight for independent tiling:
    // - Multiple small Vector tiles can process one large Cube tile
    // - Or one large Vector tile can wait for multiple Cube tiles
    //
    
    // Calculate effective rates (cycles per output element)
    double cubeRatePerElement = static_cast<double>(cubePathTotal) / (problemM * problemN);
    double vectorRatePerElement = static_cast<double>(vectorPathTotal) / (problemM * problemN);
    
    // CV balance: ratio of slower to faster path (1.0 = perfectly balanced)
    double cvBalanceRatio = std::min(cubeRatePerElement, vectorRatePerElement) /
                            std::max(cubeRatePerElement, vectorRatePerElement);
    
    // Calculate overlap based on:
    // 1. Number of pipeline stages (more tiles = more overlap opportunity)
    // 2. CV balance (better balance = more effective overlap)
    
    int64_t longerPath = std::max(cubePathTotal, vectorPathTotal);
    int64_t shorterPath = std::min(cubePathTotal, vectorPathTotal);
    
    // Effective number of pipeline stages
    // With independent tiling, use the larger tile count for overlap calculation
    int64_t effectiveTiles = std::max(numCubeOutputTiles, numVectorTiles);
    
    double overlapRatio = 0.0;
    if (effectiveTiles > 1) {
      // Base overlap potential from pipelining
      double pipelineOverlap = static_cast<double>(effectiveTiles - 1) / effectiveTiles;
      
      // Modulate by CV balance - better balance enables more overlap
      overlapRatio = pipelineOverlap * cvBalanceRatio;
      
      // Bonus for independent tiling that improves balance
      if (tiling.independentCV && cvBalanceRatio > 0.7) {
        // Good CV balance with independent tiling - can achieve better overlap
        overlapRatio = std::min(overlapRatio * 1.1, pipelineOverlap);
      }
    }
    
    // Calculate final cycles
    int64_t serialCycles = cubePathTotal + vectorPathTotal;
    int64_t overlappedCycles = longerPath + 
        static_cast<int64_t>((1.0 - overlapRatio) * shorterPath);
    
    // Add startup latencies
    overlappedCycles += config.getCubeStartupLatency();
    overlappedCycles += config.getMTE2StartupLatency();
    overlappedCycles += config.getVectorStartupLatency();
    
    // Fill result
    result.totalCycles = overlappedCycles;
    result.cubeCycles = cubePathTotal;
    result.vectorCycles = vectorPathTotal;
    result.memoryCycles = totalCubeLoad + totalCubeStore + totalVectorLoad + totalVectorStore;
    result.pipelineEfficiency = static_cast<double>(serialCycles) / 
                                 (2.0 * std::max(result.totalCycles, 1L));
    
    result.numCubeTiles = numCubeOutputTiles;
    result.numVectorTiles = numVectorTiles;
    result.cubeComputePerTile = cubeComputePerTile;
    result.vectorComputePerTile = vectorComputePerTile;
    result.overlapRatio = overlapRatio;
    result.cvBalanceRatio = cvBalanceRatio;
    
    return result;
  }
  
private:
  bool checkValidity(const TilingConfig &tiling, std::string &reason) {
    const auto &cube = tiling.cube;
    const auto &vector = tiling.vector;
    
    // Check Cube fractal alignment
    if (cube.tileM % fracM != 0) {
      reason = "Cube tileM must be multiple of fractal M (" + std::to_string(fracM) + ")";
      return false;
    }
    if (cube.tileN % fracN != 0) {
      reason = "Cube tileN must be multiple of fractal N (" + std::to_string(fracN) + ")";
      return false;
    }
    if (cube.tileK % fracK != 0) {
      reason = "Cube tileK must be multiple of fractal K (" + std::to_string(fracK) + ")";
      return false;
    }
    
    // Check Cube memory constraints
    int64_t l0aNeeded = cube.tileM * cube.tileK * bytesPerElement;
    int64_t l0bNeeded = cube.tileK * cube.tileN * bytesPerElement;
    if (l0aNeeded > static_cast<int64_t>(l0aSize)) {
      reason = "Cube A tile (" + std::to_string(l0aNeeded) + 
               " bytes) exceeds L0A size (" + std::to_string(l0aSize) + ")";
      return false;
    }
    if (l0bNeeded > static_cast<int64_t>(l0bSize)) {
      reason = "Cube B tile (" + std::to_string(l0bNeeded) + 
               " bytes) exceeds L0B size (" + std::to_string(l0bSize) + ")";
      return false;
    }
    
    int64_t l0cNeeded = cube.tileM * cube.tileN * bytesPerElement;
    if (l0cNeeded > static_cast<int64_t>(l0cSize)) {
      reason = "Cube output tile (" + std::to_string(l0cNeeded) + 
               " bytes) exceeds L0C size (" + std::to_string(l0cSize) + ")";
      return false;
    }
    
    int64_t l1Needed = l0aNeeded + l0bNeeded;
    if (l1Needed > static_cast<int64_t>(l1Size)) {
      reason = "Cube input tiles (" + std::to_string(l1Needed) + 
               " bytes) exceed L1 size (" + std::to_string(l1Size) + ")";
      return false;
    }
    
    // Check Vector memory constraints (UB)
    int64_t ubNeeded = vector.tileM * vector.tileN * bytesPerElement;
    // Vector typically needs input + output in UB
    if (ubNeeded * 2 > static_cast<int64_t>(ubSize)) {
      reason = "Vector tiles (" + std::to_string(ubNeeded * 2) + 
               " bytes) exceed UB size (" + std::to_string(ubSize) + ")";
      return false;
    }
    
    // For independent tiling, Vector tile should divide Cube output evenly
    // (or Cube output should divide Vector tile evenly)
    if (tiling.independentCV) {
      // Check alignment between Cube output and Vector input
      // Vector processes Cube's output, so dimensions should be compatible
      if (cube.tileM % vector.tileM != 0 && vector.tileM % cube.tileM != 0) {
        reason = "Vector tileM (" + std::to_string(vector.tileM) + 
                 ") and Cube tileM (" + std::to_string(cube.tileM) + 
                 ") are not divisible";
        return false;
      }
      if (cube.tileN % vector.tileN != 0 && vector.tileN % cube.tileN != 0) {
        reason = "Vector tileN (" + std::to_string(vector.tileN) + 
                 ") and Cube tileN (" + std::to_string(cube.tileN) + 
                 ") are not divisible";
        return false;
      }
    }
    
    return true;
  }
  
  const HardwareConfig &config;
  int64_t problemM, problemN, problemK;
  int elementBits;
  int fracM, fracK, fracN;
  size_t l1Size, ubSize, l0aSize, l0bSize, l0cSize;
  int bytesPerElement;
  double hbmBwBytesPerCycle;
  int64_t vectorWidth;
};

//===----------------------------------------------------------------------===//
// Search Strategies
//===----------------------------------------------------------------------===//

/// Search range for Vector tiling (separate from Cube)
struct VectorSearchRange {
  SearchRange m;
  SearchRange n;
};

class TilingSearchStrategy {
public:
  virtual ~TilingSearchStrategy() = default;
  
  /// Search for best unified tiling (Cube and Vector use same tile sizes)
  virtual TilingResult searchUnified(TilingEvaluator &evaluator,
                                     const SearchRange &mRange,
                                     const SearchRange &nRange,
                                     const SearchRange &kRange,
                                     bool verbose) = 0;
  
  /// Search for best independent CV tiling
  virtual TilingResult searchIndependent(TilingEvaluator &evaluator,
                                         const SearchRange &cubeMRange,
                                         const SearchRange &cubeNRange,
                                         const SearchRange &cubeKRange,
                                         const VectorSearchRange &vectorRange,
                                         bool verbose) {
    // Default: fall back to unified search
    return searchUnified(evaluator, cubeMRange, cubeNRange, cubeKRange, verbose);
  }
};

/// Exhaustive search: try all combinations
class ExhaustiveSearch : public TilingSearchStrategy {
public:
  TilingResult searchUnified(TilingEvaluator &evaluator,
                             const SearchRange &mRange,
                             const SearchRange &nRange,
                             const SearchRange &kRange,
                             bool verbose) override {
    TilingResult best;
    best.totalCycles = INT64_MAX;
    
    auto mValues = mRange.enumerate();
    auto nValues = nRange.enumerate();
    auto kValues = kRange.enumerate();
    
    int64_t totalConfigs = mValues.size() * nValues.size() * kValues.size();
    int64_t evaluated = 0;
    
    if (verbose) {
      llvm::outs() << "Exhaustive search (unified): " << totalConfigs << " configurations\n";
    }
    
    for (int64_t m : mValues) {
      for (int64_t n : nValues) {
        for (int64_t k : kValues) {
          TilingConfig config(m, n, k);
          TilingResult result = evaluator.evaluate(config);
          evaluated++;
          
          if (result.valid && result.totalCycles < best.totalCycles) {
            best = result;
            if (verbose) {
              llvm::outs() << "  New best: M=" << m << " N=" << n << " K=" << k
                           << " cycles=" << result.totalCycles << "\n";
            }
          }
        }
      }
    }
    
    if (verbose) {
      llvm::outs() << "Evaluated " << evaluated << " configurations\n";
    }
    
    return best;
  }
  
  TilingResult searchIndependent(TilingEvaluator &evaluator,
                                 const SearchRange &cubeMRange,
                                 const SearchRange &cubeNRange,
                                 const SearchRange &cubeKRange,
                                 const VectorSearchRange &vectorRange,
                                 bool verbose) override {
    TilingResult best;
    best.totalCycles = INT64_MAX;
    
    auto cubeMValues = cubeMRange.enumerate();
    auto cubeNValues = cubeNRange.enumerate();
    auto cubeKValues = cubeKRange.enumerate();
    auto vecMValues = vectorRange.m.enumerate();
    auto vecNValues = vectorRange.n.enumerate();
    
    int64_t totalConfigs = cubeMValues.size() * cubeNValues.size() * cubeKValues.size() *
                          vecMValues.size() * vecNValues.size();
    int64_t evaluated = 0;
    
    if (verbose) {
      llvm::outs() << "Exhaustive search (independent CV): " << totalConfigs << " configurations\n";
    }
    
    for (int64_t cm : cubeMValues) {
      for (int64_t cn : cubeNValues) {
        for (int64_t ck : cubeKValues) {
          for (int64_t vm : vecMValues) {
            for (int64_t vn : vecNValues) {
              CubeTilingConfig cubeConfig{cm, cn, ck};
              VectorTilingConfig vecConfig{vm, vn};
              TilingConfig config(cubeConfig, vecConfig);
              
              TilingResult result = evaluator.evaluate(config);
              evaluated++;
              
              if (result.valid && result.totalCycles < best.totalCycles) {
                best = result;
                if (verbose) {
                  llvm::outs() << "  New best: Cube[" << cm << "," << cn << "," << ck 
                               << "] Vec[" << vm << "," << vn << "]"
                               << " cycles=" << result.totalCycles 
                               << " balance=" << llvm::format("%.2f", result.cvBalanceRatio)
                               << "\n";
                }
              }
            }
          }
        }
      }
    }
    
    if (verbose) {
      llvm::outs() << "Evaluated " << evaluated << " configurations\n";
    }
    
    return best;
  }
};

/// Heuristic search: use greedy approach with pruning
class HeuristicSearch : public TilingSearchStrategy {
public:
  TilingResult searchUnified(TilingEvaluator &evaluator,
                             const SearchRange &mRange,
                             const SearchRange &nRange,
                             const SearchRange &kRange,
                             bool verbose) override {
    TilingResult best;
    best.totalCycles = INT64_MAX;
    
    // Start from middle of range
    int64_t startM = ((mRange.min + mRange.max) / 2 / mRange.step) * mRange.step;
    int64_t startN = ((nRange.min + nRange.max) / 2 / nRange.step) * nRange.step;
    int64_t startK = ((kRange.min + kRange.max) / 2 / kRange.step) * kRange.step;
    
    TilingConfig current(startM, startN, startK);
    
    if (verbose) {
      llvm::outs() << "Heuristic search (unified) starting from M=" << startM
                   << " N=" << startN << " K=" << startK << "\n";
    }
    
    // Hill climbing
    bool improved = true;
    int iterations = 0;
    while (improved && iterations < 100) {
      improved = false;
      iterations++;
      
      TilingResult currentResult = evaluator.evaluate(current);
      if (currentResult.valid && currentResult.totalCycles < best.totalCycles) {
        best = currentResult;
      }
      
      // Try neighbors (unified: same tile for Cube and Vector)
      std::vector<TilingConfig> neighbors;
      auto &c = current.cube;
      
      if (c.tileM - mRange.step >= mRange.min)
        neighbors.push_back(TilingConfig(c.tileM - mRange.step, c.tileN, c.tileK));
      if (c.tileM + mRange.step <= mRange.max)
        neighbors.push_back(TilingConfig(c.tileM + mRange.step, c.tileN, c.tileK));
      if (c.tileN - nRange.step >= nRange.min)
        neighbors.push_back(TilingConfig(c.tileM, c.tileN - nRange.step, c.tileK));
      if (c.tileN + nRange.step <= nRange.max)
        neighbors.push_back(TilingConfig(c.tileM, c.tileN + nRange.step, c.tileK));
      if (c.tileK - kRange.step >= kRange.min)
        neighbors.push_back(TilingConfig(c.tileM, c.tileN, c.tileK - kRange.step));
      if (c.tileK + kRange.step <= kRange.max)
        neighbors.push_back(TilingConfig(c.tileM, c.tileN, c.tileK + kRange.step));
      
      for (const auto &neighbor : neighbors) {
        TilingResult result = evaluator.evaluate(neighbor);
        if (result.valid && result.totalCycles < best.totalCycles) {
          best = result;
          current = neighbor;
          improved = true;
          
          if (verbose) {
            llvm::outs() << "  Improved: M=" << neighbor.cube.tileM 
                         << " N=" << neighbor.cube.tileN 
                         << " K=" << neighbor.cube.tileK
                         << " cycles=" << result.totalCycles << "\n";
          }
        }
      }
    }
    
    if (verbose) {
      llvm::outs() << "Heuristic search completed in " << iterations << " iterations\n";
    }
    
    return best;
  }
  
  TilingResult searchIndependent(TilingEvaluator &evaluator,
                                 const SearchRange &cubeMRange,
                                 const SearchRange &cubeNRange,
                                 const SearchRange &cubeKRange,
                                 const VectorSearchRange &vectorRange,
                                 bool verbose) override {
    TilingResult best;
    best.totalCycles = INT64_MAX;
    
    // Start from middle of ranges
    CubeTilingConfig cubeStart{
      ((cubeMRange.min + cubeMRange.max) / 2 / cubeMRange.step) * cubeMRange.step,
      ((cubeNRange.min + cubeNRange.max) / 2 / cubeNRange.step) * cubeNRange.step,
      ((cubeKRange.min + cubeKRange.max) / 2 / cubeKRange.step) * cubeKRange.step
    };
    VectorTilingConfig vecStart{
      ((vectorRange.m.min + vectorRange.m.max) / 2 / vectorRange.m.step) * vectorRange.m.step,
      ((vectorRange.n.min + vectorRange.n.max) / 2 / vectorRange.n.step) * vectorRange.n.step
    };
    
    TilingConfig current(cubeStart, vecStart);
    
    if (verbose) {
      llvm::outs() << "Heuristic search (independent CV) starting from:\n"
                   << "  Cube: M=" << cubeStart.tileM << " N=" << cubeStart.tileN 
                   << " K=" << cubeStart.tileK << "\n"
                   << "  Vector: M=" << vecStart.tileM << " N=" << vecStart.tileN << "\n";
    }
    
    // Hill climbing with alternating Cube/Vector optimization
    bool improved = true;
    int iterations = 0;
    while (improved && iterations < 200) {
      improved = false;
      iterations++;
      
      TilingResult currentResult = evaluator.evaluate(current);
      if (currentResult.valid && currentResult.totalCycles < best.totalCycles) {
        best = currentResult;
      }
      
      // Alternate between optimizing Cube and Vector tiling
      bool optimizeCube = (iterations % 2 == 1);
      
      std::vector<TilingConfig> neighbors;
      auto &c = current.cube;
      auto &v = current.vector;
      
      if (optimizeCube) {
        // Try Cube neighbors
        if (c.tileM - cubeMRange.step >= cubeMRange.min)
          neighbors.push_back(TilingConfig(
            CubeTilingConfig{c.tileM - cubeMRange.step, c.tileN, c.tileK}, v));
        if (c.tileM + cubeMRange.step <= cubeMRange.max)
          neighbors.push_back(TilingConfig(
            CubeTilingConfig{c.tileM + cubeMRange.step, c.tileN, c.tileK}, v));
        if (c.tileN - cubeNRange.step >= cubeNRange.min)
          neighbors.push_back(TilingConfig(
            CubeTilingConfig{c.tileM, c.tileN - cubeNRange.step, c.tileK}, v));
        if (c.tileN + cubeNRange.step <= cubeNRange.max)
          neighbors.push_back(TilingConfig(
            CubeTilingConfig{c.tileM, c.tileN + cubeNRange.step, c.tileK}, v));
        if (c.tileK - cubeKRange.step >= cubeKRange.min)
          neighbors.push_back(TilingConfig(
            CubeTilingConfig{c.tileM, c.tileN, c.tileK - cubeKRange.step}, v));
        if (c.tileK + cubeKRange.step <= cubeKRange.max)
          neighbors.push_back(TilingConfig(
            CubeTilingConfig{c.tileM, c.tileN, c.tileK + cubeKRange.step}, v));
      } else {
        // Try Vector neighbors
        if (v.tileM - vectorRange.m.step >= vectorRange.m.min)
          neighbors.push_back(TilingConfig(c,
            VectorTilingConfig{v.tileM - vectorRange.m.step, v.tileN}));
        if (v.tileM + vectorRange.m.step <= vectorRange.m.max)
          neighbors.push_back(TilingConfig(c,
            VectorTilingConfig{v.tileM + vectorRange.m.step, v.tileN}));
        if (v.tileN - vectorRange.n.step >= vectorRange.n.min)
          neighbors.push_back(TilingConfig(c,
            VectorTilingConfig{v.tileM, v.tileN - vectorRange.n.step}));
        if (v.tileN + vectorRange.n.step <= vectorRange.n.max)
          neighbors.push_back(TilingConfig(c,
            VectorTilingConfig{v.tileM, v.tileN + vectorRange.n.step}));
      }
      
      for (const auto &neighbor : neighbors) {
        TilingResult result = evaluator.evaluate(neighbor);
        if (result.valid && result.totalCycles < best.totalCycles) {
          best = result;
          current = neighbor;
          improved = true;
          
          if (verbose) {
            llvm::outs() << "  Improved: Cube[" << neighbor.cube.tileM 
                         << "," << neighbor.cube.tileN << "," << neighbor.cube.tileK
                         << "] Vec[" << neighbor.vector.tileM << "," << neighbor.vector.tileN
                         << "] cycles=" << result.totalCycles 
                         << " balance=" << llvm::format("%.2f", result.cvBalanceRatio) << "\n";
          }
        }
      }
    }
    
    if (verbose) {
      llvm::outs() << "Heuristic search completed in " << iterations << " iterations\n";
    }
    
    return best;
  }
};

/// Genetic algorithm search
class GeneticSearch : public TilingSearchStrategy {
public:
  GeneticSearch(int maxIterations) : maxIterations(maxIterations) {}
  
  TilingResult searchUnified(TilingEvaluator &evaluator,
                             const SearchRange &mRange,
                             const SearchRange &nRange,
                             const SearchRange &kRange,
                             bool verbose) override {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    auto mValues = mRange.enumerate();
    auto nValues = nRange.enumerate();
    auto kValues = kRange.enumerate();
    
    std::uniform_int_distribution<> mDist(0, mValues.size() - 1);
    std::uniform_int_distribution<> nDist(0, nValues.size() - 1);
    std::uniform_int_distribution<> kDist(0, kValues.size() - 1);
    
    // Initialize population
    const int populationSize = 20;
    std::vector<TilingConfig> population;
    for (int i = 0; i < populationSize; ++i) {
      population.push_back(TilingConfig(
        mValues[mDist(gen)],
        nValues[nDist(gen)],
        kValues[kDist(gen)]
      ));
    }
    
    TilingResult best;
    best.totalCycles = INT64_MAX;
    
    if (verbose) {
      llvm::outs() << "Genetic search (unified): population=" << populationSize
                   << " max_iterations=" << maxIterations << "\n";
    }
    
    for (int iter = 0; iter < maxIterations; ++iter) {
      // Evaluate fitness
      std::vector<std::pair<int64_t, int>> fitness;  // (cycles, index)
      for (int i = 0; i < populationSize; ++i) {
        TilingResult result = evaluator.evaluate(population[i]);
        int64_t cost = result.valid ? result.totalCycles : INT64_MAX;
        fitness.push_back({cost, i});
        
        if (result.valid && result.totalCycles < best.totalCycles) {
          best = result;
          if (verbose) {
            llvm::outs() << "  Gen " << iter << ": New best M=" << population[i].cube.tileM
                         << " N=" << population[i].cube.tileN 
                         << " K=" << population[i].cube.tileK
                         << " cycles=" << result.totalCycles << "\n";
          }
        }
      }
      
      // Sort by fitness
      std::sort(fitness.begin(), fitness.end());
      
      // Selection: keep top half
      std::vector<TilingConfig> newPopulation;
      for (int i = 0; i < populationSize / 2; ++i) {
        newPopulation.push_back(population[fitness[i].second]);
      }
      
      // Crossover and mutation to fill rest
      std::uniform_real_distribution<> prob(0.0, 1.0);
      while (static_cast<int>(newPopulation.size()) < populationSize) {
        int p1 = mDist(gen) % (populationSize / 2);
        int p2 = mDist(gen) % (populationSize / 2);
        
        int64_t childM = prob(gen) < 0.5 ? newPopulation[p1].cube.tileM : newPopulation[p2].cube.tileM;
        int64_t childN = prob(gen) < 0.5 ? newPopulation[p1].cube.tileN : newPopulation[p2].cube.tileN;
        int64_t childK = prob(gen) < 0.5 ? newPopulation[p1].cube.tileK : newPopulation[p2].cube.tileK;
        
        // Mutation
        if (prob(gen) < 0.1) childM = mValues[mDist(gen)];
        if (prob(gen) < 0.1) childN = nValues[nDist(gen)];
        if (prob(gen) < 0.1) childK = kValues[kDist(gen)];
        
        newPopulation.push_back(TilingConfig(childM, childN, childK));
      }
      
      population = newPopulation;
    }
    
    return best;
  }
  
private:
  int maxIterations;
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct TilingOptimizationPass
    : public impl::TilingOptimizationPassBase<TilingOptimizationPass> {
  using TilingOptimizationPassBase::TilingOptimizationPassBase;
  
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // Load hardware config
    if (!hardwareConfigPath.empty()) {
      std::string error;
      if (!loadHardwareConfigFromFile(hardwareConfigPath, error)) {
        emitError(module.getLoc(), error);
        return signalPassFailure();
      }
    }
    const HardwareConfig &config = getHardwareConfig();
    
    // Find matmul operations and extract problem size
    int64_t problemM = 0, problemN = 0, problemK = 0;
    int elementBits = 16;  // Default FP16
    bool hasVectorOps = false;
    
    module.walk([&](MatmulOp matmul) {
      problemM = std::max(problemM, matmul.getM());
      problemN = std::max(problemN, matmul.getN());
      problemK = std::max(problemK, matmul.getK());
      
      // Get element type
      if (auto tensorType = dyn_cast<RankedTensorType>(matmul.getLhs().getType())) {
        Type elemType = tensorType.getElementType();
        if (elemType.isF32()) elementBits = 32;
        else if (elemType.isInteger(8)) elementBits = 8;
        else elementBits = 16;  // FP16/BF16
      }
    });
    
    // Check if there are Vector operations (CV fusion candidate)
    module.walk([&](Operation *op) {
      if (auto cycleOp = dyn_cast<CycleEstimatable>(op)) {
        if (cycleOp.getHWUnit() == HWUnit::Vector) {
          hasVectorOps = true;
        }
      }
    });
    
    if (problemM == 0 || problemN == 0 || problemK == 0) {
      llvm::outs() << "No matmul operations found, skipping tiling optimization\n";
      return;
    }
    
    llvm::outs() << "\n===== Tiling Optimization =====\n";
    llvm::outs() << "Problem size: M=" << problemM << " N=" << problemN 
                 << " K=" << problemK << "\n";
    llvm::outs() << "Element bits: " << elementBits << "\n";
    llvm::outs() << "CV Fusion detected: " << (hasVectorOps ? "yes" : "no") << "\n";
    llvm::outs() << "Independent CV tiling: " << (independentCV ? "enabled" : "disabled") << "\n";
    
    // Get fractal sizes for default alignment
    int fracM, fracK, fracN;
    config.getCubeFractalSize(elementBits, fracM, fracK, fracN);
    
    // Setup Cube search ranges
    SearchRange cubeMRangeVal, cubeNRangeVal, cubeKRangeVal;
    
    cubeMRangeVal.min = fracM;
    cubeMRangeVal.max = std::min(problemM, 256L);
    cubeMRangeVal.step = fracM;
    
    cubeNRangeVal.min = fracN;
    cubeNRangeVal.max = std::min(problemN, 256L);
    cubeNRangeVal.step = fracN;
    
    cubeKRangeVal.min = fracK;
    cubeKRangeVal.max = std::min(problemK, 128L);
    cubeKRangeVal.step = fracK;
    
    // Override with user-specified Cube ranges
    if (!cubeMRange.empty() && !SearchRange::parse(cubeMRange, cubeMRangeVal)) {
      emitError(module.getLoc(), "Invalid cube-m-range format");
      return signalPassFailure();
    }
    if (!cubeNRange.empty() && !SearchRange::parse(cubeNRange, cubeNRangeVal)) {
      emitError(module.getLoc(), "Invalid cube-n-range format");
      return signalPassFailure();
    }
    if (!cubeKRange.empty() && !SearchRange::parse(cubeKRange, cubeKRangeVal)) {
      emitError(module.getLoc(), "Invalid cube-k-range format");
      return signalPassFailure();
    }
    
    llvm::outs() << "\nCube search ranges:\n";
    llvm::outs() << "  M: [" << cubeMRangeVal.min << ", " << cubeMRangeVal.max << "] step " << cubeMRangeVal.step << "\n";
    llvm::outs() << "  N: [" << cubeNRangeVal.min << ", " << cubeNRangeVal.max << "] step " << cubeNRangeVal.step << "\n";
    llvm::outs() << "  K: [" << cubeKRangeVal.min << ", " << cubeKRangeVal.max << "] step " << cubeKRangeVal.step << "\n";
    
    // Create evaluator
    TilingEvaluator evaluator(config, problemM, problemN, problemK, elementBits);
    
    // Create search strategy
    std::unique_ptr<TilingSearchStrategy> searchStrategy;
    if (strategy == "exhaustive") {
      searchStrategy = std::make_unique<ExhaustiveSearch>();
    } else if (strategy == "genetic") {
      searchStrategy = std::make_unique<GeneticSearch>(maxIterations);
    } else {
      searchStrategy = std::make_unique<HeuristicSearch>();
    }
    
    llvm::outs() << "Strategy: " << strategy << "\n\n";
    
    TilingResult best;
    
    if (independentCV && hasVectorOps) {
      // Independent CV tiling search
      VectorSearchRange vectorRangeVal;
      
      // Default Vector ranges (can be smaller tiles for better overlap)
      vectorRangeVal.m.min = 16;
      vectorRangeVal.m.max = std::min(problemM, 256L);
      vectorRangeVal.m.step = 16;
      
      vectorRangeVal.n.min = 16;
      vectorRangeVal.n.max = std::min(problemN, 256L);
      vectorRangeVal.n.step = 16;
      
      // Override with user-specified Vector ranges
      if (!vecMRange.empty() && !SearchRange::parse(vecMRange, vectorRangeVal.m)) {
        emitError(module.getLoc(), "Invalid vec-m-range format");
        return signalPassFailure();
      }
      if (!vecNRange.empty() && !SearchRange::parse(vecNRange, vectorRangeVal.n)) {
        emitError(module.getLoc(), "Invalid vec-n-range format");
        return signalPassFailure();
      }
      
      llvm::outs() << "Vector search ranges:\n";
      llvm::outs() << "  M: [" << vectorRangeVal.m.min << ", " << vectorRangeVal.m.max 
                   << "] step " << vectorRangeVal.m.step << "\n";
      llvm::outs() << "  N: [" << vectorRangeVal.n.min << ", " << vectorRangeVal.n.max 
                   << "] step " << vectorRangeVal.n.step << "\n\n";
      
      // Run independent CV search
      best = searchStrategy->searchIndependent(evaluator, cubeMRangeVal, cubeNRangeVal, 
                                                cubeKRangeVal, vectorRangeVal, verbose);
    } else {
      // Unified tiling search (same tile for Cube and Vector)
      best = searchStrategy->searchUnified(evaluator, cubeMRangeVal, cubeNRangeVal, 
                                           cubeKRangeVal, verbose);
    }
    
    // Report results
    if (!best.valid) {
      llvm::outs() << "No valid tiling configuration found!\n";
      return;
    }
    
    llvm::outs() << "\n===== Optimal Tiling Configuration =====\n";
    if (best.config.independentCV) {
      llvm::outs() << "Cube tiling:\n";
      llvm::outs() << "  tile_M = " << best.config.cube.tileM << "\n";
      llvm::outs() << "  tile_N = " << best.config.cube.tileN << "\n";
      llvm::outs() << "  tile_K = " << best.config.cube.tileK << "\n";
      llvm::outs() << "Vector tiling:\n";
      llvm::outs() << "  tile_M = " << best.config.vector.tileM << "\n";
      llvm::outs() << "  tile_N = " << best.config.vector.tileN << "\n";
    } else {
      llvm::outs() << "Unified tiling:\n";
      llvm::outs() << "  tile_M = " << best.config.cube.tileM << "\n";
      llvm::outs() << "  tile_N = " << best.config.cube.tileN << "\n";
      llvm::outs() << "  tile_K = " << best.config.cube.tileK << "\n";
    }
    
    llvm::outs() << "\nTile Statistics:\n";
    llvm::outs() << "  Cube output tiles: " << best.numCubeTiles << "\n";
    llvm::outs() << "  Vector tiles: " << best.numVectorTiles << "\n";
    llvm::outs() << "  Cube compute per tile: " << best.cubeComputePerTile << " cycles\n";
    llvm::outs() << "  Vector compute per tile: " << best.vectorComputePerTile << " cycles\n";
    
    llvm::outs() << "\nEstimated Performance:\n";
    llvm::outs() << "  Cube path total: " << best.cubeCycles << " cycles\n";
    llvm::outs() << "  Vector path total: " << best.vectorCycles << " cycles\n";
    llvm::outs() << "  CV balance ratio: " << llvm::format("%.2f", best.cvBalanceRatio) << "\n";
    llvm::outs() << "  CV overlap ratio: " << llvm::format("%.1f", best.overlapRatio * 100) << "%\n";
    llvm::outs() << "  Total cycles: " << best.totalCycles << "\n";
    llvm::outs() << "  Estimated time: " 
                 << llvm::format("%.3f", config.cyclesToMicroseconds(best.totalCycles)) 
                 << " us\n";
    llvm::outs() << "  Pipeline efficiency: " 
                 << llvm::format("%.1f", best.pipelineEfficiency * 100) << "%\n";
    
    // Set module attributes with optimal tiling
    auto ctx = module.getContext();
    auto i64Type = IntegerType::get(ctx, 64);
    
    module->setAttr("ascend.optimal_cube_tile_m",
                    IntegerAttr::get(i64Type, best.config.cube.tileM));
    module->setAttr("ascend.optimal_cube_tile_n",
                    IntegerAttr::get(i64Type, best.config.cube.tileN));
    module->setAttr("ascend.optimal_cube_tile_k",
                    IntegerAttr::get(i64Type, best.config.cube.tileK));
    
    if (best.config.independentCV) {
      module->setAttr("ascend.optimal_vector_tile_m",
                      IntegerAttr::get(i64Type, best.config.vector.tileM));
      module->setAttr("ascend.optimal_vector_tile_n",
                      IntegerAttr::get(i64Type, best.config.vector.tileN));
      module->setAttr("ascend.independent_cv_tiling",
                      BoolAttr::get(ctx, true));
    }
    
    module->setAttr("ascend.optimal_cycles",
                    IntegerAttr::get(i64Type, best.totalCycles));
    module->setAttr("ascend.cv_balance_ratio",
                    FloatAttr::get(Float64Type::get(ctx), best.cvBalanceRatio));
  }
};

} // namespace
} // namespace ascend
} // namespace mlir
