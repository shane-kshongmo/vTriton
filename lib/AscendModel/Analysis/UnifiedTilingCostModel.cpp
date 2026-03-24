//===----------------------------------------------------------------------===//
// UnifiedTilingCostModel.cpp - Comprehensive tiling cost model implementation
//
// This implements a unified cost model that combines:
// 1. Memory access minimization (Orojenesis methodology)
// 2. Tile-level pipeline overlap (Cube-Vector parallelism)
// 3. Dependency-aware scheduling (producer-consumer fusion)
//
// The key insight is that optimal tiling must balance:
// - Minimizing data movement (critical for memory-bound operations)
// - Maximizing pipeline utilization (critical for CV fusion)
// - Respecting buffer constraints (critical for feasibility)
//===----------------------------------------------------------------------===//

#include "AscendModel/Analysis/UnifiedTilingCostModel.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace mlir {
namespace ascend {

//===----------------------------------------------------------------------===//
// TensorDesc Implementation
//===----------------------------------------------------------------------===//

int64_t TensorDesc::getFullSize() const {
  int64_t size = elementBytes;
  for (char d : dims) {
    auto it = shape.find(d);
    if (it != shape.end()) size *= it->second;
  }
  return size;
}

int64_t TensorDesc::getTileSize(const std::map<char, int64_t> &tileSizes) const {
  int64_t size = elementBytes;
  for (char d : dims) {
    auto tileIt = tileSizes.find(d);
    auto shapeIt = shape.find(d);
    if (tileIt != tileSizes.end()) {
      size *= tileIt->second;
    } else if (shapeIt != shape.end()) {
      size *= shapeIt->second;  // Not tiled, use full size
    }
  }
  return size;
}

int64_t TensorDesc::calculateAccesses(
    const std::map<char, int64_t> &tileSizes,
    const std::map<char, int64_t> &problemSizes) const {
  
  // Base: each tile is loaded once
  int64_t tileSize = getTileSize(tileSizes);
  
  // Count how many times each tile is accessed
  // Tiles are re-accessed when iterating over dimensions NOT in this tensor
  int64_t reuseIterations = 1;
  
  for (const auto &[dim, fullSize] : problemSizes) {
    // If this dimension is in reuseDims (doesn't affect tensor), 
    // we iterate over it causing re-access of same tiles
    if (reuseDims.count(dim) > 0) {
      auto tileIt = tileSizes.find(dim);
      int64_t tileD = (tileIt != tileSizes.end()) ? tileIt->second : fullSize;
      reuseIterations *= (fullSize + tileD - 1) / tileD;
    }
  }
  
  // Count number of tiles in this tensor
  int64_t numTensorTiles = 1;
  for (char d : dims) {
    auto shapeIt = shape.find(d);
    auto tileIt = tileSizes.find(d);
    if (shapeIt != shape.end()) {
      int64_t tileD = (tileIt != tileSizes.end()) ? tileIt->second : shapeIt->second;
      numTensorTiles *= (shapeIt->second + tileD - 1) / tileD;
    }
  }
  
  // Total accesses = numTiles * tileSize * reuseIterations
  // But this overcounts because tiles can be reused within a reuse iteration
  // Simplified: accesses = tensorSize * reuseIterations
  return getFullSize() * reuseIterations;
}

//===----------------------------------------------------------------------===//
// OpDesc Implementation
//===----------------------------------------------------------------------===//

int64_t OpDesc::getTotalFlops() const {
  int64_t outputElems = getOutputElements();
  return outputElems * flopsPerOutputElement;
}

int64_t OpDesc::getOutputElements() const {
  int64_t elems = 1;
  for (char d : output.dims) {
    auto it = dims.find(d);
    if (it != dims.end()) {
      elems *= it->second.size;
    }
  }
  return elems;
}

//===----------------------------------------------------------------------===//
// UnifiedTilingCostModel Implementation
//===----------------------------------------------------------------------===//

UnifiedTilingCostModel::UnifiedTilingCostModel(const HardwareConfig &config)
    : hwConfig(config) {
  
  // Cache hardware parameters
  l0aSize = config.getMemorySizeBytes("l0a");
  l0bSize = config.getMemorySizeBytes("l0b");
  l0cSize = config.getMemorySizeBytes("l0c");
  ubSize = config.getMemorySizeBytes("ub");
  l1Size = config.getMemorySizeBytes("l1");
  
  // Set defaults if not specified
  if (l0aSize == 0) l0aSize = 64 * 1024;
  if (l0bSize == 0) l0bSize = 64 * 1024;
  if (l0cSize == 0) l0cSize = 256 * 1024;
  if (ubSize == 0) ubSize = 256 * 1024;
  if (l1Size == 0) l1Size = 1024 * 1024;
  
  hbmBandwidth = config.getHBMBandwidthGBs();
  if (hbmBandwidth == 0) hbmBandwidth = 400.0;  // Default for 910B
  
  clockFreqGHz = config.getClockFrequencyGHz();
  if (clockFreqGHz == 0) clockFreqGHz = 1.8;
  
  // Cube peak throughput in MACs/cycle: TFLOPS * 1e12 / (freq_GHz * 1e9) / 2
  // For 910B: 320 TFLOPS / (1.85 GHz * 2) = ~86486 MACs/cycle
  // But this is the aggregate; per-cube-core it's typically 256 MACs/cycle
  cubePeakThroughput = 256;  // MACs/cycle for a single cube core
  
  vectorWidth = config.getVectorWidthElements();
  if (vectorWidth == 0) vectorWidth = 128;
  
  // MTE bandwidths (bytes per cycle)
  mte2Bandwidth = static_cast<int64_t>(hbmBandwidth * 1e9 / (clockFreqGHz * 1e9));
  mte3Bandwidth = mte2Bandwidth;
  
  // Startup latencies
  cubeStartupLatency = config.getCubeStartupLatency();
  vectorStartupLatency = config.getVectorStartupLatency();
  mte2StartupLatency = config.getMTE2StartupLatency();
  mte3StartupLatency = config.getMTE3StartupLatency();
  
  // Cache fractal sizes for common precisions
  int fm, fk, fn;
  config.getCubeFractalSize(16, fm, fk, fn);  // FP16
  fractalSizes[16] = {fm, fk, fn};
  config.getCubeFractalSize(32, fm, fk, fn);  // FP32
  fractalSizes[32] = {fm, fk, fn};
  config.getCubeFractalSize(8, fm, fk, fn);   // INT8
  fractalSizes[8] = {fm, fk, fn};
}

//===----------------------------------------------------------------------===//
// Memory Cost Model (Orojenesis-inspired)
//===----------------------------------------------------------------------===//

UnifiedTilingCostModel::MemoryCost 
UnifiedTilingCostModel::computeMemoryCost(
    const OpDesc &op,
    const UnifiedTilingConfig &tiling) {
  
  MemoryCost cost;
  cost.totalAccesses = 0;
  
  // Build problem size map
  std::map<char, int64_t> problemSizes;
  for (const auto &[label, dim] : op.dims) {
    problemSizes[label] = dim.size;
  }
  
  // Calculate accesses for each input tensor
  for (const auto &tensor : op.inputs) {
    int64_t accesses = tensor.calculateAccesses(tiling.tileSizes, problemSizes);
    cost.perTensorAccesses[tensor.name] = accesses;
    cost.totalAccesses += accesses;
  }
  
  // Calculate accesses for output tensor
  // For reduction ops: output may need read-modify-write
  // For non-reduction: output written once
  {
    int64_t outputAccesses = op.output.getFullSize();  // Write once
    
    // If there's K (reduction) dimension and K is tiled, 
    // output needs accumulation across K tiles
    bool hasReductionTiling = false;
    for (const auto &[label, dim] : op.dims) {
      if (dim.isReduction) {
        auto tileIt = tiling.tileSizes.find(label);
        if (tileIt != tiling.tileSizes.end() && tileIt->second < dim.size) {
          hasReductionTiling = true;
          break;
        }
      }
    }
    
    if (hasReductionTiling) {
      // Need to read partial results back for accumulation
      // Unless hardware supports on-chip accumulation (L0C)
      // For now, assume L0C handles this
    }
    
    cost.perTensorAccesses[op.output.name] = outputAccesses;
    cost.totalAccesses += outputAccesses;
  }
  
  // Calculate algorithmic minimum for data reuse ratio
  int64_t algoMin = 0;
  for (const auto &tensor : op.inputs) {
    algoMin += tensor.getFullSize();
  }
  algoMin += op.output.getFullSize();
  
  cost.dataReuseRatio = static_cast<double>(algoMin) / 
                        std::max(cost.totalAccesses, 1L);
  
  // Convert to cycles based on bandwidth
  double bytesPerCycle = hbmBandwidth * 1e9 / (clockFreqGHz * 1e9);
  cost.hbmCycles = static_cast<int64_t>(cost.totalAccesses / bytesPerCycle);
  
  return cost;
}

//===----------------------------------------------------------------------===//
// Compute Cost Model
//===----------------------------------------------------------------------===//

UnifiedTilingCostModel::ComputeCost 
UnifiedTilingCostModel::computeComputeCost(
    const OpDesc &op,
    const UnifiedTilingConfig &tiling) {
  
  ComputeCost cost;
  cost.cubeCycles = 0;
  cost.vectorCycles = 0;
  
  if (op.hwUnit == HWUnit::Cube) {
    // Fractal-based cycle estimation for Cube
    int64_t elemBits = op.inputs[0].elementBytes * 8;
    auto fracIt = fractalSizes.find(elemBits);
    int fracM = 16, fracK = 16, fracN = 16;
    if (fracIt != fractalSizes.end()) {
      std::tie(fracM, fracK, fracN) = fracIt->second;
    }
    
    // Get M, N, K dimensions
    int64_t M = 1, N = 1, K = 1;
    for (const auto &[label, dim] : op.dims) {
      if (label == 'm') M = dim.size;
      else if (label == 'n') N = dim.size;
      else if (label == 'k') K = dim.size;
    }
    
    // Get tile sizes
    int64_t tileM = tiling.getTile('m', M);
    int64_t tileN = tiling.getTile('n', N);
    int64_t tileK = tiling.getTile('k', K);
    
    // Number of tiles
    int64_t tilesM = (M + tileM - 1) / tileM;
    int64_t tilesN = (N + tileN - 1) / tileN;
    int64_t tilesK = (K + tileK - 1) / tileK;
    
    // Fractal operations per tile
    int64_t fracOpsPerTile = ((tileM + fracM - 1) / fracM) *
                             ((tileK + fracK - 1) / fracK) *
                             ((tileN + fracN - 1) / fracN);
    
    // Total cube cycles
    cost.cubeCycles = fracOpsPerTile * tilesM * tilesN * tilesK;
    cost.cubeCycles += cubeStartupLatency;
    
    // Calculate occupancy (utilization of fractal unit)
    int64_t idealFracOps = ((M + fracM - 1) / fracM) *
                           ((K + fracK - 1) / fracK) *
                           ((N + fracN - 1) / fracN);
    cost.cubeOccupancy = (idealFracOps * 100) / std::max(cost.cubeCycles, 1L);
    
  } else if (op.hwUnit == HWUnit::Vector) {
    // Element-based estimation for Vector
    int64_t totalElements = op.getOutputElements();
    int64_t opsPerElement = op.flopsPerOutputElement;
    
    cost.vectorCycles = (totalElements * opsPerElement + vectorWidth - 1) / vectorWidth;
    cost.vectorCycles += vectorStartupLatency;
    
    // Vector occupancy (percentage of lanes utilized)
    int64_t idealCycles = totalElements * opsPerElement / vectorWidth;
    cost.vectorOccupancy = (idealCycles * 100) / std::max(cost.vectorCycles, 1L);
  }
  
  return cost;
}

//===----------------------------------------------------------------------===//
// Pipeline Overlap Model (Tile-level CV parallelism)
//===----------------------------------------------------------------------===//

UnifiedTilingCostModel::PipelineCost 
UnifiedTilingCostModel::computePipelineCost(
    const MemoryCost &mem,
    const ComputeCost &compute,
    const UnifiedTilingConfig &tiling,
    int64_t numTiles) {
  
  PipelineCost cost;
  
  // Serial execution: all memory + all compute
  int64_t memCycles = mem.hbmCycles;
  int64_t cubeCycles = compute.cubeCycles;
  int64_t vectorCycles = compute.vectorCycles;
  
  cost.serialCycles = memCycles + cubeCycles + vectorCycles;
  
  //=========================================================================//
  // Pipeline overlap analysis
  //
  // On Ascend NPU, we have multiple independent units:
  // - MTE2: Load from HBM to L1/UB
  // - Cube: Matrix computation
  // - Vector: Vector computation  
  // - MTE3: Store from L1/UB to HBM
  //
  // With proper scheduling (double buffering), these can overlap:
  //
  // Time ->
  // MTE2:  [Load T0][Load T1][Load T2]...
  // Cube:       [Cube T0][Cube T1][Cube T2]...
  // Vector:          [Vec T0][Vec T1][Vec T2]...
  // MTE3:                [Store T0][Store T1]...
  //
  // The total time is dominated by the slowest stage + startup/drain overhead
  //=========================================================================//
  
  if (numTiles <= 1) {
    // No pipelining possible with single tile
    cost.pipelinedCycles = cost.serialCycles;
    cost.overlapRatio = 0.0;
    cost.effectivePipelineDepth = 1;
    
  } else {
    // Per-tile breakdown
    int64_t memCyclesPerTile = memCycles / numTiles;
    int64_t cubeCyclesPerTile = cubeCycles / std::max(numTiles, 1L);
    int64_t vectorCyclesPerTile = vectorCycles / std::max(numTiles, 1L);
    
    // Find the bottleneck (longest stage)
    int64_t bottleneckCycles = std::max({memCyclesPerTile, 
                                          cubeCyclesPerTile, 
                                          vectorCyclesPerTile});
    
    // Steady-state: pipeline runs at rate of bottleneck
    // Startup: need to fill pipeline (depth = number of stages)
    // Drain: need to empty pipeline
    
    int64_t pipelineDepth = 3;  // MTE2 -> Cube/Vector -> MTE3
    if (cubeCycles > 0 && vectorCycles > 0) {
      pipelineDepth = 4;  // MTE2 -> Cube -> Vector -> MTE3
    }
    
    int64_t steadyStateTiles = std::max(numTiles - pipelineDepth + 1, 1L);
    int64_t startupDrainTiles = numTiles - steadyStateTiles;
    
    // Total pipelined cycles
    cost.pipelinedCycles = 
        bottleneckCycles * steadyStateTiles +  // Steady state
        cost.serialCycles / numTiles * startupDrainTiles;  // Startup + drain
    
    // Add startup latencies once
    cost.pipelinedCycles += mte2StartupLatency + mte3StartupLatency;
    
    // Calculate overlap ratio
    // If perfect overlap: time = max(mem, cube, vector)
    // If no overlap: time = mem + cube + vector
    int64_t perfectOverlap = std::max({memCycles, cubeCycles, vectorCycles});
    cost.overlapRatio = static_cast<double>(cost.serialCycles - cost.pipelinedCycles) /
                        static_cast<double>(cost.serialCycles - perfectOverlap + 1);
    cost.overlapRatio = std::clamp(cost.overlapRatio, 0.0, 1.0);
    
    cost.effectivePipelineDepth = pipelineDepth;
  }
  
  // CV balance (for CV fusion operations)
  if (cubeCycles > 0 && vectorCycles > 0) {
    cost.cvBalance = static_cast<double>(std::min(cubeCycles, vectorCycles)) /
                     static_cast<double>(std::max(cubeCycles, vectorCycles));
  } else {
    cost.cvBalance = 1.0;  // Only one unit used, "perfect" balance
  }
  
  return cost;
}

//===----------------------------------------------------------------------===//
// Buffer Constraint Checking
//===----------------------------------------------------------------------===//

UnifiedTilingCostModel::BufferCheck 
UnifiedTilingCostModel::checkBufferConstraints(
    const OpDesc &op,
    const UnifiedTilingConfig &tiling) {
  
  BufferCheck check;
  check.valid = true;
  check.l0aUsed = 0;
  check.l0bUsed = 0;
  check.l0cUsed = 0;
  check.ubUsed = 0;
  check.l1Used = 0;
  
  if (op.hwUnit == HWUnit::Cube) {
    // Cube operations use L0A, L0B, L0C
    
    // Get tile sizes
    int64_t tileM = tiling.getTile('m', 1);
    int64_t tileN = tiling.getTile('n', 1);
    int64_t tileK = tiling.getTile('k', 1);
    int64_t elemBytes = op.inputs[0].elementBytes;
    
    // A tile goes to L0A: [tileM, tileK]
    check.l0aUsed = tileM * tileK * elemBytes;
    if (tiling.doubleBufferInputs) check.l0aUsed *= 2;
    
    // B tile goes to L0B: [tileK, tileN]
    check.l0bUsed = tileK * tileN * elemBytes;
    if (tiling.doubleBufferInputs) check.l0bUsed *= 2;
    
    // C tile goes to L0C: [tileM, tileN]
    // L0C typically uses higher precision for accumulation
    check.l0cUsed = tileM * tileN * 4;  // FP32 accumulator
    
    // L1 staging for A and B
    check.l1Used = check.l0aUsed + check.l0bUsed;
    
    // Check constraints
    if (check.l0aUsed > l0aSize) {
      check.valid = false;
      check.reason = "A tile (" + std::to_string(check.l0aUsed) + 
                     " bytes) exceeds L0A (" + std::to_string(l0aSize) + ")";
      return check;
    }
    if (check.l0bUsed > l0bSize) {
      check.valid = false;
      check.reason = "B tile (" + std::to_string(check.l0bUsed) + 
                     " bytes) exceeds L0B (" + std::to_string(l0bSize) + ")";
      return check;
    }
    if (check.l0cUsed > l0cSize) {
      check.valid = false;
      check.reason = "C tile (" + std::to_string(check.l0cUsed) + 
                     " bytes) exceeds L0C (" + std::to_string(l0cSize) + ")";
      return check;
    }
    if (check.l1Used > l1Size) {
      check.valid = false;
      check.reason = "Input tiles (" + std::to_string(check.l1Used) + 
                     " bytes) exceed L1 (" + std::to_string(l1Size) + ")";
      return check;
    }
    
  } else if (op.hwUnit == HWUnit::Vector) {
    // Vector operations use UB
    int64_t inputTileSize = op.inputs[0].getTileSize(tiling.tileSizes);
    int64_t outputTileSize = op.output.getTileSize(tiling.tileSizes);
    
    check.ubUsed = inputTileSize + outputTileSize;
    if (tiling.doubleBufferInputs) check.ubUsed += inputTileSize;
    if (tiling.doubleBufferOutputs) check.ubUsed += outputTileSize;
    
    if (check.ubUsed > ubSize) {
      check.valid = false;
      check.reason = "Vector tiles (" + std::to_string(check.ubUsed) + 
                     " bytes) exceed UB (" + std::to_string(ubSize) + ")";
      return check;
    }
  }
  
  return check;
}

//===----------------------------------------------------------------------===//
// Unified Evaluation
//===----------------------------------------------------------------------===//

CostBreakdown UnifiedTilingCostModel::evaluate(
    const OpDesc &op,
    const UnifiedTilingConfig &tiling) {
  
  CostBreakdown result;
  
  // 1. Check buffer constraints first
  auto bufferCheck = checkBufferConstraints(op, tiling);
  if (!bufferCheck.valid) {
    result.totalCycles = INT64_MAX;
    return result;  // Invalid configuration
  }
  
  result.l0aRequired = bufferCheck.l0aUsed;
  result.l0bRequired = bufferCheck.l0bUsed;
  result.l0cRequired = bufferCheck.l0cUsed;
  result.ubRequired = bufferCheck.ubUsed;
  result.l1Required = bufferCheck.l1Used;
  
  // 2. Compute memory cost
  auto memCost = computeMemoryCost(op, tiling);
  result.totalMemoryAccesses = memCost.totalAccesses;
  result.tensorAccesses = memCost.perTensorAccesses;
  result.hbmAccessCycles = memCost.hbmCycles;
  
  // 3. Compute compute cost
  auto computeCost = computeComputeCost(op, tiling);
  result.cubeCycles = computeCost.cubeCycles;
  result.vectorCycles = computeCost.vectorCycles;
  result.totalComputeCycles = computeCost.cubeCycles + computeCost.vectorCycles;
  
  // 4. Calculate number of tiles
  result.numTiles = 1;
  for (const auto &[label, dim] : op.dims) {
    if (!dim.isReduction) {  // Only count output tiles
      int64_t tileD = tiling.getTile(label, dim.size);
      result.numTiles *= (dim.size + tileD - 1) / tileD;
    }
  }
  
  // 5. Compute pipeline overlap
  auto pipelineCost = computePipelineCost(memCost, computeCost, tiling, result.numTiles);
  result.serialCycles = pipelineCost.serialCycles;
  result.pipelinedCycles = pipelineCost.pipelinedCycles;
  result.pipelineEfficiency = (pipelineCost.serialCycles > 0) ?
      static_cast<double>(pipelineCost.serialCycles) / 
      (2.0 * pipelineCost.pipelinedCycles) : 0.5;
  result.cvBalanceRatio = pipelineCost.cvBalance;
  
  // 6. Final total cycles
  result.totalCycles = pipelineCost.pipelinedCycles;
  
  // 7. Derived metrics
  int64_t flops = op.getTotalFlops();
  result.operationalIntensity = static_cast<double>(flops) / 
                                 std::max(result.totalMemoryAccesses, 1L);
  
  result.memComputeRatio = static_cast<double>(result.hbmAccessCycles) /
                           std::max(result.totalComputeCycles, 1L);
  
  // Hardware utilization: achieved throughput / peak throughput
  double achievedThroughput = static_cast<double>(flops) / result.totalCycles;
  double peakThroughput = static_cast<double>(cubePeakThroughput);  // Simplified
  result.hardwareUtilization = achievedThroughput / peakThroughput;
  
  result.cyclesPerTile = result.totalCycles / std::max(result.numTiles, 1L);
  
  return result;
}

//===----------------------------------------------------------------------===//
// Fusion Analysis
//===----------------------------------------------------------------------===//

std::vector<UnifiedTilingCostModel::DependencyInfo>
UnifiedTilingCostModel::analyzeDependencies(
    const std::vector<OpDesc> &ops,
    const UnifiedTilingConfig &tiling) {
  
  std::vector<DependencyInfo> deps;
  
  for (size_t i = 0; i < ops.size(); ++i) {
    for (size_t j = i + 1; j < ops.size(); ++j) {
      // Check if ops[i].output is consumed by ops[j]
      for (const auto &input : ops[j].inputs) {
        if (input.name == ops[i].output.name) {
          DependencyInfo dep;
          dep.producer = ops[i].name;
          dep.consumer = ops[j].name;
          dep.intermediateTensor = input.name;
          dep.intermediateSize = input.getTileSize(tiling.tileSizes);
          
          // Can fuse if intermediate fits in buffer
          int64_t availableBuffer = (ops[i].hwUnit == HWUnit::Cube) ? 
                                    l0cSize : ubSize;
          dep.canFuse = (dep.intermediateSize <= availableBuffer);
          
          // Calculate saved accesses from fusion
          // Without fusion: write intermediate + read intermediate
          // With fusion: intermediate stays on-chip
          dep.savedAccesses = dep.canFuse ? 
              (ops[i].output.getFullSize() * 2) : 0;
          
          deps.push_back(dep);
        }
      }
    }
  }
  
  return deps;
}

CostBreakdown UnifiedTilingCostModel::evaluateFused(
    const std::vector<OpDesc> &ops,
    const UnifiedTilingConfig &tiling) {
  
  CostBreakdown result;
  
  if (ops.empty()) return result;
  
  // Analyze dependencies
  auto deps = analyzeDependencies(ops, tiling);
  
  // Calculate total cost with fusion benefits
  int64_t totalAccesses = 0;
  int64_t totalCubeCycles = 0;
  int64_t totalVectorCycles = 0;
  int64_t maxBufferRequired = 0;
  
  std::set<std::string> fusedIntermediates;
  for (const auto &dep : deps) {
    if (dep.canFuse) {
      fusedIntermediates.insert(dep.intermediateTensor);
    }
  }
  
  for (size_t i = 0; i < ops.size(); ++i) {
    auto singleCost = evaluate(ops[i], tiling);
    
    // Add compute cycles
    totalCubeCycles += singleCost.cubeCycles;
    totalVectorCycles += singleCost.vectorCycles;
    maxBufferRequired = std::max(maxBufferRequired, 
                                  singleCost.l0aRequired + singleCost.l0bRequired +
                                  singleCost.l0cRequired + singleCost.ubRequired);
    
    // Add memory accesses, excluding fused intermediates
    for (const auto &tensor : ops[i].inputs) {
      if (fusedIntermediates.count(tensor.name) == 0) {
        totalAccesses += singleCost.tensorAccesses[tensor.name];
      }
    }
    
    // Only count output of last op (or non-fused outputs)
    if (i == ops.size() - 1 || 
        fusedIntermediates.count(ops[i].output.name) == 0) {
      totalAccesses += singleCost.tensorAccesses[ops[i].output.name];
    }
  }
  
  result.totalMemoryAccesses = totalAccesses;
  result.cubeCycles = totalCubeCycles;
  result.vectorCycles = totalVectorCycles;
  result.totalComputeCycles = totalCubeCycles + totalVectorCycles;
  
  // Calculate pipelined cycles for fused execution
  // With fusion, CV can overlap at tile granularity
  
  int64_t numOutputTiles = 1;
  // Use first op's output dimensions for tile count
  for (const auto &[label, dim] : ops[0].dims) {
    if (!dim.isReduction) {
      int64_t tileD = tiling.getTile(label, dim.size);
      numOutputTiles *= (dim.size + tileD - 1) / tileD;
    }
  }
  result.numTiles = numOutputTiles;
  
  // Memory cycles
  double bytesPerCycle = hbmBandwidth * 1e9 / (clockFreqGHz * 1e9);
  result.hbmAccessCycles = static_cast<int64_t>(totalAccesses / bytesPerCycle);
  
  // Pipeline with fusion: Cube and Vector can overlap better
  // because intermediate data transfers are eliminated
  
  int64_t cubePerTile = totalCubeCycles / std::max(numOutputTiles, 1L);
  int64_t vectorPerTile = totalVectorCycles / std::max(numOutputTiles, 1L);
  int64_t memPerTile = result.hbmAccessCycles / std::max(numOutputTiles, 1L);
  
  // Bottleneck determines throughput in steady state
  int64_t bottleneck = std::max({cubePerTile, vectorPerTile, memPerTile});
  
  // Pipelined execution
  int64_t startupCycles = cubePerTile + vectorPerTile + memPerTile;  // Fill pipeline
  int64_t steadyStateCycles = bottleneck * (numOutputTiles - 1);
  result.pipelinedCycles = startupCycles + steadyStateCycles + 
                           cubeStartupLatency + vectorStartupLatency +
                           mte2StartupLatency + mte3StartupLatency;
  
  result.serialCycles = result.hbmAccessCycles + totalCubeCycles + totalVectorCycles;
  result.totalCycles = result.pipelinedCycles;
  
  // CV balance in fusion
  if (totalCubeCycles > 0 && totalVectorCycles > 0) {
    result.cvBalanceRatio = static_cast<double>(std::min(totalCubeCycles, totalVectorCycles)) /
                            static_cast<double>(std::max(totalCubeCycles, totalVectorCycles));
  }
  
  // OI with fusion (should be higher due to reduced memory accesses)
  int64_t totalFlops = 0;
  for (const auto &op : ops) {
    totalFlops += op.getTotalFlops();
  }
  result.operationalIntensity = static_cast<double>(totalFlops) /
                                 std::max(result.totalMemoryAccesses, 1L);
  
  return result;
}

//===----------------------------------------------------------------------===//
// Cost Function Implementation
//===----------------------------------------------------------------------===//

double TilingCostFunction::operator()(
    const OpDesc &op,
    const UnifiedTilingConfig &tiling) const {
  
  auto breakdown = costModel.evaluate(op, tiling);
  
  if (breakdown.totalCycles == INT64_MAX) {
    return std::numeric_limits<double>::max();  // Invalid
  }
  
  switch (objective) {
    case Objective::MinMemoryAccess:
      return static_cast<double>(breakdown.totalMemoryAccesses);
      
    case Objective::MinCycles:
      return static_cast<double>(breakdown.totalCycles);
      
    case Objective::MaxThroughput:
      // Negative because we minimize cost
      return -breakdown.hardwareUtilization;
      
    case Objective::Balanced:
      // Weighted combination
      return memWeight * breakdown.totalMemoryAccesses / 1e9 +  // Normalize to GB
             computeWeight * breakdown.totalCycles / 1e6 +       // Normalize to M cycles
             pipelineWeight * (1.0 - breakdown.pipelineEfficiency) * 1000;
  }
  
  return static_cast<double>(breakdown.totalCycles);
}

double TilingCostFunction::operator()(
    const std::vector<OpDesc> &ops,
    const UnifiedTilingConfig &tiling) const {
  
  auto breakdown = costModel.evaluateFused(ops, tiling);
  
  if (breakdown.totalCycles == INT64_MAX) {
    return std::numeric_limits<double>::max();
  }
  
  switch (objective) {
    case Objective::MinMemoryAccess:
      return static_cast<double>(breakdown.totalMemoryAccesses);
      
    case Objective::MinCycles:
      return static_cast<double>(breakdown.totalCycles);
      
    case Objective::MaxThroughput:
      return -breakdown.hardwareUtilization;
      
    case Objective::Balanced:
      return memWeight * breakdown.totalMemoryAccesses / 1e9 +
             computeWeight * breakdown.totalCycles / 1e6 +
             pipelineWeight * (1.0 - breakdown.pipelineEfficiency) * 1000;
  }
  
  return static_cast<double>(breakdown.totalCycles);
}

//===----------------------------------------------------------------------===//
// Optimizer Implementation
//===----------------------------------------------------------------------===//

UnifiedTilingOptimizer::UnifiedTilingOptimizer(const HardwareConfig &config)
    : costModel(config) {}

void UnifiedTilingOptimizer::setSearchRange(
    char dim, int64_t minTile, int64_t maxTile, int64_t step) {
  searchRanges[dim] = {minTile, maxTile, step};
}

void UnifiedTilingOptimizer::setDefaultRange(
    int64_t minTile, int64_t maxTile, int64_t step) {
  defaultMinTile = minTile;
  defaultMaxTile = maxTile;
  defaultStep = step;
}

UnifiedTilingOptimizer::SearchResult 
UnifiedTilingOptimizer::optimize(
    const OpDesc &op,
    TilingCostFunction::Objective objective) {
  
  TilingCostFunction costFn(costModel, objective);
  return exhaustiveSearch(op, costFn);
}

UnifiedTilingOptimizer::SearchResult 
UnifiedTilingOptimizer::exhaustiveSearch(
    const OpDesc &op,
    const TilingCostFunction &costFn) {
  
  SearchResult result;
  result.configurationsExplored = 0;
  double bestCost = std::numeric_limits<double>::max();
  
  // Collect dimensions to tile
  std::vector<std::pair<char, DimDesc>> tileDims;
  for (const auto &[label, dim] : op.dims) {
    tileDims.push_back({label, dim});
  }
  
  // Get search ranges for each dimension
  std::vector<std::tuple<int64_t, int64_t, int64_t>> ranges;  // min, max, step
  for (const auto &[label, dim] : tileDims) {
    auto rangeIt = searchRanges.find(label);
    if (rangeIt != searchRanges.end()) {
      ranges.push_back(rangeIt->second);
    } else {
      ranges.push_back({defaultMinTile, 
                        std::min(dim.size, defaultMaxTile), 
                        defaultStep});
    }
  }
  
  // Generate all combinations (for up to 3 dimensions)
  auto enumerate = [](int64_t min, int64_t max, int64_t step) {
    std::vector<int64_t> values;
    for (int64_t v = min; v <= max; v += step) {
      values.push_back(v);
    }
    return values;
  };
  
  // Simple exhaustive search for M, N, K
  if (tileDims.size() >= 3) {
    auto mValues = enumerate(std::get<0>(ranges[0]), std::get<1>(ranges[0]), std::get<2>(ranges[0]));
    auto nValues = enumerate(std::get<0>(ranges[1]), std::get<1>(ranges[1]), std::get<2>(ranges[1]));
    auto kValues = enumerate(std::get<0>(ranges[2]), std::get<1>(ranges[2]), std::get<2>(ranges[2]));
    
    for (int64_t m : mValues) {
      for (int64_t n : nValues) {
        for (int64_t k : kValues) {
          UnifiedTilingConfig config;
          config.tileSizes[tileDims[0].first] = m;
          config.tileSizes[tileDims[1].first] = n;
          config.tileSizes[tileDims[2].first] = k;
          
          double cost = costFn(op, config);
          result.configurationsExplored++;
          
          if (cost < bestCost) {
            bestCost = cost;
            result.bestConfig = config;
            result.bestCost = costFn.getBreakdown(op, config);
          }
          
          // Build Pareto frontier
          auto breakdown = costFn.getBreakdown(op, config);
          if (breakdown.totalCycles < INT64_MAX) {
            // Check if this is Pareto-optimal
            bool dominated = false;
            for (auto &[_, existingCost] : result.paretoFrontier) {
              if (existingCost.totalMemoryAccesses <= breakdown.totalMemoryAccesses &&
                  existingCost.totalCycles <= breakdown.totalCycles &&
                  (existingCost.totalMemoryAccesses < breakdown.totalMemoryAccesses ||
                   existingCost.totalCycles < breakdown.totalCycles)) {
                dominated = true;
                break;
              }
            }
            if (!dominated) {
              // Remove dominated points
              result.paretoFrontier.erase(
                std::remove_if(result.paretoFrontier.begin(), result.paretoFrontier.end(),
                  [&breakdown](const auto &p) {
                    return breakdown.totalMemoryAccesses <= p.second.totalMemoryAccesses &&
                           breakdown.totalCycles <= p.second.totalCycles &&
                           (breakdown.totalMemoryAccesses < p.second.totalMemoryAccesses ||
                            breakdown.totalCycles < p.second.totalCycles);
                  }),
                result.paretoFrontier.end());
              result.paretoFrontier.push_back({config, breakdown});
            }
          }
        }
      }
    }
  }
  
  if (verbose) {
    llvm::outs() << "Explored " << result.configurationsExplored << " configurations\n";
    llvm::outs() << "Best cost: " << bestCost << "\n";
    llvm::outs() << "Pareto frontier size: " << result.paretoFrontier.size() << "\n";
  }
  
  return result;
}

UnifiedTilingOptimizer::SearchResult
UnifiedTilingOptimizer::optimizeFused(
    const std::vector<OpDesc> &ops,
    TilingCostFunction::Objective objective) {

  SearchResult result;
  result.configurationsExplored = 0;
  double bestCost = std::numeric_limits<double>::max();

  if (ops.empty())
    return result;

  // Use the first op's dims to drive the search space
  const OpDesc &primary = ops[0];
  std::vector<std::pair<char, DimDesc>> tileDims;
  for (const auto &[label, dim] : primary.dims)
    tileDims.push_back({label, dim});

  auto enumerate = [](int64_t min, int64_t max, int64_t step) {
    std::vector<int64_t> values;
    for (int64_t v = min; v <= max; v += step)
      values.push_back(v);
    return values;
  };

  std::vector<std::tuple<int64_t, int64_t, int64_t>> ranges;
  for (const auto &[label, dim] : tileDims) {
    auto rangeIt = searchRanges.find(label);
    if (rangeIt != searchRanges.end())
      ranges.push_back(rangeIt->second);
    else
      ranges.push_back({defaultMinTile,
                        std::min(dim.size, defaultMaxTile),
                        defaultStep});
  }

  if (tileDims.size() >= 3) {
    auto mValues = enumerate(std::get<0>(ranges[0]), std::get<1>(ranges[0]), std::get<2>(ranges[0]));
    auto nValues = enumerate(std::get<0>(ranges[1]), std::get<1>(ranges[1]), std::get<2>(ranges[1]));
    auto kValues = enumerate(std::get<0>(ranges[2]), std::get<1>(ranges[2]), std::get<2>(ranges[2]));

    for (int64_t m : mValues) {
      for (int64_t n : nValues) {
        for (int64_t k : kValues) {
          UnifiedTilingConfig config;
          config.tileSizes[tileDims[0].first] = m;
          config.tileSizes[tileDims[1].first] = n;
          config.tileSizes[tileDims[2].first] = k;

          auto breakdown = costModel.evaluateFused(ops, config);
          result.configurationsExplored++;

          double cost = static_cast<double>(breakdown.totalCycles);
          if (objective == TilingCostFunction::Objective::MinMemoryAccess)
            cost = static_cast<double>(breakdown.totalMemoryAccesses);

          if (cost < bestCost) {
            bestCost = cost;
            result.bestConfig = config;
            result.bestCost = breakdown;
          }

          if (breakdown.totalCycles < INT64_MAX)
            result.paretoFrontier.push_back({config, breakdown});
        }
      }
    }
  }

  if (verbose) {
    llvm::outs() << "Explored " << result.configurationsExplored
                 << " fused configurations\n";
    llvm::outs() << "Best cost: " << bestCost << "\n";
  }

  return result;
}

} // namespace ascend
} // namespace mlir
