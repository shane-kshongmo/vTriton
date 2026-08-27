//===- PipelineAnalysisPass.cpp - Pipeline scheduling analysis -----------===//
//
// Uses HardwareConfig for configurable hardware parameters.
// Handles dynamic loop bounds via arg-bindings option (supports program_id).
// Generates Perfetto trace with loop unrolling visualization.
// Uses Roofline model for cycle estimation with HW unit overlap.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/IR/AscendModelDialect.h"
#include "AscendModel/Transforms/Passes.h"
#include "AscendModel/Analysis/PipelineAnalysis.h"
#include "AscendModel/HardwareConfig.h"
#include "AscendModel/Analysis/KernelLaunchUtils.h"
#include "AscendModel/Utils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace mlir {
namespace ascend {

#define GEN_PASS_DEF_PIPELINEANALYSISPASS
#include "AscendModel/Transforms/Passes.h.inc"

namespace {

using utils::getScfForTripCount;
using utils::getLoopMultiplier;
using utils::getScfForTripCountWithBindings;
using utils::parseBindings;
using utils::parseLoopTripCounts;

struct TileMixParams {
  bool hasAny = false;
  int64_t vectorLoop = 0;
  int64_t cubeLoop = 0;
  bool summaryPresent = false;
  bool summaryValid = false;
  bool vectorApplied = false;
  bool cubeApplied = false;
  int64_t vectorSegments = 1;
  int64_t cubeSegments = 1;
  int64_t syncOpsBefore = 0;
  int64_t syncOpsAfter = 0;
  std::string summarySource = "missing";
  std::string vectorSkipReason = "missing_pass_summary";
  std::string cubeSkipReason = "missing_pass_summary";
};

struct WorkspaceMultibufferParams {
  bool present = false;
  bool workspacePresent = false;
  bool localAutoPresent = false;
  bool localAutoEnabled = true;
  bool workspaceOnlyLocal = false;
  int64_t requestedSlots = 1;
  int64_t numStages = 2;
  std::string localScope = "no-limit";
  std::string limitedBuffer;
};

struct DynamicCVParams {
  bool present = false;
  bool enabled = false;
  bool targetSupported = false;
  bool compilerStatusPresent = false;
  bool compilerApplied = false;
  bool bufferInsertionOptimization = false;
  bool ubRefine = false;
  bool cubeBlockMerge = false;
  int64_t intraCacheCount = 2;
  int64_t interCacheCount = 1;
  int64_t loadCacheCount = 1;
  std::string compilerSkipReason = "not_reported";
  std::string statusSource = "ttir_static_inference";
};

enum class CVFeatureMode {
  Base,
  OrdinaryMultibuffer,
  DynamicCVLegacyMax,
  DynamicCV,
  OrdinaryMultibufferFallback,
};

struct TileMixModelConfig {
  int64_t loopControlCyclesPerSegment = 2;
};

struct TileMixDerivedFeatures {
  int64_t tileM = 0;
  int64_t tileN = 0;
  int64_t dtypeBytes = 0;
  int64_t handoffTileBytes = 0;
  int64_t handoffFeatureDim = 0;
  int64_t intermediateTileBytes = 0;
  std::string tileShapeSource = "none";
  std::string dtypeSource = "none";
  std::string handoffSource = "none";
  std::string intermediateSource = "none";
};

struct TileMixStats {
  bool used = false;
  bool valid = false;
  bool adjustmentApplied = false;
  bool cubeApplied = false;
  bool vectorApplied = false;
  int64_t confidencePercent = 0;
  int64_t adjustedCycles = 0;
  int64_t baseCycles = 0;
  int64_t boundaryCycles = 0;
  int64_t balancePenaltyCycles = 0;
  int64_t handoffReliefCycles = 0;
  int64_t workspaceReliefCycles = 0;
  int64_t netDeltaCycles = 0;
  int64_t cubeSegmentCount = 0;
  int64_t vectorSegmentCount = 0;
  int64_t cubeLoopTrip = 0;
  int64_t vectorLoopTrip = 0;
  int64_t cubeLayoutOpCount = 0;
  int64_t vectorLayoutOpCount = 0;
  int64_t cubeWorkspaceBytes = 0;
  int64_t vectorWorkspaceBytes = 0;
  int64_t cubeSubtileBytes = 0;
  int64_t vectorSubtileBytes = 0;
  int64_t cubeTargetBytes = 0;
  int64_t vectorTargetBytes = 0;
  int64_t inferredTileM = 0;
  int64_t inferredTileN = 0;
  int64_t handoffFeatureDim = 0;
  int64_t handoffDtypeBytes = 0;
  int64_t handoffTileBytes = 0;
  int64_t handoffSubtileBytes = 0;
  int64_t handoffSegmentCount = 0;
  int64_t handoffTargetBytes = 0;
  int64_t handoffNeutralBlockN = 0;
  int64_t intermediateTileBytes = 0;
  int64_t intermediateTargetBytes = 0;
  int64_t intermediateNeutralBlockM = 0;
  int64_t intermediatePressurePenaltyCycles = 0;
  int64_t loopGranularityReliefCycles = 0;
  int64_t loopMismatchPenaltyCycles = 0;
  int64_t bufferFitPenaltyCycles = 0;
  int64_t syncFrequencyPenaltyCycles = 0;
  int64_t gmDeltaCycles = 0;
  int64_t externalSyncDeltaCycles = 0;
  int64_t bufferDeltaCycles = 0;
  int64_t pipelineDeltaCycles = 0;
  int64_t scalarControlDeltaCycles = 0;
  int64_t syncOpsBefore = 0;
  int64_t syncOpsAfter = 0;
  std::string summarySource = "missing";
  std::string cubeSkipReason = "missing_pass_summary";
  std::string vectorSkipReason = "missing_pass_summary";
  std::string tileShapeSource = "none";
  std::string dtypeSource = "none";
  std::string handoffSource = "none";
  std::string intermediateSource = "none";
};

struct WorkspaceMultibufferStats {
  bool used = false;
  bool valid = false;
  bool adjustmentApplied = false;
  int64_t requestedSlots = 1;
  int64_t referenceSlots = 1;
  int64_t slotDelta = 0;
  int64_t extraSlots = 0;
  int64_t workspaceFamilyCount = 0;
  int64_t cubeToVectorFamilyCount = 0;
  int64_t vectorToCubeFamilyCount = 0;
  int64_t workspaceBytesPerSlot = 0;
  int64_t iterationCount = 0;
  int64_t cubeToVectorIterations = 0;
  int64_t vectorToCubeIterations = 0;
  int64_t cubeProducerTailCycles = 0;
  int64_t vectorProducerTailCycles = 0;
  int64_t syncPairCycles = 0;
  int64_t syncDeltaCycles = 0;
  int64_t blockingCycles = 0;
  int64_t referenceBlockingCycles = 0;
  int64_t producerWaitReliefCycles = 0;
  int64_t referenceQueuePenaltyCycles = 0;
  int64_t overlapReliefCycles = 0;
  int64_t queueDeltaCycles = 0;
  int64_t netDeltaCycles = 0;
  std::string skipReason = "none";
};

struct DynamicCVStats {
  bool used = false;
  bool eligible = false;
  bool segmentModelValid = false;
  bool compilerStatusPresent = false;
  bool compilerApplied = false;
  bool adjustmentApplied = false;
  int64_t segmentCount = 0;
  int64_t dataDependencyEdges = 0;
  int64_t segmentOrderEdges = 0;
  int64_t workItemCount = 0;
  int64_t crossCoreEdges = 0;
  int64_t intraCapacityEdges = 0;
  int64_t interCapacityEdges = 0;
  int64_t loadCapacityEdges = 0;
  int64_t intraBlockingCycles = 0;
  int64_t interBlockingCycles = 0;
  int64_t loadBlockingCycles = 0;
  int64_t iterationCount = 1;
  int64_t intraCacheCount = 2;
  int64_t interCacheCount = 1;
  int64_t loadCacheCount = 1;
  int64_t cacheBytes = 0;
  int64_t originalMakespanCycles = 0;
  int64_t transformedMakespanCycles = 0;
  int64_t offMakespanCycles = 0;
  int64_t onMakespanCycles = 0;
  int64_t offTotalCycles = 0;
  int64_t onTotalCycles = 0;
  int64_t referenceDeltaCycles = 0;
  int64_t syncCycles = 0;
  int64_t controlCycles = 0;
  int64_t netDeltaCycles = 0;
  std::string skipReason = "disabled";
  std::string statusSource = "ttir_static_inference";
};

enum class TileMixPath { None = 0, Cube = 1, Vector = 2 };

bool parsePositiveInt(llvm::StringRef value, int64_t &result) {
  value = value.trim();
  if (value.empty())
    return false;
  int64_t parsed = 0;
  if (value.getAsInteger(10, parsed))
    return false;
  if (parsed <= 0)
    return false;
  result = parsed;
  return true;
}

bool parseBool(llvm::StringRef value, bool &result) {
  value = value.trim();
  if (value.equals_insensitive("1") || value.equals_insensitive("true") ||
      value.equals_insensitive("on")) {
    result = true;
    return true;
  }
  if (value.equals_insensitive("0") || value.equals_insensitive("false") ||
      value.equals_insensitive("off")) {
    result = false;
    return true;
  }
  return false;
}

bool parseNonNegativeInt(llvm::StringRef value, int64_t &result) {
  value = value.trim();
  int64_t parsed = 0;
  if (value.empty() || value.getAsInteger(10, parsed) || parsed < 0)
    return false;
  result = parsed;
  return true;
}

bool parseBool01(llvm::StringRef value, bool &result) {
  int64_t parsed = 0;
  if (!parseNonNegativeInt(value, parsed) || parsed > 1)
    return false;
  result = parsed == 1;
  return true;
}

TileMixParams parseTileMixParams(llvm::StringRef compileParamsStr) {
  TileMixParams params;
  llvm::SmallVector<llvm::StringRef, 8> items;
  compileParamsStr.split(items, ',', -1, false);
  for (llvm::StringRef item : items) {
    item = item.trim();
    if (item.empty())
      continue;
    std::pair<llvm::StringRef, llvm::StringRef> kv = item.split('=');
    llvm::StringRef key = kv.first.trim();
    llvm::StringRef value = kv.second.trim();
    int64_t parsed = 0;
    if (key == "tile_mix_vector_loop" && parsePositiveInt(value, parsed)) {
      params.vectorLoop = parsed;
      params.hasAny = true;
    } else if (key == "tile_mix_cube_loop" && parsePositiveInt(value, parsed)) {
      params.cubeLoop = parsed;
      params.hasAny = true;
    } else if (key == "tile_mix_summary_source" && !value.empty()) {
      params.summarySource = value.str();
      params.summaryPresent = true;
    } else if (key == "tile_mix_summary_valid") {
      params.summaryPresent = true;
      parseBool01(value, params.summaryValid);
    } else if (key == "tile_mix_vector_applied") {
      parseBool01(value, params.vectorApplied);
    } else if (key == "tile_mix_cube_applied") {
      parseBool01(value, params.cubeApplied);
    } else if (key == "tile_mix_vector_segments" &&
               parsePositiveInt(value, parsed)) {
      params.vectorSegments = parsed;
    } else if (key == "tile_mix_cube_segments" &&
               parsePositiveInt(value, parsed)) {
      params.cubeSegments = parsed;
    } else if (key == "tile_mix_sync_ops_before" &&
               parseNonNegativeInt(value, parsed)) {
      params.syncOpsBefore = parsed;
    } else if (key == "tile_mix_sync_ops_after" &&
               parseNonNegativeInt(value, parsed)) {
      params.syncOpsAfter = parsed;
    } else if (key == "tile_mix_vector_skip_reason" && !value.empty()) {
      params.vectorSkipReason = value.str();
    } else if (key == "tile_mix_cube_skip_reason" && !value.empty()) {
      params.cubeSkipReason = value.str();
    }
  }
  return params;
}

WorkspaceMultibufferParams
parseWorkspaceMultibufferParams(llvm::StringRef compileParamsStr) {
  WorkspaceMultibufferParams params;
  llvm::SmallVector<llvm::StringRef, 8> items;
  compileParamsStr.split(items, ',', -1, false);
  for (llvm::StringRef item : items) {
    item = item.trim();
    if (item.empty())
      continue;
    std::pair<llvm::StringRef, llvm::StringRef> kv = item.split('=');
    llvm::StringRef key = kv.first.trim();
    llvm::StringRef value = kv.second.trim();
    int64_t parsed = 0;
    bool parsedBool = false;
    if (key == "set_workspace_multibuffer" &&
        parsePositiveInt(value, parsed)) {
      params.requestedSlots = parsed;
      params.present = true;
      params.workspacePresent = true;
    } else if (key == "multibuffer" && parseBool(value, parsedBool)) {
      params.localAutoPresent = true;
      params.localAutoEnabled = parsedBool;
      params.present = true;
    } else if (key == "num_stages" && parsePositiveInt(value, parsed)) {
      params.numStages = parsed;
      params.localAutoPresent = true;
      params.localAutoEnabled = parsed > 1;
      params.present = true;
    } else if (key == "limit_auto_multi_buffer_only_for_local_buffer" &&
               parseBool(value, parsedBool)) {
      params.workspaceOnlyLocal = parsedBool;
    } else if (key == "limit_auto_multi_buffer_of_local_buffer" &&
               !value.empty()) {
      params.localScope = value.str();
    } else if (key == "limit_auto_multi_buffer_buffer" && !value.empty()) {
      params.limitedBuffer = value.str();
    }
  }
  return params;
}

DynamicCVParams parseDynamicCVParams(llvm::StringRef compileParamsStr) {
  DynamicCVParams params;
  llvm::SmallVector<llvm::StringRef, 16> items;
  compileParamsStr.split(items, ',', -1, false);
  for (llvm::StringRef item : items) {
    item = item.trim();
    if (item.empty())
      continue;
    auto kv = item.split('=');
    llvm::StringRef key = kv.first.trim();
    llvm::StringRef value = kv.second.trim();
    int64_t parsed = 0;
    bool parsedBool = false;
    if (key == "enable_dynamic_cv_pipeline" && parseBool(value, parsedBool)) {
      params.present = true;
      params.enabled = parsedBool;
    } else if (key == "dynamic_cv_applied" &&
               parseBool(value, parsedBool)) {
      params.compilerStatusPresent = true;
      params.compilerApplied = parsedBool;
    } else if (key == "dynamic_cv_skip_reason" && !value.empty()) {
      params.compilerSkipReason = value.str();
    } else if (key == "dynamic_cv_status_source" && !value.empty()) {
      params.statusSource = value.str();
    } else if (key == "compile_on_910_95" &&
               parseBool(value, parsedBool)) {
      params.targetSupported = parsedBool;
    } else if (key == "intra_cache_num" && parsePositiveInt(value, parsed)) {
      params.intraCacheCount = parsed;
    } else if (key == "inter_cache_num" && parsePositiveInt(value, parsed)) {
      params.interCacheCount = parsed;
    } else if (key == "load_cache_num" && parsePositiveInt(value, parsed)) {
      params.loadCacheCount = parsed;
    } else if (key == "enable_buffer_insert_optimization" &&
               parseBool(value, parsedBool)) {
      params.bufferInsertionOptimization = parsedBool;
    } else if (key == "enable_ub_refine_opt" &&
               parseBool(value, parsedBool)) {
      params.ubRefine = parsedBool;
    } else if (key == "enable_cube_block_merge" &&
               parseBool(value, parsedBool)) {
      params.cubeBlockMerge = parsedBool;
    }
  }
  return params;
}

llvm::StringRef cvFeatureModeName(CVFeatureMode mode) {
  switch (mode) {
  case CVFeatureMode::Base:
    return "base";
  case CVFeatureMode::OrdinaryMultibuffer:
    return "ordinary_multibuffer";
  case CVFeatureMode::DynamicCVLegacyMax:
    return "dynamic_cv_legacy_max";
  case CVFeatureMode::DynamicCV:
    return "dynamic_cv";
  case CVFeatureMode::OrdinaryMultibufferFallback:
    return "ordinary_multibuffer_fallback";
  }
  llvm_unreachable("unknown CV feature mode");
}

bool isTileMixDagModelEnabled() {
  const char *mode = std::getenv("ASCEND_COSTMODEL_TILE_MIX_MODEL");
  if (!mode)
    return true;
  llvm::StringRef value(mode);
  return !(value.equals_insensitive("0") || value.equals_insensitive("off") ||
           value.equals_insensitive("false") || value.equals_insensitive("none"));
}

bool isWorkspaceMultibufferModelEnabled() {
  const char *mode =
      std::getenv("ASCEND_COSTMODEL_WORKSPACE_MULTIBUFFER_MODEL");
  // Preserve existing Off/On report behavior when the new independent switch
  // is not set. Callers can explicitly control the peer feature with the new
  // variable.
  if (!mode)
    return isTileMixDagModelEnabled();
  llvm::StringRef value(mode);
  return !(value.equals_insensitive("0") || value.equals_insensitive("off") ||
           value.equals_insensitive("false") || value.equals_insensitive("none"));
}

int64_t ceilDiv(int64_t value, int64_t divisor) {
  if (value <= 0 || divisor <= 0)
    return 0;
  return (value + divisor - 1) / divisor;
}

int64_t overflowTransferCycles(int64_t chunkBytes, int64_t targetBytes,
                               int64_t segmentCount, int64_t totalBytes,
                               int64_t transferCycles) {
  if (chunkBytes <= 0 || targetBytes <= 0 || segmentCount <= 0 ||
      totalBytes <= 0 || transferCycles <= 0)
    return 0;
  int64_t overflowPerSegment = std::max<int64_t>(0, chunkBytes - targetBytes);
  if (overflowPerSegment == 0)
    return 0;
  long double overflowBytes = static_cast<long double>(overflowPerSegment) *
                              static_cast<long double>(segmentCount);
  long double cycles = overflowBytes * static_cast<long double>(transferCycles) /
                       static_cast<long double>(totalBytes);
  return std::max<int64_t>(0, static_cast<int64_t>(std::ceil(cycles)));
}

TileMixModelConfig getTileMixModelConfig(const HardwareConfig &config) {
  TileMixModelConfig model;
  model.loopControlCyclesPerSegment = std::max<int64_t>(
      0, config.getCostModelIntParam("tilemix_loop_control_cycles_per_segment",
                                     model.loopControlCyclesPerSegment));
  return model;
}

int64_t minPositiveLocalBytes(int64_t lhs, int64_t rhs) {
  if (lhs <= 0)
    return rhs;
  if (rhs <= 0)
    return lhs;
  return std::min(lhs, rhs);
}

int64_t getElementByteWidth(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    type = shaped.getElementType();
  if (type.isIntOrFloat())
    return (type.getIntOrFloatBitWidth() + 7) / 8;
  if (type.isIndex())
    return 8;
  return 0;
}

int64_t getStaticShapedTypeBytes(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return 0;
  int64_t elemBytes = getElementByteWidth(shaped.getElementType());
  if (elemBytes <= 0)
    return 0;
  int64_t elements = 1;
  for (int64_t dim : shaped.getShape())
    elements *= dim;
  return elements * elemBytes;
}

void updateTopTwoLoads(int64_t bytes, int64_t dtypeBytes,
                       int64_t &top1Bytes, int64_t &top1DtypeBytes,
                       int64_t &top2Bytes, int64_t &top2DtypeBytes) {
  if (bytes <= 0)
    return;
  if (bytes > top1Bytes) {
    top2Bytes = top1Bytes;
    top2DtypeBytes = top1DtypeBytes;
    top1Bytes = bytes;
    top1DtypeBytes = dtypeBytes;
    return;
  }
  if (bytes > top2Bytes) {
    top2Bytes = bytes;
    top2DtypeBytes = dtypeBytes;
  }
}

TileMixDerivedFeatures inferTileMixDerivedFeatures(
    const PipelineScheduler &scheduler) {
  TileMixDerivedFeatures features;
  int64_t topVectorLoadBytes = 0;
  int64_t topVectorLoadDtypeBytes = 0;
  int64_t secondVectorLoadBytes = 0;
  int64_t secondVectorLoadDtypeBytes = 0;
  int64_t maxBoundaryBytes = 0;
  int64_t maxBoundaryDtypeBytes = 0;
  int64_t maxMatmulResultBytes = 0;

  for (const auto &op : scheduler.getAllOps()) {
    if (!op.mlirOp)
      continue;

    if (auto matmulOp = dyn_cast<MatmulOp>(op.mlirOp)) {
      if (matmulOp.getM() > 0 && matmulOp.getN() > 0) {
        features.tileM = std::max<int64_t>(features.tileM, matmulOp.getM());
        features.tileN = std::max<int64_t>(features.tileN, matmulOp.getN());
        features.tileShapeSource = "matmul_attrs";
      }
      int64_t lhsDtypeBytes = getElementByteWidth(matmulOp.getLhs().getType());
      if (lhsDtypeBytes > 0 && features.dtypeBytes <= 0) {
        features.dtypeBytes = lhsDtypeBytes;
        features.dtypeSource = "matmul_lhs_type";
      }
      int64_t resultBytes =
          getStaticShapedTypeBytes(matmulOp->getResult(0).getType());
      if (resultBytes > maxMatmulResultBytes) {
        maxMatmulResultBytes = resultBytes;
        features.intermediateTileBytes = resultBytes;
        features.intermediateSource = "matmul_result_type";
      }
      continue;
    }

    if (auto loadOp = dyn_cast<VectorLoadOp>(op.mlirOp)) {
      int64_t bytes = std::max<int64_t>(loadOp.getTransferBytes(), 0);
      int64_t dtypeBytes =
          getElementByteWidth(loadOp->getResult(0).getType());
      Operation *sourceDef = loadOp.getSource().getDefiningOp();
      if (sourceDef && isa<MatmulOp>(sourceDef)) {
        if (bytes > maxBoundaryBytes) {
          maxBoundaryBytes = bytes;
          maxBoundaryDtypeBytes = dtypeBytes;
        }
      } else {
        updateTopTwoLoads(bytes, dtypeBytes, topVectorLoadBytes,
                          topVectorLoadDtypeBytes, secondVectorLoadBytes,
                          secondVectorLoadDtypeBytes);
      }
      continue;
    }

    if (auto storeOp = dyn_cast<CubeStoreOp>(op.mlirOp)) {
      int64_t bytes = std::max<int64_t>(storeOp.getTransferBytes(), 0);
      int64_t dtypeBytes = getElementByteWidth(storeOp.getData().getType());
      if (bytes > maxBoundaryBytes) {
        maxBoundaryBytes = bytes;
        maxBoundaryDtypeBytes = dtypeBytes;
      }
    }
  }

  if (topVectorLoadBytes > 0) {
    features.handoffTileBytes = topVectorLoadBytes + secondVectorLoadBytes;
    features.handoffSource =
        secondVectorLoadBytes > 0 ? "vector_load_top2" : "vector_load_top1";
    if (topVectorLoadDtypeBytes > 0) {
      features.dtypeBytes = topVectorLoadDtypeBytes;
      features.dtypeSource = "vector_load_type";
    } else if (secondVectorLoadDtypeBytes > 0 && features.dtypeBytes <= 0) {
      features.dtypeBytes = secondVectorLoadDtypeBytes;
      features.dtypeSource = "vector_load_type";
    }
  } else if (maxBoundaryBytes > 0) {
    features.handoffTileBytes = maxBoundaryBytes;
    features.handoffSource = "cube_vector_boundary";
    if (maxBoundaryDtypeBytes > 0 && features.dtypeBytes <= 0) {
      features.dtypeBytes = maxBoundaryDtypeBytes;
      features.dtypeSource = "boundary_type";
    }
  }

  if (features.intermediateTileBytes <= 0 && maxBoundaryBytes > 0) {
    features.intermediateTileBytes = maxBoundaryBytes;
    features.intermediateSource = "cube_vector_boundary";
  }

  if (features.handoffTileBytes > 0 && features.tileN > 0 &&
      features.dtypeBytes > 0) {
    int64_t bytesPerTileN = features.tileN * features.dtypeBytes;
    if (bytesPerTileN > 0)
      features.handoffFeatureDim =
          std::max<int64_t>(1, features.handoffTileBytes / bytesPerTileN);
  }

  return features;
}

int64_t tileMixExternalSyncDeltaCycles(const TileMixParams &params,
                                       const HardwareConfig &config) {
  // The pass summary counts concrete sync operations before and after the
  // transformation. Convert that marginal count to cycles; do not scale the
  // whole roofline by an empirical percentage.
  int64_t deltaOps = params.syncOpsAfter - params.syncOpsBefore;
  int64_t cyclesPerOp = std::max<int64_t>(
      0, (config.getSyncOpCycles("set_flag", 1) +
          config.getSyncOpCycles("wait_flag", 2) + 1) /
             2);
  return deltaOps * cyclesPerOp;
}

TileMixPath getTileMixPath(HWUnit unit) {
  switch (unit) {
    case HWUnit::Cube:
    case HWUnit::CubeMTE2:
    case HWUnit::FixPipe:
      return TileMixPath::Cube;
    case HWUnit::Vector:
    case HWUnit::VecMTE2:
    case HWUnit::MTE3:
      return TileMixPath::Vector;
    default:
      return TileMixPath::None;
  }
}

bool isTileMixLayoutOp(HWUnit unit) {
  return unit == HWUnit::CubeMTE2 || unit == HWUnit::FixPipe ||
         unit == HWUnit::VecMTE2 || unit == HWUnit::MTE3;
}

struct TileMixSideDecision {
  bool known = false;
  bool applied = false;
  int64_t segments = 1;
  std::string skipReason = "unknown";
};

TileMixSideDecision inferTileMixSideFromTTIR(
    TileMixPath side, int64_t requestedSegments, int64_t pathCycles,
    int64_t layoutOps, int64_t workspaceBytes,
    int64_t localBufferBytes, int64_t vectorAlignmentBytes) {
  TileMixSideDecision decision;
  if (requestedSegments <= 1) {
    decision.known = true;
    decision.skipReason = "requested_loop_le_one";
    return decision;
  }
  if (pathCycles <= 0) {
    decision.known = true;
    decision.skipReason = side == TileMixPath::Cube ? "no_cube_path"
                                                    : "no_vector_path";
    return decision;
  }
  if (layoutOps <= 0) {
    decision.known = true;
    decision.skipReason = side == TileMixPath::Cube ? "no_cube_copyout"
                                                    : "no_vector_copyout";
    return decision;
  }
  if (workspaceBytes <= 0 || localBufferBytes <= 0) {
    decision.skipReason = side == TileMixPath::Cube
                              ? "unknown_cube_copyout_bytes"
                              : "unknown_vector_copyout_bytes";
    return decision;
  }

  // tile-mix-*-loop is the target trip count of a new inner sub-tile loop.
  // It splits one existing Cube/Vector loop iteration; it is not capped by
  // the source TTIR loop trip count. In particular, a source loop with trip
  // count one can still be split into target trip count 2/4 when CopyOut and
  // buffer evidence satisfy the pass constraints.
  decision.segments = requestedSegments;

  if (side == TileMixPath::Cube && workspaceBytes < localBufferBytes) {
    // principle.docx: Cube does not tile when the pre-tile chunk already fits
    // the total L0C capacity.
    decision.known = true;
    decision.segments = 1;
    decision.skipReason = "cube_tile_fits_l0c";
    return decision;
  }
  if (side == TileMixPath::Vector &&
      ceilDiv(workspaceBytes, decision.segments) < vectorAlignmentBytes) {
    // principle.docx: Vector does not tile when its post-tile chunk is below
    // one UB/vector alignment quantum.
    decision.known = true;
    decision.segments = 1;
    decision.skipReason = "vector_subtile_below_alignment";
    return decision;
  }

  decision.known = true;
  decision.applied = true;
  decision.skipReason = "none";
  return decision;
}

struct WorkspaceMultibufferEvidence {
  int64_t familyCount = 0;
  int64_t bytesPerSlot = 0;
  int64_t iterationCount = 0;
  int64_t cubeToVectorFamilyCount = 0;
  int64_t vectorToCubeFamilyCount = 0;
  int64_t cubeToVectorIterations = 0;
  int64_t vectorToCubeIterations = 0;
  int64_t cubeProducerTailCycles = 0;
  int64_t vectorProducerTailCycles = 0;
};

struct TransferFamilyEndpoint {
  int64_t bytes = 0;
  int64_t iterations = 0;
  int64_t endCycle = 0;
};

std::optional<int64_t> getBufferFamilyId(Operation *op) {
  if (!op)
    return std::nullopt;
  auto attr = op->getAttrOfType<IntegerAttr>("ascend.buffer_family_id");
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

bool isDynamicCVSegmentDagModelEnabled() {
  const char *mode =
      std::getenv("ASCEND_COSTMODEL_DYNAMIC_CV_SEGMENT_DAG_MODEL");
  // Compatibility gate: uploading the new binary must preserve the previous
  // max(Cube path, Vector path) result until the caller explicitly opts in.
  if (!mode)
    return false;
  llvm::StringRef value(mode);
  return !(value.equals_insensitive("0") || value.equals_insensitive("off") ||
           value.equals_insensitive("false") || value.equals_insensitive("none"));
}

WorkspaceMultibufferEvidence inferWorkspaceMultibufferEvidence(
    const PipelineScheduler &scheduler) {
  std::map<int64_t, TransferFamilyEndpoint> cubeLoads;
  std::map<int64_t, TransferFamilyEndpoint> cubeStores;
  std::map<int64_t, TransferFamilyEndpoint> vectorLoads;
  std::map<int64_t, TransferFamilyEndpoint> vectorStores;
  int64_t cubePathEndCycle = 0;
  int64_t vectorPathEndCycle = 0;

  auto recordTransfer = [](std::map<int64_t, TransferFamilyEndpoint> &map,
                           Operation *op, int64_t bytes, int64_t iterations,
                           int64_t endCycle) {
    auto familyId = getBufferFamilyId(op);
    if (!familyId || bytes <= 0)
      return;
    auto &endpoint = map[*familyId];
    endpoint.bytes = endpoint.bytes == 0 ? bytes : std::min(endpoint.bytes, bytes);
    endpoint.iterations =
        std::max(endpoint.iterations, std::max<int64_t>(iterations, 1));
    endpoint.endCycle = std::max(endpoint.endCycle, endCycle);
  };

  for (const auto &op : scheduler.getAllOps()) {
    TileMixPath path = getTileMixPath(op.hwUnit);
    if (path == TileMixPath::Cube)
      cubePathEndCycle = std::max(cubePathEndCycle, op.endCycle);
    else if (path == TileMixPath::Vector)
      vectorPathEndCycle = std::max(vectorPathEndCycle, op.endCycle);
    if (!op.mlirOp)
      continue;
    if (auto loadOp = dyn_cast<CubeLoadOp>(op.mlirOp)) {
      recordTransfer(cubeLoads, op.mlirOp, loadOp.getTransferBytes(),
                     op.loopMultiplier, op.endCycle);
    } else if (auto storeOp = dyn_cast<CubeStoreOp>(op.mlirOp)) {
      recordTransfer(cubeStores, op.mlirOp, storeOp.getTransferBytes(),
                     op.loopMultiplier, op.endCycle);
    } else if (auto loadOp = dyn_cast<VectorLoadOp>(op.mlirOp)) {
      recordTransfer(vectorLoads, op.mlirOp, loadOp.getTransferBytes(),
                     op.loopMultiplier, op.endCycle);
    } else if (auto storeOp = dyn_cast<VectorStoreOp>(op.mlirOp)) {
      recordTransfer(vectorStores, op.mlirOp, storeOp.getTransferBytes(),
                     op.loopMultiplier, op.endCycle);
    }
  }

  WorkspaceMultibufferEvidence evidence;
  auto addMatchedFamilies = [&evidence](
                                 const std::map<int64_t, TransferFamilyEndpoint> &producers,
                                 const std::map<int64_t, TransferFamilyEndpoint> &consumers,
                                 int64_t producerPathEnd,
                                 int64_t &directionFamilyCount,
                                 int64_t &directionIterations,
                                 int64_t &producerTailCycles) {
    for (const auto &[familyId, producer] : producers) {
      auto consumer = consumers.find(familyId);
      if (consumer == consumers.end())
        continue;
      if (producer.bytes <= 0 || consumer->second.bytes <= 0 ||
          producer.bytes != consumer->second.bytes)
        continue;
      ++evidence.familyCount;
      ++directionFamilyCount;
      evidence.bytesPerSlot += producer.bytes;
      int64_t commonIterations =
          std::min(producer.iterations, consumer->second.iterations);
      if (evidence.iterationCount == 0)
        evidence.iterationCount = commonIterations;
      else
        evidence.iterationCount =
            std::min(evidence.iterationCount, commonIterations);
      if (directionIterations == 0)
        directionIterations = commonIterations;
      else
        directionIterations = std::min(directionIterations, commonIterations);
      int64_t tail =
          std::max<int64_t>(0, producerPathEnd - producer.endCycle);
      if (directionFamilyCount == 1)
        producerTailCycles = tail;
      else
        producerTailCycles = std::min(producerTailCycles, tail);
    }
  };
  // Direction is part of the family identity: equally sized Cube->Vector and
  // Vector->Cube workspaces have independent version/synchronization state.
  addMatchedFamilies(
      cubeStores, vectorLoads, cubePathEndCycle,
      evidence.cubeToVectorFamilyCount,
      evidence.cubeToVectorIterations, evidence.cubeProducerTailCycles);
  addMatchedFamilies(
      vectorStores, cubeLoads, vectorPathEndCycle,
      evidence.vectorToCubeFamilyCount,
      evidence.vectorToCubeIterations, evidence.vectorProducerTailCycles);
  return evidence;
}

struct BoundedBufferSchedule {
  int64_t makespanCycles = 0;
  int64_t producerBlockingCycles = 0;
};

BoundedBufferSchedule scheduleFiniteWorkspaceBuffer(
    int64_t producerPathCycles, int64_t consumerPathCycles,
    int64_t iterationCount, int64_t bufferSlots,
    int64_t producerTailCycles = 0) {
  BoundedBufferSchedule result;
  if (producerPathCycles <= 0 || consumerPathCycles <= 0 ||
      iterationCount <= 0 || bufferSlots <= 0)
    return result;

  // Distribute the integer path cycles over concrete TTIR iterations. This is
  // an exact deterministic finite FIFO schedule for the static information
  // available at TTIR: producer i cannot reuse slot i%B before consumer i-B
  // releases it, and consumer i cannot start before producer i completes.
  auto segmentCycles = [](int64_t total, int64_t count, int64_t index) {
    int64_t base = total / count;
    int64_t remainder = total % count;
    return std::max<int64_t>(1, base + (index < remainder ? 1 : 0));
  };

  producerTailCycles = std::max<int64_t>(
      0, std::min(producerTailCycles, producerPathCycles - 1));
  int64_t producerHandoffCycles = producerPathCycles - producerTailCycles;
  std::vector<int64_t> consumerDone(iterationCount, 0);
  int64_t producerCursor = 0;
  int64_t consumerCursor = 0;
  for (int64_t i = 0; i < iterationCount; ++i) {
    int64_t slotRelease =
        i >= bufferSlots ? consumerDone[i - bufferSlots] : 0;
    int64_t producerStart = std::max(producerCursor, slotRelease);
    result.producerBlockingCycles += producerStart - producerCursor;
    int64_t producerDone =
        producerStart + segmentCycles(producerHandoffCycles, iterationCount, i);
    producerCursor = producerDone;

    int64_t consumerStart = std::max(consumerCursor, producerDone);
    consumerDone[i] =
        consumerStart + segmentCycles(consumerPathCycles, iterationCount, i);
    consumerCursor = consumerDone[i];
  }
  result.makespanCycles =
      std::max(producerCursor + producerTailCycles, consumerCursor);
  return result;
}

WorkspaceMultibufferStats estimateWorkspaceMultibuffer(
    const WorkspaceMultibufferParams &params,
    const PipelineScheduler &scheduler, const HardwareConfig &config,
    int64_t cubePathCycles, int64_t vectorPathCycles,
    int64_t baseRooflineTotalCycles) {
  WorkspaceMultibufferStats stats;
  if (!params.present || !isWorkspaceMultibufferModelEnabled())
    return stats;

  stats.used = true;
  if (!params.workspacePresent || params.workspaceOnlyLocal) {
    // This pass can prove cross-core workspace handoffs, but optimized TTIR no
    // longer exposes enough local-buffer lifetime information to charge or
    // credit local-only multibuffering safely. Keep that branch fail-closed.
    stats.valid = true;
    stats.skipReason = params.workspaceOnlyLocal
                           ? "local_buffer_only_not_proven_in_ttir"
                           : "local_multibuffer_lifetime_not_proven_in_ttir";
    return stats;
  }
  stats.requestedSlots = std::max<int64_t>(1, params.requestedSlots);
  WorkspaceMultibufferEvidence evidence =
      inferWorkspaceMultibufferEvidence(scheduler);
  stats.workspaceFamilyCount = evidence.familyCount;
  stats.cubeToVectorFamilyCount = evidence.cubeToVectorFamilyCount;
  stats.vectorToCubeFamilyCount = evidence.vectorToCubeFamilyCount;
  stats.workspaceBytesPerSlot = evidence.bytesPerSlot;
  stats.iterationCount = evidence.iterationCount;
  stats.cubeToVectorIterations = evidence.cubeToVectorIterations;
  stats.vectorToCubeIterations = evidence.vectorToCubeIterations;
  stats.cubeProducerTailCycles = evidence.cubeProducerTailCycles;
  stats.vectorProducerTailCycles = evidence.vectorProducerTailCycles;

  // One physical version is the feature-off reference. Extra versions do not
  // create extra synchronization events; they only delay slot reuse and can
  // therefore remove producer waits in the finite FIFO schedule.
  stats.referenceSlots = 1;
  stats.slotDelta = stats.requestedSlots - stats.referenceSlots;
  stats.extraSlots = std::max<int64_t>(0, stats.slotDelta);
  stats.syncPairCycles = config.getSyncOpCycles("set_flag", 1) +
                         config.getSyncOpCycles("wait_flag", 2);
  if (stats.workspaceFamilyCount > 0) {
    // One set/wait handoff is required for each produced version. Increasing
    // the number of reusable slots changes waiting, not token count.
    stats.syncDeltaCycles = stats.workspaceFamilyCount *
                            std::max<int64_t>(stats.iterationCount, 1) *
                            stats.syncPairCycles;

    auto accumulateDirection = [&](int64_t familyCount, int64_t iterations,
                                   int64_t producerCycles,
                                   int64_t consumerCycles,
                                   int64_t producerTailCycles) {
      if (familyCount <= 0 || iterations <= 0)
        return;
      BoundedBufferSchedule requested = scheduleFiniteWorkspaceBuffer(
          producerCycles, consumerCycles, iterations, stats.requestedSlots,
          producerTailCycles);
      BoundedBufferSchedule reference = scheduleFiniteWorkspaceBuffer(
          producerCycles, consumerCycles, iterations, stats.referenceSlots,
          producerTailCycles);
      stats.blockingCycles =
          std::max(stats.blockingCycles, requested.producerBlockingCycles);
      stats.referenceBlockingCycles = std::max(
          stats.referenceBlockingCycles, reference.producerBlockingCycles);
      stats.queueDeltaCycles = std::max(stats.queueDeltaCycles,
          std::max<int64_t>(0, requested.makespanCycles -
                                   baseRooflineTotalCycles));
      stats.referenceQueuePenaltyCycles = std::max(
          stats.referenceQueuePenaltyCycles,
          std::max<int64_t>(0, reference.makespanCycles -
                                   baseRooflineTotalCycles));
    };
    accumulateDirection(stats.cubeToVectorFamilyCount,
                        stats.cubeToVectorIterations, cubePathCycles,
                        vectorPathCycles, stats.cubeProducerTailCycles);
    accumulateDirection(stats.vectorToCubeFamilyCount,
                        stats.vectorToCubeIterations, vectorPathCycles,
                        cubePathCycles, stats.vectorProducerTailCycles);
    stats.overlapReliefCycles = std::max<int64_t>(
        0, stats.referenceQueuePenaltyCycles - stats.queueDeltaCycles);
    stats.producerWaitReliefCycles = std::max<int64_t>(
        0, stats.referenceBlockingCycles - stats.blockingCycles);
  }

  stats.netDeltaCycles = stats.syncDeltaCycles + stats.queueDeltaCycles;
  stats.valid = true;
  stats.adjustmentApplied = stats.netDeltaCycles != 0;
  return stats;
}

struct DynamicCVSegment {
  int64_t id = -1;
  TileMixPath path = TileMixPath::None;
  HWUnit resource = HWUnit::Scalar;
  int64_t durationCycles = 0;
  int64_t tripCount = 1;
  std::set<int64_t> intraDependencies;
  std::set<int64_t> interDependencies;
  std::set<int64_t> loadDependencies;
  std::set<int64_t> orderDependencies;
};

struct DynamicCVSegmentGraph {
  std::vector<DynamicCVSegment> segments;
  int64_t dataDependencyEdges = 0;
  int64_t crossPathOrderEdges = 0;
  int64_t iterationCount = 1;
};

struct DynamicCVExpandedNode {
  TileMixPath path = TileMixPath::None;
  HWUnit resource = HWUnit::Scalar;
  int64_t durationCycles = 0;
  std::set<int64_t> dependencies;
  std::set<int64_t> intraCapacityDependencies;
  std::set<int64_t> interCapacityDependencies;
  std::set<int64_t> loadCapacityDependencies;
};

struct DynamicCVSegmentSchedule {
  bool valid = false;
  int64_t makespanCycles = 0;
  int64_t nodeCount = 0;
  int64_t intraCapacityEdges = 0;
  int64_t interCapacityEdges = 0;
  int64_t loadCapacityEdges = 0;
  int64_t intraBlockingCycles = 0;
  int64_t interBlockingCycles = 0;
  int64_t loadBlockingCycles = 0;
};

enum class DynamicCVQueue { Intra, Inter, Load };

struct DynamicCVCacheDepths {
  int64_t intra = 1;
  int64_t inter = 1;
  int64_t load = 1;
};

bool isDynamicCVLoadResource(HWUnit unit) {
  return unit == HWUnit::CubeMTE2 || unit == HWUnit::VecMTE2;
}

DynamicCVQueue classifyDynamicCVQueue(const DynamicCVSegment &producer,
                                      const DynamicCVSegment &consumer) {
  if (producer.path != consumer.path)
    return DynamicCVQueue::Inter;
  if (isDynamicCVLoadResource(producer.resource))
    return DynamicCVQueue::Load;
  return DynamicCVQueue::Intra;
}

bool addDynamicCVDataDependency(DynamicCVSegmentGraph &graph,
                                int64_t producerId, int64_t consumerId,
                                DynamicCVQueue queue) {
  DynamicCVSegment &consumer = graph.segments[consumerId];
  std::set<int64_t> *dependencies = nullptr;
  switch (queue) {
  case DynamicCVQueue::Intra:
    dependencies = &consumer.intraDependencies;
    break;
  case DynamicCVQueue::Inter:
    dependencies = &consumer.interDependencies;
    break;
  case DynamicCVQueue::Load:
    dependencies = &consumer.loadDependencies;
    break;
  }
  if (!dependencies->insert(producerId).second)
    return false;
  ++graph.dataDependencyEdges;
  return true;
}

DynamicCVSegmentGraph
buildDynamicCVSegmentGraph(const PipelineScheduler &scheduler) {
  DynamicCVSegmentGraph graph;
  std::map<int64_t, int64_t> opToSegment;
  int64_t currentSegment = -1;

  // A segment is a maximal contiguous run on one CV hardware resource with one
  // static trip count.  Keeping MTE, compute, and store resources separate is
  // required for load/compute overlap and for assigning each dependency to the
  // correct bounded Dynamic-CV queue.
  for (const auto &op : scheduler.getAllOps()) {
    TileMixPath path = getTileMixPath(op.hwUnit);
    if (path == TileMixPath::None) {
      currentSegment = -1;
      continue;
    }
    int64_t tripCount = std::max<int64_t>(1, op.loopMultiplier);
    if (currentSegment < 0 ||
        graph.segments[currentSegment].path != path ||
        graph.segments[currentSegment].resource != op.hwUnit ||
        graph.segments[currentSegment].tripCount != tripCount) {
      DynamicCVSegment segment;
      segment.id = graph.segments.size();
      segment.path = path;
      segment.resource = op.hwUnit;
      segment.tripCount = tripCount;
      graph.segments.push_back(segment);
      currentSegment = segment.id;
    }
    DynamicCVSegment &segment = graph.segments[currentSegment];
    segment.durationCycles += std::max<int64_t>(1, op.duration);
    graph.iterationCount = std::max(graph.iterationCount, tripCount);
    opToSegment[op.opId] = currentSegment;
  }

  // Lift actual SSA producer/consumer edges from PipelineOp to segments.
  for (const auto &op : scheduler.getAllOps()) {
    auto targetIt = opToSegment.find(op.opId);
    if (targetIt == opToSegment.end())
      continue;
    DynamicCVSegment &target = graph.segments[targetIt->second];
    for (int64_t depOpId : op.dependsOn) {
      auto sourceIt = opToSegment.find(depOpId);
      if (sourceIt == opToSegment.end() || sourceIt->second == target.id)
        continue;
      const DynamicCVSegment &source = graph.segments[sourceIt->second];
      addDynamicCVDataDependency(
          graph, source.id, target.id, classifyDynamicCVQueue(source, target));
    }
  }
  // The TTIR-to-model conversion materializes cross-core transfers through
  // HBM placeholders, so their producer/consumer relationship is not always
  // an SSA edge anymore. `ascend.buffer_family_id` preserves that logical
  // identity; lift matched Store->Load families into real segment DAG edges.
  std::map<int64_t, int64_t> cubeStores;
  std::map<int64_t, int64_t> vectorStores;
  std::map<int64_t, int64_t> cubeLoads;
  std::map<int64_t, int64_t> vectorLoads;
  for (const auto &op : scheduler.getAllOps()) {
    auto segmentIt = opToSegment.find(op.opId);
    auto familyId = getBufferFamilyId(op.mlirOp);
    if (segmentIt == opToSegment.end() || !familyId)
      continue;
    if (isa<CubeStoreOp>(op.mlirOp))
      cubeStores[*familyId] = segmentIt->second;
    else if (isa<VectorStoreOp>(op.mlirOp))
      vectorStores[*familyId] = segmentIt->second;
    else if (isa<CubeLoadOp>(op.mlirOp))
      cubeLoads[*familyId] = segmentIt->second;
    else if (isa<VectorLoadOp>(op.mlirOp))
      vectorLoads[*familyId] = segmentIt->second;
  }
  auto addFamilyEdges = [&](const std::map<int64_t, int64_t> &producers,
                            const std::map<int64_t, int64_t> &consumers) {
    for (const auto &[familyId, producerSegment] : producers) {
      auto consumer = consumers.find(familyId);
      if (consumer == consumers.end() ||
          consumer->second == producerSegment)
        continue;
      addDynamicCVDataDependency(graph, producerSegment, consumer->second,
                                 DynamicCVQueue::Inter);
    }
  };
  addFamilyEdges(vectorStores, cubeLoads);
  addFamilyEdges(cubeStores, vectorLoads);
  // Consecutive Cube/Vector segments are distinct work items in the original
  // control order. Keep this edge inside one iteration; Dynamic CV may overlap
  // the next iteration, but it may not reverse the work-item order of the
  // current iteration.
  for (size_t i = 1; i < graph.segments.size(); ++i) {
    if (graph.segments[i - 1].path == graph.segments[i].path)
      continue;
    graph.segments[i].orderDependencies.insert(i - 1);
    ++graph.crossPathOrderEdges;
  }
  return graph;
}

DynamicCVSegmentSchedule scheduleDynamicCVSegmentGraph(
    const DynamicCVSegmentGraph &graph, bool featureEnabled,
    DynamicCVCacheDepths cacheDepths) {
  DynamicCVSegmentSchedule result;
  if (graph.segments.empty())
    return result;

  cacheDepths.intra = std::max<int64_t>(1, cacheDepths.intra);
  cacheDepths.inter = std::max<int64_t>(1, cacheDepths.inter);
  cacheDepths.load = std::max<int64_t>(1, cacheDepths.load);
  std::vector<std::vector<int64_t>> nodeIds(graph.segments.size());
  std::vector<DynamicCVExpandedNode> nodes;
  std::vector<int64_t> previousIterationNodes;
  std::map<HWUnit, int64_t> lastResourceNode;

  for (int64_t iteration = 0; iteration < graph.iterationCount; ++iteration) {
    std::vector<int64_t> currentIterationNodes;
    for (const DynamicCVSegment &segment : graph.segments) {
      if (iteration >= segment.tripCount)
        continue;
      DynamicCVExpandedNode node;
      node.path = segment.path;
      node.resource = segment.resource;
      node.durationCycles = segment.durationCycles;
      int64_t nodeId = nodes.size();
      nodes.push_back(node);
      nodeIds[segment.id].push_back(nodeId);
      currentIterationNodes.push_back(nodeId);

      auto previousResourceIt = lastResourceNode.find(segment.resource);
      if (previousResourceIt != lastResourceNode.end())
        nodes[nodeId].dependencies.insert(previousResourceIt->second);
      lastResourceNode[segment.resource] = nodeId;

      if (!featureEnabled && iteration > 0) {
        // Feature Off preserves the original per-iteration barrier. This is
        // the waiting that max(Cube, Vector) previously erased completely.
        nodes[nodeId].dependencies.insert(previousIterationNodes.begin(),
                                          previousIterationNodes.end());
      }
    }
    previousIterationNodes = std::move(currentIterationNodes);
  }

  auto nodeForIteration = [&](int64_t segmentId, int64_t iteration) {
    const auto &ids = nodeIds[segmentId];
    if (ids.empty())
      return int64_t{-1};
    return ids[std::min<int64_t>(iteration, ids.size() - 1)];
  };

  // Add the true segment data edges for each concrete static iteration.
  for (const DynamicCVSegment &target : graph.segments) {
    for (int64_t iteration = 0; iteration < target.tripCount; ++iteration) {
      int64_t targetNode = nodeForIteration(target.id, iteration);
      auto addDependencies = [&](const std::set<int64_t> &dependencies) {
        for (int64_t sourceId : dependencies) {
        const DynamicCVSegment &source = graph.segments[sourceId];
        int64_t sourceNode = nodeForIteration(
            sourceId, std::min<int64_t>(iteration, source.tripCount - 1));
        if (sourceNode >= 0 && sourceNode != targetNode)
          nodes[targetNode].dependencies.insert(sourceNode);
        }
      };
      addDependencies(target.intraDependencies);
      addDependencies(target.interDependencies);
      addDependencies(target.loadDependencies);
      for (int64_t sourceId : target.orderDependencies) {
        int64_t sourceNode = nodeForIteration(
            sourceId, std::min<int64_t>(
                          iteration, graph.segments[sourceId].tripCount - 1));
        if (sourceNode >= 0 && sourceNode != targetNode)
          nodes[targetNode].dependencies.insert(sourceNode);
      }
    }
  }

  if (featureEnabled) {
    // A produced value occupies one slot in its queue until the corresponding
    // consumer releases it.  Reusing slot i%B therefore depends on completion
    // of consumer i-B.  The three queue depths come directly from compiler
    // parameters; no fitted benefit or penalty is added.
    auto addCapacityEdges = [&](const DynamicCVSegment &consumer,
                                const std::set<int64_t> &producerIds,
                                int64_t slots, DynamicCVQueue queue,
                                int64_t &edgeCount) {
      for (int64_t producerId : producerIds) {
        const DynamicCVSegment &producer = graph.segments[producerId];
        for (int64_t iteration = slots;
             iteration < producer.tripCount; ++iteration) {
          int64_t producerNode = nodeForIteration(producerId, iteration);
          int64_t releaseNode = nodeForIteration(
              consumer.id, std::min<int64_t>(iteration - slots,
                                             consumer.tripCount - 1));
          if (producerNode < 0 || releaseNode < 0 ||
              producerNode == releaseNode)
            continue;
          std::set<int64_t> *capacityDependencies = nullptr;
          switch (queue) {
          case DynamicCVQueue::Intra:
            capacityDependencies =
                &nodes[producerNode].intraCapacityDependencies;
            break;
          case DynamicCVQueue::Inter:
            capacityDependencies =
                &nodes[producerNode].interCapacityDependencies;
            break;
          case DynamicCVQueue::Load:
            capacityDependencies =
                &nodes[producerNode].loadCapacityDependencies;
            break;
          }
          if (capacityDependencies->insert(releaseNode).second)
            ++edgeCount;
        }
      }
    };
    for (const DynamicCVSegment &consumer : graph.segments) {
      addCapacityEdges(consumer, consumer.intraDependencies,
                       cacheDepths.intra, DynamicCVQueue::Intra,
                       result.intraCapacityEdges);
      addCapacityEdges(consumer, consumer.interDependencies,
                       cacheDepths.inter, DynamicCVQueue::Inter,
                       result.interCapacityEdges);
      addCapacityEdges(consumer, consumer.loadDependencies,
                       cacheDepths.load, DynamicCVQueue::Load,
                       result.loadCapacityEdges);
    }
  }

  std::vector<int64_t> endCycles(nodes.size(), 0);
  std::map<HWUnit, int64_t> resourceAvailable;
  for (size_t nodeId = 0; nodeId < nodes.size(); ++nodeId) {
    const DynamicCVExpandedNode &node = nodes[nodeId];
    int64_t startCycle = resourceAvailable[node.resource];
    for (int64_t dependency : node.dependencies) {
      if (dependency < 0 || dependency >= static_cast<int64_t>(nodeId))
        return result;
      startCycle = std::max(startCycle, endCycles[dependency]);
    }
    const int64_t startWithoutCapacity = startCycle;
    auto applyCapacityDependencies = [&](const std::set<int64_t> &dependencies,
                                         int64_t &blockingCycles) {
      int64_t queueReadyCycle = startWithoutCapacity;
      for (int64_t dependency : dependencies) {
        if (dependency < 0 || dependency >= static_cast<int64_t>(nodeId)) {
          startCycle = -1;
          return;
        }
        queueReadyCycle = std::max(queueReadyCycle, endCycles[dependency]);
      }
      blockingCycles +=
          std::max<int64_t>(0, queueReadyCycle - startWithoutCapacity);
      startCycle = std::max(startCycle, queueReadyCycle);
    };
    applyCapacityDependencies(node.intraCapacityDependencies,
                              result.intraBlockingCycles);
    applyCapacityDependencies(node.interCapacityDependencies,
                              result.interBlockingCycles);
    applyCapacityDependencies(node.loadCapacityDependencies,
                              result.loadBlockingCycles);
    if (startCycle < 0)
      return result;
    endCycles[nodeId] = startCycle + node.durationCycles;
    resourceAvailable[node.resource] = endCycles[nodeId];
    result.makespanCycles =
        std::max(result.makespanCycles, endCycles[nodeId]);
  }
  result.valid = true;
  result.nodeCount = nodes.size();
  return result;
}

DynamicCVStats estimateDynamicCV(
    const DynamicCVParams &params, const PipelineScheduler &scheduler,
    const HardwareConfig &config, int64_t cubePathCycles,
    int64_t vectorPathCycles, int64_t baseRooflineTotalCycles) {
  DynamicCVStats stats;
  if (!params.present)
    return stats;

  stats.used = true;
  stats.compilerStatusPresent = params.compilerStatusPresent;
  stats.compilerApplied = params.compilerApplied;
  stats.statusSource = params.compilerStatusPresent
                           ? "compiler_final"
                           : params.statusSource;
  stats.intraCacheCount = std::max<int64_t>(1, params.intraCacheCount);
  stats.interCacheCount = std::max<int64_t>(1, params.interCacheCount);
  stats.loadCacheCount = std::max<int64_t>(1, params.loadCacheCount);

  DynamicCVSegmentGraph graph = buildDynamicCVSegmentGraph(scheduler);
  stats.segmentCount = graph.segments.size();
  stats.workItemCount = stats.segmentCount;
  stats.dataDependencyEdges = graph.dataDependencyEdges;
  stats.segmentOrderEdges = graph.crossPathOrderEdges;
  stats.iterationCount = graph.iterationCount;
  bool hasCube = false;
  bool hasVector = false;
  for (const DynamicCVSegment &segment : graph.segments) {
    hasCube |= segment.path == TileMixPath::Cube;
    hasVector |= segment.path == TileMixPath::Vector;
    stats.crossCoreEdges += segment.interDependencies.size();
  }

  DynamicCVSegmentSchedule off =
      scheduleDynamicCVSegmentGraph(graph, false, {1, 1, 1});
  DynamicCVSegmentSchedule on = scheduleDynamicCVSegmentGraph(
      graph, true,
      {stats.intraCacheCount, stats.interCacheCount, stats.loadCacheCount});
  if (!off.valid || !on.valid || !hasCube || !hasVector ||
      cubePathCycles <= 0 || vectorPathCycles <= 0) {
    stats.skipReason = "mixed_cube_vector_segment_dag_not_proven";
    return stats;
  }

  stats.segmentModelValid = true;
  stats.offMakespanCycles = off.makespanCycles;
  stats.onMakespanCycles = on.makespanCycles;
  stats.intraCapacityEdges = on.intraCapacityEdges;
  stats.interCapacityEdges = on.interCapacityEdges;
  stats.loadCapacityEdges = on.loadCapacityEdges;
  stats.intraBlockingCycles = on.intraBlockingCycles;
  stats.interBlockingCycles = on.interBlockingCycles;
  stats.loadBlockingCycles = on.loadBlockingCycles;
  stats.originalMakespanCycles = stats.offMakespanCycles;
  stats.transformedMakespanCycles = stats.onMakespanCycles;
  stats.referenceDeltaCycles =
      stats.offMakespanCycles - baseRooflineTotalCycles;

  WorkspaceMultibufferEvidence evidence =
      inferWorkspaceMultibufferEvidence(scheduler);
  stats.cacheBytes = evidence.bytesPerSlot *
      (stats.intraCacheCount + stats.interCacheCount + stats.loadCacheCount);
  stats.syncCycles = stats.crossCoreEdges * stats.iterationCount *
      (config.getSyncOpCycles("set_flag", 1) +
       config.getSyncOpCycles("wait_flag", 2));
  stats.controlCycles = on.nodeCount;
  stats.offTotalCycles = stats.offMakespanCycles;
  stats.onTotalCycles =
      stats.onMakespanCycles + stats.syncCycles + stats.controlCycles;

  if (!params.enabled) {
    stats.skipReason = "disabled";
    return stats;
  }
  if (!params.targetSupported) {
    stats.skipReason = "target_not_supported";
    return stats;
  }
  if (params.compilerStatusPresent && !params.compilerApplied) {
    stats.skipReason = params.compilerSkipReason.empty()
                           ? "compiler_rejected"
                           : params.compilerSkipReason;
    return stats;
  }

  stats.netDeltaCycles = stats.onTotalCycles - stats.offTotalCycles;
  stats.eligible = true;
  stats.adjustmentApplied = stats.netDeltaCycles != 0;
  stats.skipReason = "none";
  return stats;
}

TileMixStats estimateTileMix(
    const TileMixParams &params, const PipelineScheduler &scheduler,
    const HardwareConfig &config, int64_t cubePathCycles,
    int64_t vectorPathCycles, int64_t cubeTransferCycles,
    int64_t vectorTransferCycles, int64_t baseCycles) {
  TileMixStats stats;
  stats.baseCycles = baseCycles;
  stats.adjustedCycles = baseCycles;
  if (!params.hasAny || !isTileMixDagModelEnabled())
    return stats;

  stats.used = true;
  stats.summarySource = params.summarySource;
  stats.cubeSkipReason = params.cubeSkipReason;
  stats.vectorSkipReason = params.vectorSkipReason;
  stats.syncOpsBefore = params.syncOpsBefore;
  stats.syncOpsAfter = params.syncOpsAfter;

  for (const auto &op : scheduler.getAllOps()) {
    TileMixPath path = getTileMixPath(op.hwUnit);
    if (path == TileMixPath::None)
      continue;
    int64_t loopMultiplier = std::max<int64_t>(op.loopMultiplier, 1);
    if (path == TileMixPath::Cube) {
      stats.cubeLoopTrip = std::max(stats.cubeLoopTrip, loopMultiplier);
      if (isTileMixLayoutOp(op.hwUnit))
        ++stats.cubeLayoutOpCount;
      if (op.mlirOp) {
        if (auto storeOp = dyn_cast<CubeStoreOp>(op.mlirOp))
          stats.cubeWorkspaceBytes +=
              std::max<int64_t>(storeOp.getTransferBytes(), 0);
      }
    } else {
      stats.vectorLoopTrip = std::max(stats.vectorLoopTrip, loopMultiplier);
      if (isTileMixLayoutOp(op.hwUnit))
        ++stats.vectorLayoutOpCount;
      if (op.mlirOp) {
        if (auto loadOp = dyn_cast<VectorLoadOp>(op.mlirOp))
          stats.vectorWorkspaceBytes +=
              std::max<int64_t>(loadOp.getTransferBytes(), 0);
        if (auto storeOp = dyn_cast<VectorStoreOp>(op.mlirOp))
          stats.vectorWorkspaceBytes +=
              std::max<int64_t>(storeOp.getTransferBytes(), 0);
      }
    }
  }

  int64_t l0cBytes = static_cast<int64_t>(config.getMemorySizeBytes("l0c"));
  int64_t ubBytes = static_cast<int64_t>(config.getMemorySizeBytes("ub"));
  if (l0cBytes <= 0)
    l0cBytes = 256 * 1024;
  if (ubBytes <= 0)
    ubBytes = 256 * 1024;
  TileMixModelConfig model = getTileMixModelConfig(config);
  TileMixDerivedFeatures features = inferTileMixDerivedFeatures(scheduler);
  stats.inferredTileM = features.tileM;
  stats.inferredTileN = features.tileN;
  stats.tileShapeSource = features.tileShapeSource;
  stats.dtypeSource = features.dtypeSource;
  stats.handoffSource = features.handoffSource;
  stats.intermediateSource = features.intermediateSource;
  stats.handoffFeatureDim = features.handoffFeatureDim;
  stats.handoffDtypeBytes = features.dtypeBytes;
  stats.handoffTileBytes = features.handoffTileBytes;
  stats.intermediateTileBytes = features.intermediateTileBytes;

  int64_t vectorAlignment = std::max<int64_t>(1, config.getVectorWidthBytes());
  TileMixSideDecision cubeDecision;
  TileMixSideDecision vectorDecision;
  bool usingPassSummary = params.summaryPresent && params.summaryValid;
  if (usingPassSummary) {
    cubeDecision.known = true;
    cubeDecision.applied = params.cubeApplied;
    cubeDecision.segments = params.cubeApplied ? params.cubeSegments : 1;
    cubeDecision.skipReason = params.cubeSkipReason;
    vectorDecision.known = true;
    vectorDecision.applied = params.vectorApplied;
    vectorDecision.segments = params.vectorApplied ? params.vectorSegments : 1;
    vectorDecision.skipReason = params.vectorSkipReason;
    stats.summarySource = params.summarySource;
    stats.confidencePercent = 100;
  } else {
    // P0-1 primary path: infer only the eligibility conditions explicitly
    // stated by principle.docx from TTIR/AscendModel evidence. The optional
    // HIVM pass summary is a higher-confidence override, not a prerequisite.
    cubeDecision = inferTileMixSideFromTTIR(
        TileMixPath::Cube, params.cubeLoop, cubePathCycles,
        stats.cubeLayoutOpCount, stats.cubeWorkspaceBytes, l0cBytes,
        vectorAlignment);
    vectorDecision = inferTileMixSideFromTTIR(
        TileMixPath::Vector, params.vectorLoop, vectorPathCycles,
        stats.vectorLayoutOpCount, stats.vectorWorkspaceBytes, ubBytes,
        vectorAlignment);
    stats.summarySource = params.summaryPresent
        ? "ttir_principle_v2_target_trip_after_invalid_summary"
        : "ttir_principle_v2_target_trip";
    bool hasUnknownRequestedSide =
        (params.cubeLoop > 1 && !cubeDecision.known) ||
        (params.vectorLoop > 1 && !vectorDecision.known);
    stats.confidencePercent = hasUnknownRequestedSide ? 50 : 70;
  }

  // P0-3 validates each side independently. Unknown/contradictory evidence on
  // one side disables only that side; no global heuristic benefit is invented.
  auto validatePassDecision = [](TileMixSideDecision &decision,
                                 int64_t pathCycles, int64_t layoutOps,
                                 int64_t workspaceBytes,
                                 llvm::StringRef side) {
    if (!decision.applied)
      return;
    if (decision.segments <= 1 || pathCycles <= 0 || layoutOps <= 0 ||
        workspaceBytes <= 0) {
      decision.known = false;
      decision.applied = false;
      decision.segments = 1;
      decision.skipReason = "model_missing_" + side.str() + "_evidence";
    }
  };
  validatePassDecision(cubeDecision, cubePathCycles, stats.cubeLayoutOpCount,
                       stats.cubeWorkspaceBytes, "cube");
  validatePassDecision(vectorDecision, vectorPathCycles,
                       stats.vectorLayoutOpCount, stats.vectorWorkspaceBytes,
                       "vector");

  stats.cubeApplied = cubeDecision.applied;
  stats.vectorApplied = vectorDecision.applied;
  stats.cubeSkipReason = cubeDecision.skipReason;
  stats.vectorSkipReason = vectorDecision.skipReason;
  stats.cubeSegmentCount = cubeDecision.applied ? cubeDecision.segments : 1;
  stats.vectorSegmentCount = vectorDecision.applied ? vectorDecision.segments : 1;
  stats.cubeTargetBytes = l0cBytes;
  stats.vectorTargetBytes = ubBytes;
  stats.cubeSubtileBytes =
      ceilDiv(stats.cubeWorkspaceBytes, stats.cubeSegmentCount);
  stats.vectorSubtileBytes =
      ceilDiv(stats.vectorWorkspaceBytes, stats.vectorSegmentCount);
  stats.handoffSegmentCount =
      std::max(stats.cubeSegmentCount, stats.vectorSegmentCount);
  stats.handoffSubtileBytes =
      ceilDiv(stats.handoffTileBytes, stats.handoffSegmentCount);
  stats.handoffTargetBytes = minPositiveLocalBytes(l0cBytes, ubBytes);
  stats.intermediateTargetBytes = ubBytes;

  bool anyKnownRequestedSide =
      (params.cubeLoop > 1 && cubeDecision.known) ||
      (params.vectorLoop > 1 && vectorDecision.known);
  if (!anyKnownRequestedSide)
    return stats;

  bool cubeApplied = cubeDecision.applied;
  bool vectorApplied = vectorDecision.applied;

  // P0-2: every modeled term below is a marginal number of cycles. There is no
  // baseCycles * empirical_ratio relief, no generic loop mismatch penalty, and
  // no reward merely for requesting a larger segment count.
  int64_t cubeBeforePressure = overflowTransferCycles(
      stats.cubeWorkspaceBytes, stats.cubeTargetBytes, 1,
      stats.cubeWorkspaceBytes, cubeTransferCycles);
  int64_t cubeAfterPressure = cubeApplied
      ? overflowTransferCycles(stats.cubeSubtileBytes, stats.cubeTargetBytes,
                               stats.cubeSegmentCount,
                               stats.cubeWorkspaceBytes, cubeTransferCycles)
      : cubeBeforePressure;
  int64_t vectorBeforePressure = overflowTransferCycles(
      stats.vectorWorkspaceBytes, stats.vectorTargetBytes, 1,
      stats.vectorWorkspaceBytes, vectorTransferCycles);
  int64_t vectorAfterPressure = vectorApplied
      ? overflowTransferCycles(stats.vectorSubtileBytes, stats.vectorTargetBytes,
                               stats.vectorSegmentCount,
                               stats.vectorWorkspaceBytes, vectorTransferCycles)
      : vectorBeforePressure;

  int64_t cubeDelta = cubeAfterPressure - cubeBeforePressure;
  int64_t vectorDelta = vectorAfterPressure - vectorBeforePressure;
  int64_t adjustedCubePath = std::max<int64_t>(1, cubePathCycles + cubeDelta);
  int64_t adjustedVectorPath =
      std::max<int64_t>(1, vectorPathCycles + vectorDelta);
  int64_t pressureAdjustedCycles =
      std::max(adjustedCubePath, adjustedVectorPath);

  stats.bufferDeltaCycles = cubeDelta + vectorDelta;
  stats.pipelineDeltaCycles = pressureAdjustedCycles - baseCycles;
  stats.externalSyncDeltaCycles = usingPassSummary
      ? tileMixExternalSyncDeltaCycles(params, config)
      : 0;
  int64_t extraSegments =
      (cubeApplied ? stats.cubeSegmentCount - 1 : 0) +
      (vectorApplied ? stats.vectorSegmentCount - 1 : 0);
  stats.scalarControlDeltaCycles =
      extraSegments * model.loopControlCyclesPerSegment;
  // GM bytes before/after are not yet emitted by the proprietary pass. Unknown
  // terms are zero by fail-closed policy, never inferred as a percentage.
  stats.gmDeltaCycles = 0;
  stats.workspaceReliefCycles = std::max<int64_t>(0, -stats.pipelineDeltaCycles);
  stats.bufferFitPenaltyCycles = cubeAfterPressure + vectorAfterPressure;
  stats.adjustedCycles = std::max<int64_t>(
      1, baseCycles + stats.gmDeltaCycles + stats.externalSyncDeltaCycles +
             stats.pipelineDeltaCycles + stats.scalarControlDeltaCycles);
  stats.netDeltaCycles = stats.adjustedCycles - baseCycles;
  stats.valid = true;
  stats.adjustmentApplied = cubeApplied || vectorApplied;
  return stats;
}

int getTrackId(HWUnit unit) {
  switch (unit) {
    case HWUnit::Cube:     return 1;
    case HWUnit::CubeMTE2: return 2;
    case HWUnit::FixPipe:  return 3;
    case HWUnit::Vector:   return 4;
    case HWUnit::VecMTE2:  return 5;
    case HWUnit::MTE3:     return 6;
    case HWUnit::Scalar:   return 7;
    default:               return 0;
  }
}

const char* getColorName(HWUnit unit) {
  switch (unit) {
    case HWUnit::Cube:     return "rail_response";
    case HWUnit::CubeMTE2: return "rail_load";
    case HWUnit::FixPipe:  return "cq_build_passed";
    case HWUnit::Vector:   return "rail_animation";
    case HWUnit::VecMTE2:  return "good";
    case HWUnit::MTE3:     return "bad";
    case HWUnit::Scalar:   return "grey";
    default:               return "generic_work";
  }
}

HWUnit getOpHWUnit(Operation *op) {
  if (isa<MatmulOp>(op)) return HWUnit::Cube;
  if (isa<CubeLoadOp>(op)) return HWUnit::CubeMTE2;
  if (isa<CubeStoreOp>(op)) return HWUnit::FixPipe;
  if (isa<VectorLoadOp>(op)) return HWUnit::VecMTE2;
  if (isa<VectorStoreOp>(op)) return HWUnit::MTE3;
  if (isa<AddOp, SubOp, MulOp, DivOp, MaxOp, MinOp,
          ExpOp, LogOp, SqrtOp, RsqrtOp, TanhOp, SigmoidOp,
          NegOp, AbsOp, ReluOp, CastOp,
          ReduceSumOp, ReduceMaxOp, ReduceMinOp, ReduceProdOp,
          BroadcastOp, SelectOp>(op))
    return HWUnit::Vector;
  return HWUnit::Scalar;
}

static KernelLaunchContext makeKernelLaunchContext(
    int64_t bodyCycles, const PipelineScheduler &scheduler,
    llvm::StringRef bindingsStr) {
  KernelLaunchContext ctx;
  ctx.bodyCycles = bodyCycles;
  ctx.opCount = scheduler.getAllOps().size();
  for (const PipelineOp &op : scheduler.getAllOps()) {
    ctx.hasVector |= op.hwUnit == HWUnit::Vector || op.hwUnit == HWUnit::VecMTE2;
    ctx.hasCube |= op.hwUnit == HWUnit::Cube || op.hwUnit == HWUnit::CubeMTE2 ||
                   op.hwUnit == HWUnit::FixPipe;
    ctx.hasMTE |= op.hwUnit == HWUnit::VecMTE2 || op.hwUnit == HWUnit::MTE3 ||
                  op.hwUnit == HWUnit::CubeMTE2 ||
                  op.hwUnit == HWUnit::FixPipe;
  }
  utils::inferKernelModeFromLaunchFeatures(ctx);
  utils::applyKernelLaunchBindings(ctx, bindingsStr);
  return ctx;
}

/// Generate Perfetto trace with loop unrolling.
/// If maxIterations > 0, limits the number of iterations shown in trace.
void generatePerfettoTrace(const PipelineScheduler &scheduler,
                           StringRef filename,
                           int64_t oneIterCycles,
                           int64_t bodyCycles,
                           int64_t launchOverheadCycles,
                           int64_t predictedTotalCycles,
                           int64_t maxIterations = 100) {
  std::error_code EC;
  llvm::raw_fd_ostream file(filename, EC, llvm::sys::fs::OF_Text);
  if (EC) {
    llvm::errs() << "Error opening file " << filename << ": " << EC.message() << "\n";
    return;
  }
  
  const auto &config = scheduler.getConfig();
  const auto &allOps = scheduler.getAllOps();
  
  // Calculate the maximum loop multiplier to determine iteration count
  int64_t maxLoopMultiplier = 1;
  for (const auto &op : allOps) {
    maxLoopMultiplier = std::max(maxLoopMultiplier, op.loopMultiplier);
  }
  
  // Limit iterations for visualization (avoid huge traces)
  int64_t numIterations = std::min(maxLoopMultiplier, maxIterations);
  bool truncated = (maxLoopMultiplier > maxIterations);
  
  double cycleToUs = 1.0;  // 1 cycle = 1 unit for visualization
  
  file << "{\n  \"traceEvents\": [\n";
  bool first = true;
  
  // Track metadata
  struct TrackInfo { int tid; const char* name; };
  TrackInfo tracks[] = {
    {1, "Cube Core"}, {2, "Cube MTE2 (HBM->L1)"}, {3, "FixPipe (L0C->HBM)"},
    {4, "Vector Core"}, {5, "Vec MTE2 (HBM->UB)"}, {6, "MTE3 (UB->HBM)"}, {7, "Scalar"}
  };
  
  // Write track metadata
  for (const auto &track : tracks) {
    if (!first) file << ",\n";
    first = false;
    file << "    {\"name\": \"thread_name\", \"ph\": \"M\", \"pid\": 1, \"tid\": " 
         << track.tid << ", \"args\": {\"name\": \"" << track.name << "\"}}";
  }
  
  for (const auto &track : tracks) {
    file << ",\n    {\"name\": \"thread_sort_index\", \"ph\": \"M\", \"pid\": 1, \"tid\": " 
         << track.tid << ", \"args\": {\"sort_index\": " << track.tid << "}}";
  }
  
  file << ",\n    {\"name\": \"process_name\", \"ph\": \"M\", \"pid\": 1, "
       << "\"args\": {\"name\": \"" << config.getName().str() << " Pipeline";
  if (truncated) {
    file << " (showing " << numIterations << "/" << maxLoopMultiplier << " iterations)";
  }
  file << "\"}}";
  
  // Calculate actual total cycles shown in trace
  int64_t traceTotalCycles = 0;
  
  // Generate events for each iteration
  // Key insight: operations with different loopMultipliers execute different numbers of times
  // We need to track per-HW-unit time to model pipeline parallelism across iterations
  
  // Track end time for each hardware unit
  llvm::DenseMap<HWUnit, int64_t> hwUnitEndTime;
  for (int i = 0; i <= static_cast<int>(HWUnit::Scalar); ++i) {
    hwUnitEndTime[static_cast<HWUnit>(i)] = 0;
  }
  
  // For each iteration
  for (int64_t iter = 0; iter < numIterations; ++iter) {
    // Track dependencies within this iteration
    llvm::DenseMap<int64_t, int64_t> opEndTimes;  // opId -> endTime in this iter
    
    for (const auto &op : allOps) {
      // Check if this op executes in this iteration
      // An op with loopMultiplier=N executes N times
      if (iter >= op.loopMultiplier)
        continue;
      
      // Calculate start time considering:
      // 1. Dependencies from previous ops in this iteration
      // 2. Hardware unit availability
      int64_t startTime = hwUnitEndTime[op.hwUnit];
      
      // Check dependencies
      for (int64_t depId : op.dependsOn) {
        auto it = opEndTimes.find(depId);
        if (it != opEndTimes.end()) {
          startTime = std::max(startTime, it->second);
        }
      }
      
      int64_t endTime = startTime + op.duration;
      
      // Update tracking
      hwUnitEndTime[op.hwUnit] = endTime;
      opEndTimes[op.opId] = endTime;
      traceTotalCycles = std::max(traceTotalCycles, endTime);
      
      // Write event
      int tid = getTrackId(op.hwUnit);
      file << ",\n    {\"name\": \"" << op.opName;
      if (op.loopMultiplier > 1) {
        file << "[" << iter << "]";  // Show iteration number
      }
      file << "\", "
           << "\"cat\": \"" << stringifyHWUnit(op.hwUnit).str() << "\", \"ph\": \"X\", "
           << "\"ts\": " << llvm::format("%.3f", startTime * cycleToUs) << ", "
           << "\"dur\": " << llvm::format("%.3f", op.duration * cycleToUs) << ", "
           << "\"pid\": 1, \"tid\": " << tid << ", "
           << "\"cname\": \"" << getColorName(op.hwUnit) << "\", "
           << "\"args\": {"
           << "\"op_id\": " << op.opId << ", "
           << "\"iteration\": " << iter << ", "
           << "\"cycles\": " << op.duration << ", "
           << "\"loop_multiplier\": " << op.loopMultiplier
           << "}}";
    }
  }
  
  // Add markers for total timeline
  for (const auto &track : tracks) {
    file << ",\n    {\"name\": \"\", \"cat\": \"marker\", \"ph\": \"i\", \"s\": \"t\", "
         << "\"ts\": 0, \"pid\": 1, \"tid\": " << track.tid << "}";
    file << ",\n    {\"name\": \"\", \"cat\": \"marker\", \"ph\": \"i\", \"s\": \"t\", "
         << "\"ts\": " << llvm::format("%.3f", traceTotalCycles * cycleToUs) 
         << ", \"pid\": 1, \"tid\": " << track.tid << "}";
  }
  
  // Add iteration markers
  if (numIterations > 1) {
    // Add counter track for iteration progress
    file << ",\n    {\"name\": \"Iterations\", \"ph\": \"C\", \"ts\": 0, \"pid\": 1, "
         << "\"args\": {\"shown\": " << numIterations << ", \"total\": " << maxLoopMultiplier << "}}";
  }
  
  file << "\n  ],\n";
  
  // Metadata
  file << "  \"metadata\": {\n";
  file << "    \"hardware\": \"" << config.getName().str() << "\",\n";
  file << "    \"one_iter_cycles\": " << oneIterCycles << ",\n";
  file << "    \"total_cycles\": " << bodyCycles << ",\n";
  file << "    \"body_cycles\": " << bodyCycles << ",\n";
  file << "    \"kernel_launch_overhead_cycles\": "
       << launchOverheadCycles << ",\n";
  file << "    \"predicted_total_cycles\": " << predictedTotalCycles << ",\n";
  file << "    \"trace_cycles\": " << traceTotalCycles << ",\n";
  file << "    \"iterations_shown\": " << numIterations << ",\n";
  file << "    \"iterations_total\": " << maxLoopMultiplier << ",\n";
  file << "    \"clock_freq_ghz\": " << config.getClockFrequencyGHz() << ",\n";
  file << "    \"estimated_time_us\": "
       << llvm::format("%.3f", config.cyclesToMicroseconds(bodyCycles))
       << ",\n";
  file << "    \"predicted_total_time_us\": "
       << llvm::format("%.3f",
                       config.cyclesToMicroseconds(predictedTotalCycles))
       << "\n";
  file << "  },\n";
  
  file << "  \"displayTimeUnit\": \"ns\"\n";
  file << "}\n";
  
  file.close();
  
  llvm::outs() << "Perfetto trace: " << filename << "\n";
  if (truncated) {
    llvm::outs() << "  Note: Showing " << numIterations << " of " << maxLoopMultiplier 
                 << " iterations (use full trace for complete view)\n";
  }
  llvm::outs() << "  Trace cycles: " << traceTotalCycles << " (";
  if (truncated) {
    llvm::outs() << "partial, ";
  }
  llvm::outs() << "actual body: " << bodyCycles
               << ", predicted total: " << predictedTotalCycles << ")\n";
  llvm::outs() << "  Open with: https://ui.perfetto.dev/\n";
}

struct PipelineAnalysisPass
    : public impl::PipelineAnalysisPassBase<PipelineAnalysisPass> {
  using PipelineAnalysisPassBase::PipelineAnalysisPassBase;
  
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // Load an independent, validated hardware config for this analysis without
    // mutating process-global state (back-port of triton-ascend #337).
    std::string hardwareConfigError;
    auto hardwareConfig =
        loadHardwareConfigForAnalysis(hardwareConfigPath, hardwareConfigError);
    if (!hardwareConfig) {
      emitError(module.getLoc(), hardwareConfigError);
      return signalPassFailure();
    }
    const HardwareConfig &config = *hardwareConfig;

    // Parse bindings
    llvm::DenseMap<unsigned, int64_t> argBindings;
    llvm::StringMap<int64_t> programIdBindings;
    SmallVector<int64_t> loopTripCountOverrides;
    
    if (!argBindingsStr.empty()) {
      std::string parseError;
      if (!parseBindings(argBindingsStr, argBindings, programIdBindings, parseError)) {
        emitError(module.getLoc(), parseError);
        return signalPassFailure();
      }
    }
    
    if (!loopTripCountsStr.empty()) {
      std::string parseError;
      if (!parseLoopTripCounts(loopTripCountsStr, loopTripCountOverrides, parseError)) {
        emitError(module.getLoc(), parseError);
        return signalPassFailure();
      }
    }

    TileMixParams tileMixParams = parseTileMixParams(compileParamsStr);
    WorkspaceMultibufferParams workspaceMultibufferParams =
        parseWorkspaceMultibufferParams(compileParamsStr);
    DynamicCVParams dynamicCVParams =
        parseDynamicCVParams(compileParamsStr);
    
    // Collect loops and ensure trip counts are set
    SmallVector<scf::ForOp> allLoops;
    module.walk([&](scf::ForOp forOp) { allLoops.push_back(forOp); });
    
    bool hasError = false;
    for (size_t loopIdx = 0; loopIdx < allLoops.size(); ++loopIdx) {
      scf::ForOp forOp = allLoops[loopIdx];
      
      if (forOp->hasAttr("ascend.trip_count"))
        continue;
      
      int64_t tripCount = 1;
      if (loopIdx < loopTripCountOverrides.size()) {
        tripCount = loopTripCountOverrides[loopIdx];
      } else {
        auto result = getScfForTripCountWithBindings(forOp, argBindings, programIdBindings);
        if (result.isStatic) {
          tripCount = result.staticTripCount;
        } else {
          emitError(forOp.getLoc(), "Loop " + std::to_string(loopIdx) + 
                    " trip count unknown. " + result.errorMsg);
          hasError = true;
          continue;
        }
      }
      
      forOp->setAttr("ascend.trip_count",
                     IntegerAttr::get(IntegerType::get(forOp.getContext(), 64), tripCount));
    }
    
    if (hasError) return signalPassFailure();
    
    // Build scheduler
    PipelineScheduler scheduler(&config);
    llvm::DenseMap<Value, int64_t> valueProducers;
    
    module.walk([&](Operation *op) {
      if (isa<scf::ForOp, scf::YieldOp, scf::IfOp>(op)) return;
      
      auto opIdAttr = op->getAttrOfType<IntegerAttr>("op_id");
      if (!opIdAttr) return;
      
      int64_t opId = opIdAttr.getInt();
      auto cyclesAttr = op->getAttrOfType<IntegerAttr>("estimated_cycles");
      int64_t cycles = cyclesAttr ? cyclesAttr.getInt() : 1;
      
      PipelineOp pipelineOp;
      pipelineOp.opId = opId;
      pipelineOp.hwUnit = getOpHWUnit(op);
      pipelineOp.duration = cycles;
      pipelineOp.mlirOp = op;
      pipelineOp.opName = op->getName().getStringRef().str();
      pipelineOp.loopMultiplier = getLoopMultiplier(op);
      
      for (Value operand : op->getOperands()) {
        auto it = valueProducers.find(operand);
        if (it != valueProducers.end()) {
          pipelineOp.dependsOn.push_back(it->second);
          scheduler.addDependency(it->second, opId);
        }
      }
      
      for (Value result : op->getResults())
        valueProducers[result] = opId;
      
      scheduler.addOperation(pipelineOp);
    });
    
    if (!scheduler.schedule()) {
      emitError(module.getLoc(), "Failed to schedule pipeline");
      return signalPassFailure();
    }
    
    // Calculate cycles using roofline model
    // oneIterCycles from scheduler already considers HW unit parallelism for one iteration
    int64_t oneIterCycles = scheduler.getTotalCycles();
    
    // For total cycles with loops, we need to consider:
    // 1. Each HW unit's total work across all iterations
    // 2. Take max (not sum) since they can overlap
    
    // Collect per-HW-unit cycles
    llvm::DenseMap<HWUnit, int64_t> hwUnitCycles;
    for (const auto &pipelineOp : scheduler.getAllOps()) {
      hwUnitCycles[pipelineOp.hwUnit] += pipelineOp.duration * pipelineOp.loopMultiplier;
    }
    
    // Group by path and apply roofline model.
    // Cube path: max(Cube, CubeMTE2, FixPipe). No cube-side mutex on 910B
    // (tilesim pipe_exclusive_config only pairs AIV MTE2<->MTE3).
    int64_t cubePathCycles = std::max({
      hwUnitCycles[HWUnit::Cube],
      hwUnitCycles[HWUnit::CubeMTE2],
      hwUnitCycles[HWUnit::FixPipe]
    });

    // Vector path. Vector compute overlaps with load/store transfers, but on
    // 910B AIV the MTE2 (load) and MTE3 (store) units share one physical
    // pipeline (tilesim MutexComponents) and must serialize. With the mutex
    // the transfer time is VecMTE2 + MTE3; without it (legacy) they are
    // assumed to overlap (max). This is root cause 3.
    int64_t vecTransfer;
    if (config.areMutexUnits("vec_mte2", "mte3"))
      vecTransfer = hwUnitCycles[HWUnit::VecMTE2] + hwUnitCycles[HWUnit::MTE3];
    else
      vecTransfer = std::max(hwUnitCycles[HWUnit::VecMTE2], hwUnitCycles[HWUnit::MTE3]);
    int64_t vectorPathCycles = std::max(hwUnitCycles[HWUnit::Vector], vecTransfer);

    // Total: max of paths (Cube and Vector paths overlap). The primary
    // pre-compilation path uses principle-backed TTIR eligibility. An optional
    // real TileCubeVectorLoop summary overrides that inference for validation.
    int64_t baseRooflineTotalCycles = std::max(cubePathCycles, vectorPathCycles);
    int64_t cubeTransferCycles =
        hwUnitCycles[HWUnit::CubeMTE2] + hwUnitCycles[HWUnit::FixPipe];
    int64_t vectorTransferCycles =
        hwUnitCycles[HWUnit::VecMTE2] + hwUnitCycles[HWUnit::MTE3];
    TileMixStats tileMixStats = estimateTileMix(
        tileMixParams, scheduler, config, cubePathCycles, vectorPathCycles,
        cubeTransferCycles, vectorTransferCycles, baseRooflineTotalCycles);
    bool dynamicCVSegmentDagModelEnabled =
        isDynamicCVSegmentDagModelEnabled();
    DynamicCVStats dynamicCVStats;
    if (dynamicCVSegmentDagModelEnabled) {
      dynamicCVStats = estimateDynamicCV(
          dynamicCVParams, scheduler, config, cubePathCycles, vectorPathCycles,
          baseRooflineTotalCycles);
    } else {
      // Keep enough metadata for diagnostics, but do not build the segment DAG
      // or add any Dynamic-CV/Multibuffer delta in legacy-max mode.
      dynamicCVStats.used = dynamicCVParams.present;
      dynamicCVStats.compilerStatusPresent =
          dynamicCVParams.compilerStatusPresent;
      dynamicCVStats.compilerApplied = dynamicCVParams.compilerApplied;
      dynamicCVStats.intraCacheCount = dynamicCVParams.intraCacheCount;
      dynamicCVStats.interCacheCount = dynamicCVParams.interCacheCount;
      dynamicCVStats.loadCacheCount = dynamicCVParams.loadCacheCount;
      dynamicCVStats.statusSource = dynamicCVParams.statusSource;
      dynamicCVStats.skipReason = dynamicCVParams.enabled
                                      ? "segment_dag_model_disabled"
                                      : "disabled";
    }
    CVFeatureMode cvFeatureMode = CVFeatureMode::Base;
    WorkspaceMultibufferStats workspaceMultibufferStats;
    if (dynamicCVParams.present && dynamicCVParams.enabled &&
        !dynamicCVSegmentDagModelEnabled) {
      cvFeatureMode = CVFeatureMode::DynamicCVLegacyMax;
    } else if (dynamicCVStats.eligible) {
      // The compiler selects Dynamic CV instead of ordinary Multibuffer. Do
      // not add both deltas for one config.
      cvFeatureMode = CVFeatureMode::DynamicCV;
    } else if (dynamicCVParams.present && dynamicCVParams.enabled) {
      if (workspaceMultibufferParams.present) {
        cvFeatureMode = CVFeatureMode::OrdinaryMultibufferFallback;
        workspaceMultibufferStats = estimateWorkspaceMultibuffer(
            workspaceMultibufferParams, scheduler, config, cubePathCycles,
            vectorPathCycles, baseRooflineTotalCycles);
      }
    } else if (workspaceMultibufferParams.present) {
      cvFeatureMode = CVFeatureMode::OrdinaryMultibuffer;
      workspaceMultibufferStats = estimateWorkspaceMultibuffer(
          workspaceMultibufferParams, scheduler, config, cubePathCycles,
          vectorPathCycles, baseRooflineTotalCycles);
    }
    int64_t tileMixDeltaCycles =
        tileMixStats.valid ? tileMixStats.netDeltaCycles : 0;
    int64_t workspaceMultibufferDeltaCycles =
        workspaceMultibufferStats.valid
            ? workspaceMultibufferStats.netDeltaCycles
            : 0;
    int64_t dynamicCVDeltaCycles =
        dynamicCVStats.eligible ? dynamicCVStats.netDeltaCycles : 0;
    int64_t dynamicCVReferenceDeltaCycles =
        dynamicCVStats.segmentModelValid
            ? dynamicCVStats.referenceDeltaCycles
            : 0;
    int64_t rooflineTotalCycles = std::max<int64_t>(
        1, baseRooflineTotalCycles + tileMixDeltaCycles +
               workspaceMultibufferDeltaCycles +
               dynamicCVReferenceDeltaCycles + dynamicCVDeltaCycles);

    // Also calculate simple sum for comparison
    int64_t simpleSumCycles = 0;
    for (const auto &pipelineOp : scheduler.getAllOps())
      simpleSumCycles += pipelineOp.duration * pipelineOp.loopMultiplier;

    KernelLaunchContext launchCtx =
        makeKernelLaunchContext(rooflineTotalCycles, scheduler, argBindingsStr);
    KernelLaunchEstimate launch =
        config.estimateKernelLaunchOverhead(launchCtx);
    int64_t predictedTotalCycles = rooflineTotalCycles + launch.totalCycles;
    
    module->setAttr("ascend.scheduled_cycles_one_iter",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64), oneIterCycles));
    module->setAttr("ascend.roofline_cycles",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64), rooflineTotalCycles));
    module->setAttr("ascend.base_roofline_cycles",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64), baseRooflineTotalCycles));
    module->setAttr("ascend.kernel_body_cycles",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                                     rooflineTotalCycles));
    module->setAttr("ascend.kernel_launch_overhead_cycles",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                                     launch.totalCycles));
    module->setAttr("ascend.predicted_total_cycles",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                                     predictedTotalCycles));
    module->setAttr("ascend.cv_feature_mode",
                    StringAttr::get(module.getContext(),
                                    cvFeatureModeName(cvFeatureMode)));
    if (tileMixStats.used) {
      module->setAttr("ascend.tile_mix_schedule_model",
                      StringAttr::get(module.getContext(), "ttir_principle_marginal_cycles_v5_target_trip_peer_model"));
      module->setAttr("ascend.tile_mix_model_valid",
                      BoolAttr::get(module.getContext(), tileMixStats.valid));
      module->setAttr("ascend.tile_mix_adjustment_applied",
                      BoolAttr::get(module.getContext(), tileMixStats.adjustmentApplied));
      module->setAttr("ascend.tile_mix_confidence_percent",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.confidencePercent));
      module->setAttr("ascend.tile_mix_summary_source",
                      StringAttr::get(module.getContext(), tileMixStats.summarySource));
      module->setAttr("ascend.tile_mix_cube_applied",
                      BoolAttr::get(module.getContext(), tileMixStats.cubeApplied));
      module->setAttr("ascend.tile_mix_vector_applied",
                      BoolAttr::get(module.getContext(), tileMixStats.vectorApplied));
      module->setAttr("ascend.tile_mix_cube_skip_reason",
                      StringAttr::get(module.getContext(), tileMixStats.cubeSkipReason));
      module->setAttr("ascend.tile_mix_vector_skip_reason",
                      StringAttr::get(module.getContext(), tileMixStats.vectorSkipReason));
      module->setAttr("ascend.tile_mix_adjusted_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.adjustedCycles));
      module->setAttr("ascend.tile_mix_net_delta_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.netDeltaCycles));
      module->setAttr("ascend.tile_mix_boundary_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.boundaryCycles));
      module->setAttr("ascend.tile_mix_balance_penalty_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.balancePenaltyCycles));
      module->setAttr("ascend.tile_mix_handoff_relief_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.handoffReliefCycles));
      module->setAttr("ascend.tile_mix_workspace_relief_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.workspaceReliefCycles));
      module->setAttr("ascend.tile_mix_buffer_fit_penalty_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.bufferFitPenaltyCycles));
      module->setAttr("ascend.tile_mix_sync_frequency_penalty_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.syncFrequencyPenaltyCycles));
      module->setAttr("ascend.tile_mix_delta_gm_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.gmDeltaCycles));
      module->setAttr("ascend.tile_mix_delta_external_sync_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.externalSyncDeltaCycles));
      module->setAttr("ascend.tile_mix_delta_buffer_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.bufferDeltaCycles));
      module->setAttr("ascend.tile_mix_delta_pipeline_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.pipelineDeltaCycles));
      module->setAttr("ascend.tile_mix_delta_scalar_control_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.scalarControlDeltaCycles));
    }
    if (workspaceMultibufferStats.used) {
      module->setAttr("ascend.workspace_multibuffer_schedule_model",
                      StringAttr::get(module.getContext(), "ttir_finite_fifo_v1"));
      module->setAttr("ascend.workspace_multibuffer_model_valid",
                      BoolAttr::get(module.getContext(), workspaceMultibufferStats.valid));
      module->setAttr("ascend.workspace_multibuffer_adjustment_applied",
                      BoolAttr::get(module.getContext(), workspaceMultibufferStats.adjustmentApplied));
      module->setAttr("ascend.workspace_multibuffer_slots",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.requestedSlots));
      module->setAttr("ascend.workspace_multibuffer_reference_slots",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.referenceSlots));
      module->setAttr("ascend.workspace_multibuffer_slot_delta",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.slotDelta));
      module->setAttr("ascend.workspace_multibuffer_extra_slots",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.extraSlots));
      module->setAttr("ascend.workspace_multibuffer_family_count",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.workspaceFamilyCount));
      module->setAttr("ascend.workspace_multibuffer_cube_to_vector_family_count",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.cubeToVectorFamilyCount));
      module->setAttr("ascend.workspace_multibuffer_vector_to_cube_family_count",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.vectorToCubeFamilyCount));
      module->setAttr("ascend.workspace_multibuffer_bytes_per_slot",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.workspaceBytesPerSlot));
      module->setAttr("ascend.workspace_multibuffer_iteration_count",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.iterationCount));
      module->setAttr("ascend.workspace_multibuffer_cube_to_vector_iterations",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.cubeToVectorIterations));
      module->setAttr("ascend.workspace_multibuffer_vector_to_cube_iterations",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.vectorToCubeIterations));
      module->setAttr("ascend.workspace_multibuffer_cube_producer_tail_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.cubeProducerTailCycles));
      module->setAttr("ascend.workspace_multibuffer_vector_producer_tail_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.vectorProducerTailCycles));
      module->setAttr("ascend.workspace_multibuffer_sync_pair_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.syncPairCycles));
      module->setAttr("ascend.workspace_multibuffer_delta_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.syncDeltaCycles));
      module->setAttr("ascend.workspace_multibuffer_blocking_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.blockingCycles));
      module->setAttr("ascend.workspace_multibuffer_reference_blocking_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.referenceBlockingCycles));
      module->setAttr("ascend.workspace_multibuffer_producer_wait_relief_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.producerWaitReliefCycles));
      module->setAttr("ascend.workspace_multibuffer_reference_queue_penalty_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.referenceQueuePenaltyCycles));
      module->setAttr("ascend.workspace_multibuffer_overlap_relief_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.overlapReliefCycles));
      module->setAttr("ascend.workspace_multibuffer_queue_delta_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.queueDeltaCycles));
      module->setAttr("ascend.workspace_multibuffer_net_delta_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.netDeltaCycles));
      module->setAttr("ascend.workspace_multibuffer_skip_reason",
                      StringAttr::get(module.getContext(), workspaceMultibufferStats.skipReason));
    }
    if (dynamicCVStats.used) {
      module->setAttr("ascend.dynamic_cv_schedule_model",
                      StringAttr::get(
                          module.getContext(),
                          dynamicCVSegmentDagModelEnabled
                              ? "ttir_segment_dag_three_fifo_v3"
                              : "legacy_roofline_max_v1"));
      module->setAttr("ascend.dynamic_cv_segment_dag_model_enabled",
                      BoolAttr::get(module.getContext(),
                                    dynamicCVSegmentDagModelEnabled));
      module->setAttr("ascend.dynamic_cv_eligible",
                      BoolAttr::get(module.getContext(), dynamicCVStats.eligible));
      module->setAttr("ascend.dynamic_cv_adjustment_applied",
                      BoolAttr::get(module.getContext(), dynamicCVStats.adjustmentApplied));
      module->setAttr("ascend.dynamic_cv_work_item_count",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), dynamicCVStats.workItemCount));
      module->setAttr("ascend.dynamic_cv_segment_count",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), dynamicCVStats.segmentCount));
      module->setAttr("ascend.dynamic_cv_data_dependency_edges",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), dynamicCVStats.dataDependencyEdges));
      module->setAttr("ascend.dynamic_cv_segment_order_edges",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), dynamicCVStats.segmentOrderEdges));
      module->setAttr("ascend.dynamic_cv_cross_core_edges",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), dynamicCVStats.crossCoreEdges));
      module->setAttr("ascend.dynamic_cv_intra_capacity_edges",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), dynamicCVStats.intraCapacityEdges));
      module->setAttr("ascend.dynamic_cv_inter_capacity_edges",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), dynamicCVStats.interCapacityEdges));
      module->setAttr("ascend.dynamic_cv_load_capacity_edges",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), dynamicCVStats.loadCapacityEdges));
      module->setAttr("ascend.dynamic_cv_intra_blocking_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), dynamicCVStats.intraBlockingCycles));
      module->setAttr("ascend.dynamic_cv_inter_blocking_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), dynamicCVStats.interBlockingCycles));
      module->setAttr("ascend.dynamic_cv_load_blocking_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), dynamicCVStats.loadBlockingCycles));
      module->setAttr("ascend.dynamic_cv_iteration_count",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), dynamicCVStats.iterationCount));
      module->setAttr("ascend.dynamic_cv_original_makespan_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), dynamicCVStats.originalMakespanCycles));
      module->setAttr("ascend.dynamic_cv_transformed_makespan_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), dynamicCVStats.transformedMakespanCycles));
      module->setAttr("ascend.dynamic_cv_off_total_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), dynamicCVStats.offTotalCycles));
      module->setAttr("ascend.dynamic_cv_on_total_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), dynamicCVStats.onTotalCycles));
      module->setAttr("ascend.dynamic_cv_reference_delta_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), dynamicCVStats.referenceDeltaCycles));
      module->setAttr("ascend.dynamic_cv_sync_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), dynamicCVStats.syncCycles));
      module->setAttr("ascend.dynamic_cv_control_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), dynamicCVStats.controlCycles));
      module->setAttr("ascend.dynamic_cv_net_delta_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), dynamicCVStats.netDeltaCycles));
      module->setAttr("ascend.dynamic_cv_skip_reason",
                      StringAttr::get(module.getContext(), dynamicCVStats.skipReason));
      module->setAttr("ascend.dynamic_cv_status_source",
                      StringAttr::get(module.getContext(), dynamicCVStats.statusSource));
      module->setAttr("ascend.dynamic_cv_compiler_applied",
                      BoolAttr::get(module.getContext(), dynamicCVStats.compilerApplied));
    }
    if (tileMixStats.used) {
      module->setAttr("ascend.tile_mix_sync_ops_before",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.syncOpsBefore));
      module->setAttr("ascend.tile_mix_sync_ops_after",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.syncOpsAfter));
      module->setAttr("ascend.tile_mix_cube_segments",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.cubeSegmentCount));
      module->setAttr("ascend.tile_mix_vector_segments",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.vectorSegmentCount));
      module->setAttr("ascend.tile_mix_cube_loop_trip",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.cubeLoopTrip));
      module->setAttr("ascend.tile_mix_vector_loop_trip",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.vectorLoopTrip));
      module->setAttr("ascend.tile_mix_cube_layout_ops",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.cubeLayoutOpCount));
      module->setAttr("ascend.tile_mix_vector_layout_ops",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.vectorLayoutOpCount));
      module->setAttr("ascend.tile_mix_cube_workspace_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.cubeWorkspaceBytes));
      module->setAttr("ascend.tile_mix_vector_workspace_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.vectorWorkspaceBytes));
      module->setAttr("ascend.tile_mix_cube_subtile_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.cubeSubtileBytes));
      module->setAttr("ascend.tile_mix_vector_subtile_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.vectorSubtileBytes));
      module->setAttr("ascend.tile_mix_cube_target_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.cubeTargetBytes));
      module->setAttr("ascend.tile_mix_vector_target_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.vectorTargetBytes));
      if (tileMixStats.inferredTileM > 0) {
        module->setAttr("ascend.tile_mix_block_m",
                        IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.inferredTileM));
        module->setAttr("ascend.tile_mix_inferred_block_m",
                        IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.inferredTileM));
      }
      if (tileMixStats.inferredTileN > 0) {
        module->setAttr("ascend.tile_mix_block_n",
                        IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.inferredTileN));
        module->setAttr("ascend.tile_mix_inferred_block_n",
                        IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.inferredTileN));
      }
      module->setAttr("ascend.tile_mix_tile_shape_source",
                      StringAttr::get(module.getContext(), tileMixStats.tileShapeSource));
      module->setAttr("ascend.tile_mix_dtype_source",
                      StringAttr::get(module.getContext(), tileMixStats.dtypeSource));
      module->setAttr("ascend.tile_mix_handoff_source",
                      StringAttr::get(module.getContext(), tileMixStats.handoffSource));
      module->setAttr("ascend.tile_mix_intermediate_source",
                      StringAttr::get(module.getContext(), tileMixStats.intermediateSource));
      module->setAttr("ascend.tile_mix_handoff_feature_dim",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.handoffFeatureDim));
      module->setAttr("ascend.tile_mix_handoff_dtype_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.handoffDtypeBytes));
      module->setAttr("ascend.tile_mix_handoff_tile_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.handoffTileBytes));
      module->setAttr("ascend.tile_mix_handoff_subtile_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.handoffSubtileBytes));
      module->setAttr("ascend.tile_mix_handoff_segments",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.handoffSegmentCount));
      module->setAttr("ascend.tile_mix_handoff_target_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.handoffTargetBytes));
      module->setAttr("ascend.tile_mix_handoff_neutral_block_n",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.handoffNeutralBlockN));
      module->setAttr("ascend.tile_mix_intermediate_tile_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.intermediateTileBytes));
      module->setAttr("ascend.tile_mix_intermediate_target_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.intermediateTargetBytes));
      module->setAttr("ascend.tile_mix_intermediate_neutral_block_m",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.intermediateNeutralBlockM));
      module->setAttr("ascend.tile_mix_intermediate_pressure_penalty_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.intermediatePressurePenaltyCycles));
      module->setAttr("ascend.tile_mix_loop_granularity_relief_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.loopGranularityReliefCycles));
      module->setAttr("ascend.tile_mix_loop_mismatch_penalty_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.loopMismatchPenaltyCycles));
    }
    if (tileMixParams.vectorLoop > 0) {
      module->setAttr("ascend.tile_mix_vector_loop",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixParams.vectorLoop));
    }
    if (tileMixParams.cubeLoop > 0) {
      module->setAttr("ascend.tile_mix_cube_loop",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixParams.cubeLoop));
    }
    module->setAttr("ascend.scheduled_cycles",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64), rooflineTotalCycles));
    module->setAttr("ascend.simple_sum_cycles",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64), simpleSumCycles));
    
    // Print results
    llvm::outs() << "\n=== Pipeline Analysis (" << config.getName() << ") ===\n";
    llvm::outs() << "One iteration cycles (scheduled): " << oneIterCycles << "\n";
    llvm::outs() << "\nPer-HW-Unit cycles (with loops):\n";
    llvm::outs() << "  Cube path:\n";
    llvm::outs() << "    Cube compute: " << hwUnitCycles[HWUnit::Cube] << "\n";
    llvm::outs() << "    CubeMTE2 (load): " << hwUnitCycles[HWUnit::CubeMTE2] << "\n";
    llvm::outs() << "    FixPipe (store): " << hwUnitCycles[HWUnit::FixPipe] << "\n";
    llvm::outs() << "    Path total (max): " << cubePathCycles << "\n";
    llvm::outs() << "  Vector path:\n";
    llvm::outs() << "    Vector compute: " << hwUnitCycles[HWUnit::Vector] << "\n";
    llvm::outs() << "    VecMTE2 (load): " << hwUnitCycles[HWUnit::VecMTE2] << "\n";
    llvm::outs() << "    MTE3 (store): " << hwUnitCycles[HWUnit::MTE3] << "\n";
    llvm::outs() << "    Path total (max): " << vectorPathCycles << "\n";
    llvm::outs() << "\nTotal cycles:\n";
    llvm::outs() << "  Simple sum (no overlap): " << simpleSumCycles 
                 << " (" << llvm::format("%.3f", config.cyclesToMicroseconds(simpleSumCycles)) << " us)\n";
    llvm::outs() << "  Roofline base (with overlap): " << baseRooflineTotalCycles
                 << " (" << llvm::format("%.3f", config.cyclesToMicroseconds(baseRooflineTotalCycles)) << " us)\n";
    llvm::outs() << "  CV feature mode: " << cvFeatureModeName(cvFeatureMode)
                 << "\n";
    if (tileMixStats.used) {
      llvm::outs() << "  Tile mix params: vector_loop=" << tileMixParams.vectorLoop
                   << ", cube_loop=" << tileMixParams.cubeLoop
                   << "\n";
      llvm::outs() << "  Tile mix inferred features: block_m="
                   << tileMixStats.inferredTileM
                   << ", block_n=" << tileMixStats.inferredTileN
                   << ", tile_shape_source="
                   << tileMixStats.tileShapeSource
                   << ", dtype_source=" << tileMixStats.dtypeSource
                   << ", handoff_source=" << tileMixStats.handoffSource
                   << ", intermediate_source="
                   << tileMixStats.intermediateSource << "\n";
      llvm::outs() << "  Tile mix pass summary: source="
                   << tileMixStats.summarySource
                   << ", confidence=" << tileMixStats.confidencePercent
                   << "%"
                   << ", cube_applied="
                   << (tileMixStats.cubeApplied ? "true" : "false")
                   << ", vector_applied="
                   << (tileMixStats.vectorApplied ? "true" : "false")
                   << ", cube_skip_reason=" << tileMixStats.cubeSkipReason
                   << ", vector_skip_reason=" << tileMixStats.vectorSkipReason
                   << "\n";
      llvm::outs() << "  Tile mix eligibility: "
                   << (tileMixStats.valid ? "valid" : "fallback")
                   << ", cube_segments=" << tileMixStats.cubeSegmentCount
                   << ", vector_segments=" << tileMixStats.vectorSegmentCount
                   << ", cube_loop_trip=" << tileMixStats.cubeLoopTrip
                   << ", vector_loop_trip=" << tileMixStats.vectorLoopTrip
                   << "\n";
      llvm::outs() << "  Tile mix layout proxies: cube_layout_ops="
                   << tileMixStats.cubeLayoutOpCount
                   << ", vector_layout_ops="
                   << tileMixStats.vectorLayoutOpCount << "\n";
      llvm::outs() << "  Tile mix buffer fit: cube_workspace_bytes="
                   << tileMixStats.cubeWorkspaceBytes
                   << ", vector_workspace_bytes="
                   << tileMixStats.vectorWorkspaceBytes
                   << ", cube_subtile_bytes="
                   << tileMixStats.cubeSubtileBytes
                   << ", vector_subtile_bytes="
                   << tileMixStats.vectorSubtileBytes
                   << ", cube_target_bytes="
                   << tileMixStats.cubeTargetBytes
                   << ", vector_target_bytes="
                   << tileMixStats.vectorTargetBytes << "\n";
      llvm::outs() << "  Tile mix handoff footprint: feature_dim="
                   << tileMixStats.handoffFeatureDim
                   << ", dtype_bytes="
                   << tileMixStats.handoffDtypeBytes
                   << ", tile_bytes="
                   << tileMixStats.handoffTileBytes
                   << ", subtile_bytes="
                   << tileMixStats.handoffSubtileBytes
                   << ", segments="
                   << tileMixStats.handoffSegmentCount
                   << ", target_bytes="
                   << tileMixStats.handoffTargetBytes
                   << ", neutral_block_n="
                   << tileMixStats.handoffNeutralBlockN << "\n";
      llvm::outs() << "  Tile mix intermediate footprint: tile_bytes="
                   << tileMixStats.intermediateTileBytes
                   << ", target_bytes="
                   << tileMixStats.intermediateTargetBytes
                   << ", neutral_block_m="
                   << tileMixStats.intermediateNeutralBlockM << "\n";
    }
    if (workspaceMultibufferStats.used) {
      llvm::outs() << "  Workspace multibuffer params: requested_slots="
                   << workspaceMultibufferParams.requestedSlots << "\n";
      llvm::outs() << "  Workspace multibuffer: slots="
                   << workspaceMultibufferStats.requestedSlots
                   << ", reference_slots="
                   << workspaceMultibufferStats.referenceSlots
                   << ", slot_delta="
                   << workspaceMultibufferStats.slotDelta
                   << ", extra_slots="
                   << workspaceMultibufferStats.extraSlots
                   << ", workspace_families="
                   << workspaceMultibufferStats.workspaceFamilyCount
                   << ", cube_to_vector_families="
                   << workspaceMultibufferStats.cubeToVectorFamilyCount
                   << ", vector_to_cube_families="
                   << workspaceMultibufferStats.vectorToCubeFamilyCount
                   << ", bytes_per_slot="
                   << workspaceMultibufferStats.workspaceBytesPerSlot
                   << ", iterations="
                   << workspaceMultibufferStats.iterationCount
                   << ", cube_to_vector_iterations="
                   << workspaceMultibufferStats.cubeToVectorIterations
                   << ", vector_to_cube_iterations="
                   << workspaceMultibufferStats.vectorToCubeIterations
                   << ", cube_producer_tail_cycles="
                   << workspaceMultibufferStats.cubeProducerTailCycles
                   << ", vector_producer_tail_cycles="
                   << workspaceMultibufferStats.vectorProducerTailCycles
                   << ", sync_pair_cycles="
                   << workspaceMultibufferStats.syncPairCycles
                   << ", sync_delta_cycles="
                   << workspaceMultibufferStats.syncDeltaCycles
                   << ", blocking_cycles="
                   << workspaceMultibufferStats.blockingCycles
                   << ", reference_blocking_cycles="
                   << workspaceMultibufferStats.referenceBlockingCycles
                   << ", producer_wait_relief_cycles="
                   << workspaceMultibufferStats.producerWaitReliefCycles
                   << ", reference_queue_penalty_cycles="
                   << workspaceMultibufferStats.referenceQueuePenaltyCycles
                   << ", overlap_relief_cycles="
                   << workspaceMultibufferStats.overlapReliefCycles
                   << ", queue_delta_cycles="
                   << workspaceMultibufferStats.queueDeltaCycles
                    << ", net_delta_cycles="
                    << workspaceMultibufferStats.netDeltaCycles
                    << ", skip_reason="
                    << workspaceMultibufferStats.skipReason << "\n";
      llvm::outs() << "  Workspace multibuffer marginal cycles: sync="
                   << workspaceMultibufferStats.syncDeltaCycles
                   << ", queue=" << workspaceMultibufferStats.queueDeltaCycles
                   << ", relief=" << workspaceMultibufferStats.overlapReliefCycles
                    << ", net=" << workspaceMultibufferStats.netDeltaCycles
                    << "\n";
    }
    if (dynamicCVStats.used) {
      llvm::outs() << "  Dynamic CV segment DAG model: "
                   << (dynamicCVSegmentDagModelEnabled ? "enabled"
                                                       : "disabled")
                   << "\n";
      llvm::outs() << "  Dynamic CV params: intra_cache_num="
                   << dynamicCVStats.intraCacheCount
                   << ", inter_cache_num=" << dynamicCVStats.interCacheCount
                   << ", load_cache_num=" << dynamicCVStats.loadCacheCount
                   << ", target_supported="
                   << (dynamicCVParams.targetSupported ? "true" : "false")
                   << "\n";
      llvm::outs() << "  Dynamic CV schedule: eligible="
                   << (dynamicCVStats.eligible ? "true" : "false")
                   << ", status_source=" << dynamicCVStats.statusSource
                   << ", compiler_applied="
                   << (dynamicCVStats.compilerApplied ? "true" : "false")
                   << ", skip_reason=" << dynamicCVStats.skipReason
                   << ", segment_count=" << dynamicCVStats.segmentCount
                   << ", data_dependency_edges="
                   << dynamicCVStats.dataDependencyEdges
                   << ", segment_order_edges="
                   << dynamicCVStats.segmentOrderEdges
                   << ", work_items=" << dynamicCVStats.workItemCount
                   << ", cross_core_edges=" << dynamicCVStats.crossCoreEdges
                   << ", intra_capacity_edges="
                   << dynamicCVStats.intraCapacityEdges
                   << ", inter_capacity_edges="
                   << dynamicCVStats.interCapacityEdges
                   << ", load_capacity_edges="
                   << dynamicCVStats.loadCapacityEdges
                   << ", intra_blocking_cycles="
                   << dynamicCVStats.intraBlockingCycles
                   << ", inter_blocking_cycles="
                   << dynamicCVStats.interBlockingCycles
                   << ", load_blocking_cycles="
                   << dynamicCVStats.loadBlockingCycles
                   << ", iterations=" << dynamicCVStats.iterationCount
                   << ", cache_bytes=" << dynamicCVStats.cacheBytes
                   << ", original_makespan_cycles="
                   << dynamicCVStats.originalMakespanCycles
                   << ", transformed_makespan_cycles="
                   << dynamicCVStats.transformedMakespanCycles
                   << ", off_makespan_cycles="
                   << dynamicCVStats.offMakespanCycles
                   << ", on_makespan_cycles="
                   << dynamicCVStats.onMakespanCycles
                   << ", off_total_cycles=" << dynamicCVStats.offTotalCycles
                   << ", on_total_cycles=" << dynamicCVStats.onTotalCycles
                   << ", reference_delta_cycles="
                   << dynamicCVStats.referenceDeltaCycles
                   << ", sync_cycles=" << dynamicCVStats.syncCycles
                   << ", control_cycles=" << dynamicCVStats.controlCycles
                   << ", dynamic_cv_net_delta_cycles="
                   << dynamicCVStats.netDeltaCycles
                   << ", net_delta_cycles=" << dynamicCVStats.netDeltaCycles
                   << "\n";
    }
    if (tileMixStats.used) {
      llvm::outs() << "  Tile mix delta cycles: boundary="
                   << tileMixStats.boundaryCycles
                   << ", balance_penalty="
                   << tileMixStats.balancePenaltyCycles
                   << ", handoff_relief="
                   << tileMixStats.handoffReliefCycles
                   << ", loop_granularity_relief="
                   << tileMixStats.loopGranularityReliefCycles
                   << ", workspace_relief="
                   << tileMixStats.workspaceReliefCycles
                   << ", buffer_fit_penalty="
                   << tileMixStats.bufferFitPenaltyCycles
                   << ", intermediate_pressure_penalty="
                   << tileMixStats.intermediatePressurePenaltyCycles
                   << ", loop_mismatch_penalty="
                   << tileMixStats.loopMismatchPenaltyCycles
                    << ", sync_frequency_penalty="
                    << tileMixStats.syncFrequencyPenaltyCycles
                    << ", net_delta=" << tileMixStats.netDeltaCycles << "\n";
      llvm::outs() << "  Tile mix marginal cycles: gm="
                   << tileMixStats.gmDeltaCycles
                   << ", external_sync="
                   << tileMixStats.externalSyncDeltaCycles
                   << ", buffer=" << tileMixStats.bufferDeltaCycles
                   << ", pipeline=" << tileMixStats.pipelineDeltaCycles
                   << ", scalar_control="
                   << tileMixStats.scalarControlDeltaCycles << "\n";
    }
    llvm::outs() << "  Combined feature model delta cycles: tile_mix="
                  << tileMixDeltaCycles << ", workspace_multibuffer="
                  << workspaceMultibufferDeltaCycles
                  << ", dynamic_cv_off_reference="
                  << dynamicCVReferenceDeltaCycles << ", dynamic_cv="
                  << dynamicCVDeltaCycles << "\n";
    llvm::outs() << "  Roofline model (TTIR principle marginal tile mix): " << rooflineTotalCycles
                 << " (" << llvm::format("%.3f", config.cyclesToMicroseconds(rooflineTotalCycles)) << " us)\n";
    llvm::outs() << "  Kernel launch overhead: " << launch.totalCycles
                 << " ("
                 << llvm::format("%.3f",
                                 config.cyclesToMicroseconds(launch.totalCycles))
                 << " us)";
    if (launch.blockDim > 0)
      llvm::outs() << ", block_dim=" << launch.blockDim;
    if (launch.numWaves > 0)
      llvm::outs() << ", waves=" << launch.numWaves;
    llvm::outs() << "\n";
    llvm::outs() << "  Predicted total: " << predictedTotalCycles
                 << " ("
                 << llvm::format(
                        "%.3f",
                        config.cyclesToMicroseconds(predictedTotalCycles))
                 << " us)\n";
    llvm::outs() << "  Speedup from overlap: " << llvm::format("%.2fx", 
                    static_cast<double>(simpleSumCycles) / std::max(rooflineTotalCycles, 1L)) << "\n";
    
    llvm::outs() << "\n=== Pipeline Timeline (one iteration) ===\n";
    scheduler.printTimeline(llvm::outs());
    llvm::outs() << "\n=== Utilization Report ===\n";
    scheduler.printUtilizationReport(llvm::outs());
    
    // Generate Perfetto trace only when an explicit path is provided.
    // Replaces the old hard-coded "pipeline_trace.json" cwd write.
    if (!perfettoTraceFile.empty()) {
      generatePerfettoTrace(scheduler, perfettoTraceFile,
                            oneIterCycles, rooflineTotalCycles,
                            launch.totalCycles, predictedTotalCycles);
    }

    // Emit dependency graph JSON for downstream performance bound model consumers
    // (perfbound/model/serialization.py mandatory/avoidable split)
    // Only writes when an explicit output path is provided — never pollutes cwd.
    if (!dependencyGraphFile.empty()) {
      std::error_code depEC;
      llvm::raw_fd_ostream depFile(dependencyGraphFile, depEC,
                                    llvm::sys::fs::OF_Text);
      if (!depEC) {
        scheduler.emitDependencyGraphJSON(depFile);
        llvm::outs() << "Dependency graph: " << dependencyGraphFile << "\n";
      } else {
        llvm::errs() << "Warning: could not write " << dependencyGraphFile
                     << ": " << depEC.message() << "\n";
      }
    }
  }
};

} // namespace
} // namespace ascend
} // namespace mlir
