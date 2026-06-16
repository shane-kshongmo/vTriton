//===- HIVMBottleneckDiagnosis.cpp - HIVM bottleneck diagnosis -*- C++ -*-===//
//
// Implements per-op, per-pipeline, and global bottleneck diagnosis for
// HIVM IR scheduling results.  Every suggestion is accompanied by the
// specific HIVMAnalysisReport field(s) that motivated it.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Analysis/HIVMBottleneckDiagnosis.h"
#include "AscendModel/Analysis/HIVMAnalysis.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace mlir {
namespace ascend {

static std::string formatDouble(double val, const char *fmt) {
  char buf[32];
  snprintf(buf, sizeof(buf), fmt, val);
  return buf;
}

static llvm::StringRef pipeName(HIVMPipe pipe) {
  return HIVMAnalyzer::stringifyPipe(pipe);
}

static double safeDivide(int64_t a, int64_t b) {
  return b > 0 ? static_cast<double>(a) / static_cast<double>(b) : 0.0;
}

static std::string fmtCycles(int64_t c) { return std::to_string(c) + " cyc"; }

llvm::StringRef HIVMBottleneckDiagnoser::stringifyBottleneckType(BottleneckType type) {
  switch (type) {
  case BottleneckType::BandwidthBound: return "BandwidthBound";
  case BottleneckType::ComputeBound: return "ComputeBound";
  case BottleneckType::StartupOverhead: return "StartupOverhead";
  case BottleneckType::SyncOverhead: return "SyncOverhead";
  case BottleneckType::PipelineImbalance: return "PipelineImbalance";
  case BottleneckType::LowParallelism: return "LowParallelism";
  }
  return "Unknown";
}

HIVMBottleneckDiagnoser::HIVMBottleneckDiagnoser(const HardwareConfig &config)
    : config(config) {}

int64_t HIVMBottleneckDiagnoser::computeTheoreticalMin(const HIVMOp &op) const {
  if (op.bytes <= 0 && op.elements <= 0)
    return 0;

  switch (op.pipe) {
  case HIVMPipe::VectorMTE2:
  case HIVMPipe::CubeMTE2: {
    if (op.bytes <= 0) return 0;
    llvm::StringRef src = op.srcSpace;
    double bw = config.getMemoryBandwidthBytesPerCycle(src);
    if (bw <= 0.0) bw = config.getMemoryBandwidthBytesPerCycle("gm");
    if (bw <= 0.0) bw = static_cast<double>(config.getVectorWidthBytes());
    return static_cast<int64_t>(std::ceil(static_cast<double>(op.bytes) / bw));
  }
  case HIVMPipe::MTE3: {
    if (op.bytes <= 0) return 0;
    double bw = config.getMemoryBandwidthBytesPerCycle("gm");
    if (bw <= 0.0) bw = static_cast<double>(config.getVectorWidthBytes());
    return static_cast<int64_t>(std::ceil(static_cast<double>(op.bytes) / bw));
  }
  case HIVMPipe::Vector: {
    if (op.elements <= 0) return 0;
    int64_t vecWidthElems = config.getVectorWidthBytes() / 4;
    int64_t vecStartup = config.getVectorStartupLatency();
    return vecStartup + static_cast<int64_t>(
        std::ceil(static_cast<double>(op.elements) / static_cast<double>(vecWidthElems)));
  }
  case HIVMPipe::Cube: {
    return config.getCubeStartupLatency() +
           config.estimateCubeCycles(op.elements > 0 ? op.elements : 16, 16, 16);
  }
  case HIVMPipe::FixPipe: {
    if (op.bytes <= 0) return 0;
    return config.getFixPipeStartupLatency() +
           static_cast<int64_t>(std::ceil(static_cast<double>(op.bytes) /
                                          static_cast<double>(config.getVectorWidthBytes())));
  }
  default:
    return op.duration;
  }
}

OpDiagnosis HIVMBottleneckDiagnoser::diagnoseOp(const HIVMOp &op) const {
  OpDiagnosis diag;
  diag.opId = op.id;
  diag.lineNumber = op.lineNumber;
  diag.opName = op.opName;
  diag.pipe = op.pipe;
  diag.actualCycles = op.duration;
  diag.theoreticalMinCycles = computeTheoreticalMin(op);
  diag.overheadRatio = safeDivide(op.duration - diag.theoreticalMinCycles, op.duration);

  bool isTransfer = (op.pipe == HIVMPipe::VectorMTE2 ||
                     op.pipe == HIVMPipe::CubeMTE2 ||
                     op.pipe == HIVMPipe::MTE3);
  bool isCompute = (op.pipe == HIVMPipe::Vector ||
                    op.pipe == HIVMPipe::Cube ||
                    op.pipe == HIVMPipe::MTE1);
  int64_t startup = 0;
  switch (op.pipe) {
  case HIVMPipe::Vector: startup = config.getVectorStartupLatency(); break;
  case HIVMPipe::VectorMTE2: case HIVMPipe::CubeMTE2: startup = config.getMTE2StartupLatency(); break;
  case HIVMPipe::MTE3: startup = config.getMTE3StartupLatency(); break;
  case HIVMPipe::Cube: startup = config.getCubeStartupLatency(); break;
  case HIVMPipe::FixPipe: startup = config.getFixPipeStartupLatency(); break;
  default: startup = 1;
  }

  // --- Sync ops ---
  if (op.isSyncOp || op.isBarrier) {
    diag.rootCause = BottleneckType::SyncOverhead;
    diag.evidence = "[isSyncOp=" + std::string(op.isSyncOp ? "true" : "false") +
                    ", isBarrier=" + std::string(op.isBarrier ? "true" : "false") +
                    ", duration=" + fmtCycles(op.duration) + "]";
    diag.explanation = op.opName + " on " + pipeName(op.pipe).str() +
                       ": pure synchronization (duration=" + fmtCycles(op.duration) +
                       " carries no data/compute work)";
    if (op.opName == "pipe_barrier" && op.pipe == HIVMPipe::All) {
      diag.suggestions.push_back(
          "Replace PIPE_ALL barrier with per-pipe set_flag/wait_flag pairs"
          " [reason: isBarrier=true, pipe=PIPE_ALL, duration=" +
          fmtCycles(op.duration) + " blocks all pipes simultaneously]");
      diag.suggestions.push_back(
          "Enable multi-buffer pipelining to reduce barrier frequency"
          " [reason: isBarrier=true means entire core stalls; multi-buffer hides this]");
    } else if (op.opName == "wait_flag") {
      diag.suggestions.push_back(
          "Consider reordering ops to reduce wait stall time"
          " [reason: isSyncOp=true, duration=" + fmtCycles(op.duration) +
          " spent idle waiting for producer pipe]");
    }
    return diag;
  }

  // --- Transfer ops (MTE2/MTE3) ---
  if (isTransfer && op.bytes > 0) {
    int64_t transferOnly = op.duration - startup;
    if (transferOnly <= 0 || startup > transferOnly) {
      diag.rootCause = BottleneckType::StartupOverhead;
      diag.evidence = "[duration=" + fmtCycles(op.duration) +
                       ", startup_latency=" + fmtCycles(startup) +
                       ", bytes=" + std::to_string(op.bytes) +
                       ", startup/duration=" +
                       formatDouble(safeDivide(startup, op.duration) * 100.0, "%.1f%%") + "]";
      diag.explanation = op.opName + " on " + pipeName(op.pipe).str() +
                          ": startup_latency (" + fmtCycles(startup) +
                          ") > transfer_work (" + fmtCycles(transferOnly) +
                          "), bytes=" + std::to_string(op.bytes) +
                          " too small to amortize startup";
      diag.suggestions.push_back(
          "Increase tile size to amortize startup latency"
          " [reason: startup_latency=" + fmtCycles(startup) +
          " > transfer_work; larger bytes makes startup占比 smaller]");
      diag.suggestions.push_back(
          "Fuse adjacent transfers to reduce per-transfer overhead"
          " [reason: each transfer pays " + fmtCycles(startup) +
          " startup; fewer transfers = fewer startups]");
    } else {
      diag.rootCause = BottleneckType::BandwidthBound;
      diag.evidence = "[duration=" + fmtCycles(op.duration) +
                       ", transfer_cycles=" + fmtCycles(transferOnly) +
                       ", bytes=" + std::to_string(op.bytes) +
                       ", src_space=" + op.srcSpace +
                       ", dst_space=" + op.dstSpace + "]";
      diag.explanation = op.opName + " on " + pipeName(op.pipe).str() +
                          ": bandwidth-limited, bytes=" + std::to_string(op.bytes) +
                          " from " + op.srcSpace + " to " + op.dstSpace +
                          " takes " + fmtCycles(transferOnly) + " for transfer alone";
      diag.suggestions.push_back(
          "Reduce data movement via in-place computation or tiling reuse"
          " [reason: bytes=" + std::to_string(op.bytes) +
          " dominates duration; fewer bytes → fewer transfer cycles]");
      diag.suggestions.push_back(
          "Overlap transfer with compute via multi-buffer pipelining"
          " [reason: transfer_cycles=" + fmtCycles(transferOnly) +
          " on dedicated pipe; can overlap with compute on other pipes]");
      if (op.pipe == HIVMPipe::CubeMTE2)
        diag.suggestions.push_back(
            "Prefetch data into L1 before Cube compute starts"
            " [reason: dst_space=" + op.dstSpace +
            " = L1; early prefetch hides CubeMTE2 latency behind Cube compute]");
    }
    return diag;
  }

  // --- Compute ops (Vector/Cube) ---
  if (isCompute) {
    int64_t computeOnly = op.duration - startup;
    if (startup > computeOnly && computeOnly > 0) {
      diag.rootCause = BottleneckType::StartupOverhead;
      diag.evidence = "[duration=" + fmtCycles(op.duration) +
                       ", startup_latency=" + fmtCycles(startup) +
                       ", compute_work=" + fmtCycles(computeOnly) +
                       ", elements=" + std::to_string(op.elements) + "]";
      diag.explanation = op.opName + " on " + pipeName(op.pipe).str() +
                          ": startup_latency (" + fmtCycles(startup) +
                          ") > compute_work (" + fmtCycles(computeOnly) +
                          "), elements=" + std::to_string(op.elements) +
                          " too few to amortize startup";
      diag.suggestions.push_back(
          "Increase tile size to amortize startup latency"
          " [reason: startup_latency=" + fmtCycles(startup) +
          " > compute_work; more elements dilutes startup占比]");
    } else {
      diag.rootCause = BottleneckType::ComputeBound;
      diag.evidence = "[duration=" + fmtCycles(op.duration) +
                       ", elements=" + std::to_string(op.elements) +
                       ", flops=" + std::to_string(op.flops) +
                       ", compute_work=" + fmtCycles(computeOnly) + "]";
      diag.explanation = op.opName + " on " + pipeName(op.pipe).str() +
                          ": compute-heavy, elements=" + std::to_string(op.elements) +
                          ", duration=" + fmtCycles(op.duration) +
                          " (compute_work=" + fmtCycles(computeOnly) + ")";
      if (op.pipe == HIVMPipe::Cube)
        diag.suggestions.push_back(
            "Increase K tile dimension to improve arithmetic intensity"
            " [reason: Cube pipe compute_work=" + fmtCycles(computeOnly) +
            " dominates; larger K → more MACs per data load]");
      else
        diag.suggestions.push_back(
            "Fuse with adjacent vector ops to reduce kernel launch overhead"
            " [reason: Vector compute_work=" + fmtCycles(computeOnly) +
            " per standalone op; fusion amortizes per-op startup]");
      diag.suggestions.push_back(
          "Overlap with MTE2 prefetch via software pipelining"
          " [reason: compute on " + pipeName(op.pipe).str() +
          " can run while MTE2 loads next tile on different pipe]");
    }
    return diag;
  }

  // --- FixPipe ---
  if (op.pipe == HIVMPipe::FixPipe && op.bytes > 0) {
    diag.rootCause = BottleneckType::BandwidthBound;
    diag.evidence = "[duration=" + fmtCycles(op.duration) +
                    ", bytes=" + std::to_string(op.bytes) +
                    ", dst_space=" + op.dstSpace + "]";
    diag.explanation = "fixpipe: draining " + std::to_string(op.bytes) +
                       " bytes from L0C to " + op.dstSpace +
                       " (duration=" + fmtCycles(op.duration) + ")";
    diag.suggestions.push_back(
        "Increase tile size to fill Cube pipeline before draining"
        " [reason: bytes=" + std::to_string(op.bytes) +
        " per drain; larger tile → more useful compute per drain cycle]");
    return diag;
  }

  diag.rootCause = BottleneckType::LowParallelism;
  diag.evidence = "[duration=" + fmtCycles(op.duration) +
                  ", pipe=" + pipeName(op.pipe).str() + "]";
  diag.explanation = op.opName + " on " + pipeName(op.pipe).str() +
                     ": unclassified, duration=" + fmtCycles(op.duration);
  return diag;
}

PipelineDiagnosis HIVMBottleneckDiagnoser::diagnosePipeline(
    const HIVMAnalysisReport &report) const {
  PipelineDiagnosis diag;

  if (report.weightedPipeCycles.empty()) {
    diag.rootCause = BottleneckType::LowParallelism;
    diag.explanation = "No pipe activity recorded";
    return diag;
  }

  int64_t maxWeighted = 0;
  int64_t minWeighted = std::numeric_limits<int64_t>::max();
  HIVMPipe maxPipe = HIVMPipe::Unknown;
  HIVMPipe minPipe = HIVMPipe::Unknown;

  for (const auto &entry : report.weightedPipeCycles) {
    if (entry.second > maxWeighted) {
      maxWeighted = entry.second;
      maxPipe = entry.first;
    }
    if (entry.second < minWeighted && entry.second > 0) {
      minWeighted = entry.second;
      minPipe = entry.first;
    }
  }

  diag.bottleneckPipe = maxPipe;
  diag.imbalanceRatio = safeDivide(maxWeighted, minWeighted);

  double maxUtil = 0.0;
  double minUtil = 100.0;
  for (const auto &entry : report.pipeBusyCycles) {
    double util = safeDivide(entry.second, report.oneIterationCycles) * 100.0;
    if (util > maxUtil) maxUtil = util;
    if (util < minUtil && util > 0.0) minUtil = util;
  }

  if (diag.imbalanceRatio > 3.0) {
    diag.rootCause = BottleneckType::PipelineImbalance;
    diag.evidence = "[weightedPipeCycles[" + pipeName(maxPipe).str() +
                    "=" + std::to_string(maxWeighted) +
                    ", weightedPipeCycles[" + pipeName(minPipe).str() +
                    "=" + std::to_string(minWeighted) +
                    ", ratio=" + formatDouble(diag.imbalanceRatio, "%.1f") + "x]";
    diag.explanation = pipeName(maxPipe).str() + " weightedPipeCycles=" +
                       std::to_string(maxWeighted) +
                       " vs " + pipeName(minPipe).str() +
                       " weightedPipeCycles=" + std::to_string(minWeighted) +
                       " (ratio " + formatDouble(diag.imbalanceRatio, "%.1f") + "x > 3.0x threshold)";
    diag.suggestions.push_back(
        "Overlap " + pipeName(maxPipe).str() + " work with " +
        pipeName(minPipe).str() + " prefetch via multi-buffer pipelining"
        " [reason: weightedPipeCycles imbalance ratio=" +
        formatDouble(diag.imbalanceRatio, "%.1f") +
        "x; multi-buffer lets slow pipe prefetch while busy pipe computes]");
    diag.suggestions.push_back(
        "Consider Cube-Vector split to exploit dual-core parallelism"
        " [reason: weightedPipeCycles shows single-core imbalance; AIC+AIV dual-core can distribute load]");
  } else if (maxUtil > 0.0 && minUtil < 20.0 && maxUtil > 60.0) {
    diag.rootCause = BottleneckType::PipelineImbalance;
    diag.evidence = "[pipeBusyCycles utilization: max=" +
                    formatDouble(maxUtil, "%.1f%%") +
                    ", min=" + formatDouble(minUtil, "%.1f%%") + "]";
    diag.explanation = pipeName(maxPipe).str() + " utilization=" +
                       formatDouble(maxUtil, "%.1f%%") +
                       " vs " + pipeName(minPipe).str() +
                       " utilization=" + formatDouble(minUtil, "%.1f%%") +
                       " (one pipe saturated, another mostly idle)";
    diag.suggestions.push_back(
        "Increase software pipeline depth to fill idle pipe slots"
        " [reason: pipeBusyCycles shows min utilization=" +
        formatDouble(minUtil, "%.1f%%") +
        "; deeper pipeline fills more idle slots]");
  } else {
    diag.rootCause = BottleneckType::BandwidthBound;
    if (maxPipe == HIVMPipe::VectorMTE2 || maxPipe == HIVMPipe::CubeMTE2 ||
        maxPipe == HIVMPipe::MTE3) {
      diag.evidence = "[max weightedPipeCycles on " + pipeName(maxPipe).str() +
                      "=" + std::to_string(maxWeighted) + "]";
      diag.explanation = "Memory pipe " + pipeName(maxPipe).str() +
                          " has max weightedPipeCycles=" + std::to_string(maxWeighted);
      diag.suggestions.push_back(
          "Reduce total data movement or increase tile reuse factor"
          " [reason: weightedPipeCycles[" + pipeName(maxPipe).str() +
          "]=" + std::to_string(maxWeighted) +
          " is highest; data transfer dominates total workload]");
    } else if (maxPipe == HIVMPipe::Cube || maxPipe == HIVMPipe::Vector) {
      diag.rootCause = BottleneckType::ComputeBound;
      diag.evidence = "[max weightedPipeCycles on " + pipeName(maxPipe).str() +
                      "=" + std::to_string(maxWeighted) + "]";
      diag.explanation = "Compute pipe " + pipeName(maxPipe).str() +
                          " has max weightedPipeCycles=" + std::to_string(maxWeighted);
      diag.suggestions.push_back(
          "Increase arithmetic intensity (compute-to-data ratio)"
          " [reason: weightedPipeCycles[" + pipeName(maxPipe).str() +
          "]=" + std::to_string(maxWeighted) +
          " shows compute workload > data transfer workload]");
    } else {
      diag.evidence = "[max weightedPipeCycles on " + pipeName(maxPipe).str() +
                      "=" + std::to_string(maxWeighted) + "]";
      diag.explanation = "Bottleneck on " + pipeName(maxPipe).str() +
                          " weightedPipeCycles=" + std::to_string(maxWeighted);
    }
  }

  return diag;
}

BottleneckType HIVMBottleneckDiagnoser::diagnoseGlobal(
    const HIVMBottleneckReport &partial,
    const HIVMAnalysisReport &report) const {
  double syncRatio = safeDivide(report.syncCycles, report.oneIterationCycles) * 100.0;
  double barrierRatio = safeDivide(report.barrierCycles, report.oneIterationCycles) * 100.0;

  if (syncRatio > 20.0 || barrierRatio > 15.0)
    return BottleneckType::SyncOverhead;

  if (partial.pipeDiagnosis.rootCause == BottleneckType::PipelineImbalance)
    return BottleneckType::PipelineImbalance;

  auto maxWPCycles = std::max_element(
      report.weightedPipeCycles.begin(), report.weightedPipeCycles.end(),
      [](const auto &a, const auto &b) { return a.second < b.second; });

  if (maxWPCycles != report.weightedPipeCycles.end()) {
    HIVMPipe maxPipe = maxWPCycles->first;
    if (maxPipe == HIVMPipe::VectorMTE2 || maxPipe == HIVMPipe::CubeMTE2 ||
        maxPipe == HIVMPipe::MTE3)
      return BottleneckType::BandwidthBound;
    if (maxPipe == HIVMPipe::Cube || maxPipe == HIVMPipe::Vector)
      return BottleneckType::ComputeBound;
  }

  return partial.pipeDiagnosis.rootCause;
}

bool HIVMBottleneckDiagnoser::diagnose(const HIVMAnalysisReport &report,
                                       HIVMBottleneckReport &out) const {
  out = HIVMBottleneckReport();

  out.opDiagnoses.reserve(report.operations.size());
  for (const HIVMOp &op : report.operations) {
    if (op.duration <= 0) continue;
    auto diag = diagnoseOp(op);
    out.opDiagnoses.push_back(diag);
  }

  out.pipeDiagnosis = diagnosePipeline(report);

  out.syncOverheadRatio = safeDivide(report.syncCycles, report.oneIterationCycles) * 100.0;
  out.barrierOverheadRatio = safeDivide(report.barrierCycles, report.oneIterationCycles) * 100.0;

  out.globalRootCause = diagnoseGlobal(out, report);

  double syncRatio = out.syncOverheadRatio;
  double barrierRatio = out.barrierOverheadRatio;

  switch (out.globalRootCause) {
  case BottleneckType::SyncOverhead:
    out.globalEvidence = "[syncCycles=" + fmtCycles(report.syncCycles) +
                         ", barrierCycles=" + fmtCycles(report.barrierCycles) +
                         ", oneIterationCycles=" + fmtCycles(report.oneIterationCycles) +
                         ", syncCycles/oneIterationCycles=" +
                         formatDouble(syncRatio, "%.1f%%") +
                         ", barrierCycles/oneIterationCycles=" +
                         formatDouble(barrierRatio, "%.1f%%") + "]";
    out.globalExplanation = "syncCycles/oneIterationCycles=" +
                            formatDouble(syncRatio, "%.1f%%") +
                            ", barrierCycles/oneIterationCycles=" +
                            formatDouble(barrierRatio, "%.1f%%") +
                            " (thresholds: sync>20% or barrier>15%)";
    out.globalSuggestions.push_back(
        "Minimize global barriers; use set_flag/wait_flag for per-pipe sync"
        " [reason: barrierCycles=" + fmtCycles(report.barrierCycles) +
        " at " + formatDouble(barrierRatio, "%.1f%%") +
        " of oneIterationCycles; per-pipe sync avoids blocking all pipes]");
    out.globalSuggestions.push_back(
        "Enable multi-buffer pipelining to decouple producer/consumer"
        " [reason: syncCycles=" + fmtCycles(report.syncCycles) +
        " at " + formatDouble(syncRatio, "%.1f%%") +
        " of oneIterationCycles; multi-buffer reduces sync frequency]");
    break;
  case BottleneckType::PipelineImbalance:
    out.globalEvidence = out.pipeDiagnosis.evidence;
    out.globalExplanation = out.pipeDiagnosis.explanation;
    out.globalSuggestions = out.pipeDiagnosis.suggestions;
    break;
  case BottleneckType::BandwidthBound: {
    auto maxWP = std::max_element(
        report.weightedPipeCycles.begin(), report.weightedPipeCycles.end(),
        [](const auto &a, const auto &b) { return a.second < b.second; });
    std::string maxPipeName = maxWP != report.weightedPipeCycles.end()
                                  ? pipeName(maxWP->first).str()
                                  : "unknown";
    int64_t maxWPVal = maxWP != report.weightedPipeCycles.end() ? maxWP->second : 0;
    out.globalEvidence = "[max weightedPipeCycles on " + maxPipeName +
                         "=" + std::to_string(maxWPVal) + " (memory transfer pipe)]";
    out.globalExplanation = "weightedPipeCycles max on memory pipe " + maxPipeName +
                            "=" + std::to_string(maxWPVal) +
                            " — data movement dominates total workload";
    out.globalSuggestions.push_back(
        "Reduce data volume via tiling reuse or in-place computation"
        " [reason: weightedPipeCycles[" + maxPipeName +
        "]=" + std::to_string(maxWPVal) +
        " is highest; reducing bytes reduces transfer cycles]");
    out.globalSuggestions.push_back(
        "Overlap transfers with compute via software pipelining"
        " [reason: transfer on " + maxPipeName +
        " can overlap with compute on other pipes]");
    break;
  }
  case BottleneckType::ComputeBound: {
    auto maxWP = std::max_element(
        report.weightedPipeCycles.begin(), report.weightedPipeCycles.end(),
        [](const auto &a, const auto &b) { return a.second < b.second; });
    std::string maxPipeName = maxWP != report.weightedPipeCycles.end()
                                  ? pipeName(maxWP->first).str()
                                  : "unknown";
    int64_t maxWPVal = maxWP != report.weightedPipeCycles.end() ? maxWP->second : 0;
    out.globalEvidence = "[max weightedPipeCycles on " + maxPipeName +
                         "=" + std::to_string(maxWPVal) + " (compute pipe)]";
    out.globalExplanation = "weightedPipeCycles max on compute pipe " + maxPipeName +
                            "=" + std::to_string(maxWPVal) +
                            " — arithmetic throughput is the bottleneck";
    out.globalSuggestions.push_back(
        "Increase arithmetic intensity via larger K/M/N tiles"
        " [reason: weightedPipeCycles[" + maxPipeName +
        "]=" + std::to_string(maxWPVal) +
        " shows compute > data; larger tiles increases ops per byte loaded]");
    out.globalSuggestions.push_back(
        "Fuse ops to reduce kernel launch and sync overhead"
        " [reason: each standalone op pays startup and sync; fusion eliminates inter-op gaps]");
    break;
  }
  case BottleneckType::StartupOverhead:
    out.globalEvidence = "[multiple ops where startup_latency > compute/transfer_work]";
    out.globalExplanation = "Startup latency dominates many small ops — tile size too small";
    out.globalSuggestions.push_back(
        "Increase tile sizes to amortize startup costs"
        " [reason: startup > work on multiple ops; larger tiles dilutes startup占比]");
    break;
  default:
    out.globalEvidence = "";
    out.globalExplanation = "Insufficient data to determine root cause";
    break;
  }

  return true;
}

void HIVMBottleneckReport::print(llvm::raw_ostream &os,
                                 const HardwareConfig &config) const {
  os << "\n=== Bottleneck Diagnosis ===\n";
  os << "Global root cause: "
     << HIVMBottleneckDiagnoser::stringifyBottleneckType(globalRootCause) << "\n";
  os << "  Evidence: " << globalEvidence << "\n";
  os << "  " << globalExplanation << "\n";
  for (const auto &s : globalSuggestions)
    os << "  -> " << s << "\n";

  os << "\nSync/barrier overhead (from syncCycles/barrierCycles/oneIterationCycles):\n";
  os << "  syncCycles/oneIterationCycles = "
     << llvm::format("%.1f%%", syncOverheadRatio) << "\n";
  os << "  barrierCycles/oneIterationCycles = "
     << llvm::format("%.1f%%", barrierOverheadRatio) << "\n";

  os << "\nPipeline diagnosis:\n";
  os << "  Evidence: " << pipeDiagnosis.evidence << "\n";
  os << "  Bottleneck pipe: "
     << HIVMAnalyzer::stringifyPipe(pipeDiagnosis.bottleneckPipe) << "\n";
  os << "  Imbalance ratio (weightedPipeCycles max/min): "
     << llvm::format("%.1f", pipeDiagnosis.imbalanceRatio) << "x\n";
  os << "  " << pipeDiagnosis.explanation << "\n";
  for (const auto &s : pipeDiagnosis.suggestions)
    os << "  -> " << s << "\n";

  std::vector<const OpDiagnosis *> sorted;
  sorted.reserve(opDiagnoses.size());
  for (const OpDiagnosis &d : opDiagnoses)
    if (d.rootCause != BottleneckType::SyncOverhead)
      sorted.push_back(&d);
  std::sort(sorted.begin(), sorted.end(),
            [](const OpDiagnosis *lhs, const OpDiagnosis *rhs) {
              return lhs->actualCycles > rhs->actualCycles;
            });

  size_t limit = std::min<size_t>(10, sorted.size());
  os << "\nPer-op diagnosis (top " << limit << " by duration, excluding sync ops):\n";
  for (size_t i = 0; i < limit; ++i) {
    const OpDiagnosis *d = sorted[i];
    os << "  line " << d->lineNumber << " " << d->opName
       << " [" << pipeName(d->pipe) << "]: "
       << HIVMBottleneckDiagnoser::stringifyBottleneckType(d->rootCause) << "\n";
    os << "    Evidence: " << d->evidence << "\n";
    os << "    " << d->explanation << "\n";
    os << "    duration=" << d->actualCycles << " cyc, theoretical_min="
       << d->theoreticalMinCycles << " cyc, overhead="
       << llvm::format("%.1f%%", d->overheadRatio * 100.0) << "\n";
    for (const auto &s : d->suggestions)
      os << "    -> " << s << "\n";
  }
}

} // namespace ascend
} // namespace mlir