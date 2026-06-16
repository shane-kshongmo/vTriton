//===- HIVMBottleneckDiagnosis.h - HIVM bottleneck diagnosis -*- C++ -*-===//
//
// Post-scheduling bottleneck diagnosis for HIVM IR analysis.
// Consumes an HIVMAnalysisReport and HardwareConfig to classify per-op and
// global root causes and generate actionable suggestions.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_ANALYSIS_HIVMBOTTLENECKDIAGNOSIS_H
#define ASCENDMODEL_ANALYSIS_HIVMBOTTLENECKDIAGNOSIS_H

#include "AscendModel/Analysis/HIVMPipeEnum.h"
#include "AscendModel/Analysis/HardwareConfig.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

namespace mlir {
namespace ascend {

struct HIVMOp;
struct HIVMAnalysisReport;

enum class BottleneckType {
  BandwidthBound,
  ComputeBound,
  StartupOverhead,
  SyncOverhead,
  PipelineImbalance,
  LowParallelism
};

struct OpDiagnosis {
  size_t opId;
  int lineNumber = 0;
  std::string opName;
  HIVMPipe pipe = HIVMPipe::Unknown;
  BottleneckType rootCause;
  std::string evidence;
  std::string explanation;
  std::vector<std::string> suggestions;
  int64_t theoreticalMinCycles = 0;
  int64_t actualCycles = 0;
  double overheadRatio = 0.0;
};

struct PipelineDiagnosis {
  BottleneckType rootCause;
  HIVMPipe bottleneckPipe = HIVMPipe::Unknown;
  double imbalanceRatio = 0.0;
  std::string evidence;
  std::string explanation;
  std::vector<std::string> suggestions;
};

struct HIVMBottleneckReport {
  std::vector<OpDiagnosis> opDiagnoses;
  PipelineDiagnosis pipeDiagnosis;
  BottleneckType globalRootCause = BottleneckType::BandwidthBound;
  std::string globalEvidence;
  std::string globalExplanation;
  std::vector<std::string> globalSuggestions;
  double syncOverheadRatio = 0.0;
  double barrierOverheadRatio = 0.0;

  void print(llvm::raw_ostream &os, const HardwareConfig &config) const;
};

class HIVMBottleneckDiagnoser {
public:
  HIVMBottleneckDiagnoser(const HardwareConfig &config);

  bool diagnose(const HIVMAnalysisReport &report,
                HIVMBottleneckReport &out) const;

  static llvm::StringRef stringifyBottleneckType(BottleneckType type);

private:
  const HardwareConfig &config;

  OpDiagnosis diagnoseOp(const HIVMOp &op) const;
  PipelineDiagnosis diagnosePipeline(const HIVMAnalysisReport &report) const;
  BottleneckType diagnoseGlobal(const HIVMBottleneckReport &partial,
                                const HIVMAnalysisReport &report) const;

  int64_t computeTheoreticalMin(const HIVMOp &op) const;
};

} // namespace ascend
} // namespace mlir

#endif // ASCENDMODEL_ANALYSIS_HIVMBOTTLENECKDIAGNOSIS_H