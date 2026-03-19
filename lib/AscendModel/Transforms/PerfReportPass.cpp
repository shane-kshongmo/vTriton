//===- PerfReportPass.cpp - Generate performance report ------------------===//
//
// This file generates a comprehensive performance report.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/IR/AscendModelDialect.h"
#include "AscendModel/Transforms/Passes.h"
#include "AscendModel/Analysis/PipelineAnalysis.h"
#include "AscendModel/HardwareParams.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Format.h"

namespace mlir {
namespace ascend {

#define GEN_PASS_DEF_PERFREPORTPASS
#include "AscendModel/Transforms/Passes.h.inc"

namespace {

/// Get the HWUnit for an operation
static HWUnit getOpHWUnit(Operation *op) {
  if (isa<MatmulOp>(op))
    return HWUnit::Cube;
  if (isa<CubeLoadOp>(op))
    return HWUnit::CubeMTE2;
  if (isa<CubeStoreOp>(op))
    return HWUnit::FixPipe;
  if (isa<VectorLoadOp>(op))
    return HWUnit::VecMTE2;
  if (isa<VectorStoreOp>(op))
    return HWUnit::MTE3;
  if (isa<AddOp, SubOp, MulOp, DivOp, MaxOp, MinOp,
          ExpOp, LogOp, SqrtOp, RsqrtOp, TanhOp, SigmoidOp,
          NegOp, AbsOp, ReluOp, CastOp,
          ReduceSumOp, ReduceMaxOp, ReduceMinOp, ReduceProdOp,
          BroadcastOp, SelectOp>(op))
    return HWUnit::Vector;
  return HWUnit::Scalar;
}

static int64_t getNumElementsFromType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    int64_t count = 1;
    for (int64_t dim : tensorType.getShape()) {
      if (dim == ShapedType::kDynamic)
        return 1024;
      count *= dim;
    }
    return count;
  }
  return 1;
}

struct PerfReportPass
    : public impl::PerfReportPassBase<PerfReportPass> {
  using PerfReportPassBase::PerfReportPassBase;
  
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    int64_t totalCycles = 0;
    int64_t totalFLOPs = 0;
    int64_t totalBytes = 0;
    
    std::map<HWUnit, int64_t> unitCycles;
    std::map<std::string, int64_t> opCounts;
    std::map<std::string, int64_t> opCycles;
    
    for (int i = 0; i <= static_cast<int>(HWUnit::Scalar); ++i) {
      unitCycles[static_cast<HWUnit>(i)] = 0;
    }
    
    module.walk([&](Operation *op) {
      auto cyclesAttr = op->getAttrOfType<IntegerAttr>("estimated_cycles");
      if (!cyclesAttr)
        return;
      
      int64_t cycles = cyclesAttr.getInt();
      std::string opName = op->getName().getStringRef().str();
      
      opCounts[opName]++;
      opCycles[opName] += cycles;
      
      HWUnit hwUnit = getOpHWUnit(op);
      
      // Count FLOPs
      if (auto matmulOp = dyn_cast<MatmulOp>(op)) {
        int64_t m = matmulOp.getM();
        int64_t n = matmulOp.getN();
        int64_t k = matmulOp.getK();
        totalFLOPs += 2 * m * n * k;
      } else if (isa<AddOp, SubOp, MulOp, DivOp, MaxOp, MinOp,
                     NegOp, AbsOp, ReluOp>(op)) {
        if (op->getNumResults() > 0) {
          totalFLOPs += getNumElementsFromType(op->getResult(0).getType());
        }
      } else if (isa<ExpOp, LogOp, SqrtOp, RsqrtOp, TanhOp, SigmoidOp>(op)) {
        if (op->getNumResults() > 0) {
          totalFLOPs += getNumElementsFromType(op->getResult(0).getType()) * 10;
        }
      } else if (isa<ReduceSumOp, ReduceMaxOp, ReduceMinOp, ReduceProdOp>(op)) {
        if (op->getNumOperands() > 0) {
          totalFLOPs += getNumElementsFromType(op->getOperand(0).getType());
        }
      }
      
      // Count bytes
      if (isa<CubeLoadOp>(op)) {
        if (auto bytesAttr = op->getAttrOfType<IntegerAttr>("bytes"))
          totalBytes += bytesAttr.getInt();
      } else if (isa<CubeStoreOp>(op)) {
        if (auto bytesAttr = op->getAttrOfType<IntegerAttr>("bytes"))
          totalBytes += bytesAttr.getInt();
      } else if (isa<VectorLoadOp>(op)) {
        if (auto bytesAttr = op->getAttrOfType<IntegerAttr>("bytes"))
          totalBytes += bytesAttr.getInt();
      } else if (isa<VectorStoreOp>(op)) {
        if (auto bytesAttr = op->getAttrOfType<IntegerAttr>("bytes"))
          totalBytes += bytesAttr.getInt();
      }
      
      unitCycles[hwUnit] += cycles;
      totalCycles += cycles;
    });
    
    auto scheduledCyclesAttr = module->getAttrOfType<IntegerAttr>("ascend.scheduled_cycles");
    int64_t scheduledCycles = scheduledCyclesAttr ? scheduledCyclesAttr.getInt() : totalCycles;
    
    double timeUs = scheduledCycles / ::ascend::hw::CYCLES_PER_US;
    double achievedTFLOPS = (totalFLOPs / 1e12) / (timeUs / 1e6);
    double achievedBandwidth = (totalBytes / 1e9) / (timeUs / 1e6);
    double arithmeticIntensity = totalBytes > 0 ? static_cast<double>(totalFLOPs) / totalBytes : 0;
    
    double ridgePoint = ::ascend::hw::CUBE_TFLOPS / ::ascend::hw::HBM_BANDWIDTH_GBS;
    bool isComputeBound = arithmeticIntensity > ridgePoint;
    
    HWUnit bottleneck = HWUnit::Scalar;
    int64_t maxCycles = 0;
    for (const auto &[unit, cycles] : unitCycles) {
      if (cycles > maxCycles) {
        maxCycles = cycles;
        bottleneck = unit;
      }
    }
    
    llvm::raw_ostream &os = llvm::outs();
    
    os << "\n";
    os << "+==============================================================+\n";
    os << "|          Ascend 910B Performance Analysis Report             |\n";
    os << "+==============================================================+\n";
    os << "|                                                              |\n";
    os << "|  Timing Summary                                              |\n";
    os << "|  --------------                                              |\n";
    os << llvm::format("|  Total Cycles:        %12ld                        |\n", scheduledCycles);
    os << llvm::format("|  Estimated Time:      %12.3f us                     |\n", timeUs);
    os << "|                                                              |\n";
    os << "|  Compute Summary                                             |\n";
    os << "|  ---------------                                             |\n";
    os << llvm::format("|  Total FLOPs:         %12ld                        |\n", totalFLOPs);
    os << llvm::format("|  Achieved TFLOPS:     %12.3f                        |\n", achievedTFLOPS);
    os << llvm::format("|  Peak TFLOPS (Cube):  %12.1f                        |\n", ::ascend::hw::CUBE_TFLOPS);
    os << "|                                                              |\n";
    os << "|  Memory Summary                                              |\n";
    os << "|  --------------                                              |\n";
    os << llvm::format("|  Total Bytes:         %12ld                        |\n", totalBytes);
    os << llvm::format("|  Achieved BW (GB/s):  %12.3f                        |\n", achievedBandwidth);
    os << llvm::format("|  Peak BW (GB/s):      %12.1f                        |\n", ::ascend::hw::HBM_BANDWIDTH_GBS);
    os << "|                                                              |\n";
    os << "|  Roofline Analysis                                           |\n";
    os << "|  -----------------                                           |\n";
    os << llvm::format("|  Arithmetic Intensity:%12.3f FLOP/Byte             |\n", arithmeticIntensity);
    os << llvm::format("|  Ridge Point:         %12.3f FLOP/Byte             |\n", ridgePoint);
    os << "|  Bound:               ";
    os << (isComputeBound ? "Compute-bound" : " Memory-bound");
    os << "                          |\n";
    os << "|                                                              |\n";
    os << "|  Bottleneck Unit:     ";
    os << llvm::format("%-12s", stringifyHWUnit(bottleneck).str().c_str());
    os << "                        |\n";
    os << "|                                                              |\n";
    os << "+==============================================================+\n";
    os << "\n";
  }
};

} // namespace
} // namespace ascend
} // namespace mlir
