//===- ascend-perf-model-opt.cpp - Ascend Performance Model Tool ---------===//
//
// This file implements the entry point for the Ascend 910B performance
// modeling tool. It extends mlir-opt with AscendModel dialect support,
// and optionally registers Triton dialect for parsing .ttir files.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/IR/AscendModelDialect.h"
#include "AscendModel/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

// Triton dialect support (optional — enabled when built with Triton)
#ifdef TRITONSIM_HAS_TRITON
#include "triton/Dialect/Triton/IR/Dialect.h"
#endif

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;

  // Core MLIR dialects
  registry.insert<func::FuncDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<math::MathDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<tensor::TensorDialect>();

  // AscendModel dialect
  registry.insert<ascend::AscendModelDialect>();

  // Triton dialect — enables parsing .ttir files directly
#ifdef TRITONSIM_HAS_TRITON
  registry.insert<mlir::triton::TritonDialect>();
#endif

  // Register all AscendModel passes and pipelines
  ascend::registerAllAscendModelPasses();

  return asMainReturnCode(
      MlirOptMain(argc, argv, "Ascend 910B Performance Model Optimizer\n", registry));
}
