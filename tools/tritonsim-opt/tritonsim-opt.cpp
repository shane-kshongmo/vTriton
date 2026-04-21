//===- tritonsim-opt.cpp - Ascend performance modeling opt tool ----------===//
//
// mlir-opt-style tool for the TTIR → AscendModel modelling pipeline.
// Reads generic MLIR (piped from triton-opt), converts tt.* ops through the
// ascend-perf-model pipeline, and emits cycle estimates.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/IR/AscendModelDialect.h"
#include "AscendModel/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#ifdef TRITONSIM_HAS_TRITON
#include "triton/Dialect/Triton/IR/Dialect.h"
#endif

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;

  // Core MLIR dialects
  registry.insert<affine::AffineDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<math::MathDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<tensor::TensorDialect>();

  // AscendModel dialect
  registry.insert<ascend::AscendModelDialect>();

#ifdef TRITONSIM_HAS_TRITON
  registry.insert<mlir::triton::TritonDialect>();
#endif

  // Register all AscendModel passes and the ascend-perf-model pipeline
  ascend::registerAllAscendModelPasses();

  return asMainReturnCode(
      MlirOptMain(argc, argv, "Ascend performance modeling tool\n", registry));
}
