//===- hivm-optimize.cpp - Apply optimisation to HIVM MLIR -------------===//
//
// Uses HivmOpsEditor C++ API to apply bottleneck-driven optimisations
// to a kernel .npuir.mlir file.  Reads the file, applies a sequence of
// transformations, and writes the result.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Transforms/HivmOpsEditor.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::ascend;
using namespace mlir::hivm;

static llvm::cl::opt<std::string> inputFilename(
    "input", llvm::cl::desc("Input MLIR file"),
    llvm::cl::value_desc("filename"), llvm::cl::Required);

static llvm::cl::opt<std::string> outputFilename(
    "output", llvm::cl::desc("Output MLIR file"),
    llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<unsigned> removePipeVBarriers(
    "remove-pipe-v-barriers",
    llvm::cl::desc("Remove pipe_barrier[PIPE_V] ops from Vector inner loops"),
    llvm::cl::value_desc("0|1"), llvm::cl::init(1));

static llvm::cl::opt<unsigned> replaceVDivWithVMul(
    "replace-vdiv-with-vmul",
    llvm::cl::desc("Replace vdiv ops with vmul (uses same src/dst as placeholder)"),
    llvm::cl::value_desc("0|1"), llvm::cl::init(1));

static llvm::cl::opt<unsigned> removeGMTrips(
    "remove-gm-trips",
    llvm::cl::desc("Remove N redundant GM load+sync round-trips"),
    llvm::cl::value_desc("N"), llvm::cl::init(0));

static llvm::cl::opt<unsigned> fuseConsecutiveCompute(
    "fuse-consecutive-compute",
    llvm::cl::desc("Fuse consecutive vadd ops that form dependency chains"),
    llvm::cl::value_desc("0|1"), llvm::cl::init(0));

static llvm::cl::opt<unsigned> verbose(
    "verbose",
    llvm::cl::desc("Print detailed op counts before and after"),
    llvm::cl::value_desc("0|1"), llvm::cl::init(1));

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
      "HIVM optimiser - apply bottleneck-driven transformations\n");

  DialectRegistry registry;
  registry.insert<HIVMDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::annotation::AnnotationDialect>();
  registry.insert<mlir::hacc::HACCDialect>();

  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  auto module = HivmOpsEditor::loadFromFile(ctx, inputFilename);
  if (!module) {
    llvm::errs() << "Error: failed to parse " << inputFilename << "\n";
    return 1;
  }

  HivmOpsEditor editor(*module);

  if (verbose) {
    llvm::outs() << "=== Before optimisation ===\n";
    editor.printSummary(llvm::outs());
  }

  // --- 1. Remove pipe_barrier[<PIPE_V>] from Vector inner loops ---
  if (removePipeVBarriers) {
    SmallVector<Operation *> toErase;
    module->walk([&](PipeBarrierOp barrier) {
      auto pipeAttr = barrier.getPipeAttr();
      if (!pipeAttr)
        return WalkResult::advance();
      if (pipeAttr.getPipe() == PIPE::PIPE_V) {
        // Only remove barriers that are inside scf.for loops
        // (i.e., not top-level function barriers)
        auto *parentOp = barrier->getParentOp();
        while (parentOp) {
          if (isa<scf::ForOp>(parentOp)) {
            toErase.push_back(barrier);
            break;
          }
          parentOp = parentOp->getParentOp();
        }
      }
      return WalkResult::advance();
    });
    for (auto *op : toErase)
      op->erase();
    llvm::outs() << "Removed " << toErase.size()
                 << " pipe_barrier[<PIPE_V>] from Vector inner loops\n";
  }

  // --- 2. Replace vdiv with vmul ---
  if (replaceVDivWithVMul) {
    SmallVector<VDivOp> vdivOps;
    module->walk([&](VDivOp op) { vdivOps.push_back(op); });
    unsigned replaced = 0;
    for (auto vdiv : vdivOps) {
      editor.replaceVDivWithVMul(vdiv);
      ++replaced;
    }
    llvm::outs() << "Replaced " << replaced << " vdiv with vmul\n";
  }

  // --- 3. Remove redundant GM round-trips ---
  if (removeGMTrips > 0) {
    editor.removeRedundantLoadStorePair(removeGMTrips);
    llvm::outs() << "Removed " << removeGMTrips
                 << " redundant GM round-trips\n";
  }

  // --- 4. Fuse consecutive compute ops ---
  if (fuseConsecutiveCompute) {
    editor.fuseConsecutiveComputeOps();
    llvm::outs() << "Fused consecutive vadd chains\n";
  }

  if (verbose) {
    llvm::outs() << "\n=== After optimisation ===\n";
    editor.printSummary(llvm::outs());
  }

  if (outputFilename == "-" || outputFilename.empty()) {
    llvm::outs() << "\n=== Output MLIR ===\n";
    llvm::outs() << editor.exportToString() << "\n";
  } else {
    if (failed(editor.exportToFile(outputFilename))) {
      llvm::errs() << "Error writing output to " << outputFilename << "\n";
      return 1;
    }
    llvm::outs() << "Output written to " << outputFilename << "\n";
  }

  return 0;
}
