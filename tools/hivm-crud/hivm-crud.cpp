//===- hivm-crud.cpp - CLI wrapper for HivmOpsEditor ---------------------===//
//
// Thin CLI wrapper around the HivmOpsEditor C++ API.
// Upper-level C++ code should call HivmOpsEditor directly instead of this
// CLI tool.
//
// Usage:
//   hivm-crud --input kernel.hivm.mlir.bak --output kernel.hivm.mlir \
//             --mode <read|add|delete|modify|roundtrip>
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Transforms/HivmOpsEditor.h"

#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::ascend;
using namespace mlir::hivm;

static llvm::cl::opt<std::string> inputFilename(
    "input", llvm::cl::desc("Input MLIR file"),
    llvm::cl::value_desc("filename"), llvm::cl::init(""));

static llvm::cl::opt<std::string> outputFilename(
    "output", llvm::cl::desc("Output MLIR file"),
    llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<std::string> crudMode(
    "mode",
    llvm::cl::desc("CRUD mode: read|add|delete|modify|roundtrip"),
    llvm::cl::value_desc("mode"), llvm::cl::init("read"));

static llvm::cl::opt<unsigned> removeGMTrips(
    "remove-gm-trips",
    llvm::cl::desc("Remove N redundant GM round-trips (load+sync pairs)"),
    llvm::cl::value_desc("N"), llvm::cl::init(0));

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
      "HIVM CRUD tool - apply optimisation suggestions to HIVM MLIR\n");

  DialectRegistry registry;
  registry.insert<HIVMDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<memref::MemRefDialect>();

  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  if (inputFilename.empty()) {
    llvm::errs() << "Error: --input is required\n";
    return 1;
  }

  auto module = HivmOpsEditor::loadFromFile(ctx, inputFilename);
  if (!module) {
    llvm::errs() << "Error: failed to parse " << inputFilename << "\n";
    return 1;
  }

  HivmOpsEditor editor(*module);

  if (crudMode == "read") {
    editor.printSummary(llvm::outs());
  } else if (crudMode == "add") {
    editor.addSetFlagWaitFlagBefore(
        editor.collectOps<VAddOp>()[0],
        PIPE::PIPE_V, PIPE::PIPE_MTE2, EVENT_ID::EVENT_ID2);
    llvm::outs() << "Added set_flag+wait_flag before first vadd\n";
  } else if (crudMode == "delete") {
    editor.deleteAllOpsOfKind<SetFlagOp>();
    editor.deleteAllOpsOfKind<WaitFlagOp>();
    llvm::outs() << "Deleted all set_flag and wait_flag ops\n";
  } else if (crudMode == "modify") {
    auto vadds = editor.collectOps<VAddOp>();
    for (auto vadd : vadds)
      editor.replaceOpWith<VAddOp, VSubOp>(vadd, vadd.getSrc(), vadd.getDst());
    llvm::outs() << "Replaced all vadd with vsub\n";
  } else if (crudMode == "roundtrip") {
    llvm::outs() << "=== Before ===\n";
    editor.printSummary(llvm::outs());

    if (removeGMTrips > 0) {
      editor.removeRedundantLoadStorePair(removeGMTrips);
      llvm::outs() << "Removed " << removeGMTrips << " redundant GM trips\n";
    }

    auto vadds = editor.collectOps<VAddOp>();
    for (auto vadd : vadds)
      editor.replaceOpWith<VAddOp, VSubOp>(vadd, vadd.getSrc(), vadd.getDst());

    llvm::outs() << "\n=== After ===\n";
    editor.printSummary(llvm::outs());
  }

  if (outputFilename == "-" || outputFilename.empty()) {
    llvm::outs() << "\n=== Output MLIR ===\n";
    llvm::outs() << editor.exportToString() << "\n";
  } else {
    if (failed(editor.exportToFile(outputFilename))) {
      llvm::errs() << "Error writing output\n";
      return 1;
    }
    llvm::outs() << "Output written to " << outputFilename << "\n";
  }

  return 0;
}
