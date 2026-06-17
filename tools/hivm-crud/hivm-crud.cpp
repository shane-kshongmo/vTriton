//===- hivm-crud.cpp - CLI wrapper for HivmOpsEditor / HivmCompiler -------===//
//
// Thin CLI wrapper around the HivmOpsEditor C++ API and HivmCompiler.
// Upper-level C++ code should call the APIs directly instead of this CLI tool.
//
// Usage:
//   hivm-crud --input kernel.hivm.mlir.bak --output kernel.hivm.mlir \
//             --mode <read|add|delete|modify|roundtrip>
//   hivm-crud --input kernel.hivm.mlir --output kernel.o --mode compile
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Transforms/HivmCompiler.h"
#include "AscendModel/Transforms/HivmOpsEditor.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"

#ifdef TRITONSIM_HAS_BISHENGIR_HIVM
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#endif

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
    llvm::cl::desc("Mode: read|add|delete|modify|roundtrip|compile"),
    llvm::cl::value_desc("mode"), llvm::cl::init("read"));

static llvm::cl::opt<unsigned> removeGMTrips(
    "remove-gm-trips",
    llvm::cl::desc("Remove N redundant GM round-trips (load+sync pairs)"),
    llvm::cl::value_desc("N"), llvm::cl::init(0));

// Compile-specific options
static llvm::cl::opt<bool> enableTritonKernel(
    "enable-triton-kernel",
    llvm::cl::desc("Enable Triton kernel compile (default: true in compile mode)"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> enableAutoMultiBuffer(
    "enable-auto-multi-buffer",
    llvm::cl::desc("Enable auto multi-buffer (default: true in compile mode)"),
    llvm::cl::init(true));

static llvm::cl::list<std::string> hivmcArgs(
    "hivmc-arg",
    llvm::cl::desc("Extra argument forwarded to hivmc (can be repeated)"),
    llvm::cl::ZeroOrMore, llvm::cl::value_desc("arg"));

// Helper function to collect ops by name (supports both regular and HIR ops)
void collectOpsByName(HivmOpsEditor &editor, const std::string &opName, 
                      std::vector<Operation *> &ops) {
  auto allOps = editor.listOps();
  for (auto &info : allOps) {
    if (info.op->getName().getStringRef().contains(opName)) {
      ops.push_back(info.op);
    }
  }
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
      "HIVM CRUD tool - apply optimisation suggestions to HIVM MLIR\n");

  DialectRegistry registry;
  registry.insert<HIVMDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
#ifdef TRITONSIM_HAS_BISHENGIR_HIVM
  registry.insert<mlir::annotation::AnnotationDialect>();
  registry.insert<mlir::hacc::HACCDialect>();
#endif

  if (inputFilename.empty()) {
    llvm::errs() << "Error: --input is required\n";
    return 1;
  }

  // compile mode: bypass MLIR parsing, delegate directly to bishengir-compile.
  if (crudMode == "compile") {
    if (outputFilename == "-" || outputFilename.empty()) {
      llvm::errs() << "Error: --output is required in compile mode\n";
      return 1;
    }

    HivmCompiler::Options opts;
    opts.enableTritonKernelCompile = enableTritonKernel;
    opts.enableAutoMultiBuffer = enableAutoMultiBuffer;
    for (const auto &arg : hivmcArgs)
      opts.extraArgs.push_back("--hivmc-args=" + arg);

    if (failed(HivmCompiler::compileFile(inputFilename, outputFilename, opts)))
      return 1;
    llvm::outs() << "Compiled " << inputFilename << " -> " << outputFilename << "\n";
    return 0;
  }

  // Other modes need MLIR parsing.
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  auto module = HivmOpsEditor::loadFromFile(ctx, inputFilename);
  if (!module) {
    llvm::errs() << "Error: failed to parse " << inputFilename << "\n";
    return 1;
  }

  HivmOpsEditor editor(*module);

  if (crudMode == "read") {
    editor.printSummary(llvm::outs());
  } else if (crudMode == "add") {
    // 收集所有类型的 vadd 操作（包括 hivm.hir.vadd）
    std::vector<Operation *> allVAddOps;
    collectOpsByName(editor, "vadd", allVAddOps);
    
    if (!allVAddOps.empty()) {
      MLIRContext *ctx = allVAddOps[0]->getContext();
      editor.addSetFlagWaitFlagBefore(
          allVAddOps[0],
          hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_V), 
          hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_MTE2), 
          hivm::EventAttr::get(ctx, hivm::EVENT::EVENT_ID2));
      llvm::outs() << "Added set_flag+wait_flag before first vadd\n";
    } else {
      llvm::errs() << "Error: No vadd op found to modify\n";
      return 1;
    }
  } else if (crudMode == "delete") {
    // 收集并删除所有 set_flag 和 wait_flag 操作
    std::vector<Operation *> setFlagOps, waitFlagOps;
    collectOpsByName(editor, "set_flag", setFlagOps);
    collectOpsByName(editor, "wait_flag", waitFlagOps);
    
    for (auto op : setFlagOps) editor.deleteOp(op);
    for (auto op : waitFlagOps) editor.deleteOp(op);
    llvm::outs() << "Deleted " << setFlagOps.size() << " set_flag and " 
                 << waitFlagOps.size() << " wait_flag ops\n";
  } else if (crudMode == "modify") {
    std::vector<Operation *> allVAddOps;
    collectOpsByName(editor, "vadd", allVAddOps);
    
    llvm::outs() << "Found " << allVAddOps.size() << " vadd ops to replace\n";
    for (auto op : allVAddOps) {
      auto vadd = dyn_cast<VAddOp>(op);
      if (vadd) {
        editor.replaceOpWith<VAddOp, VSubOp>(vadd, vadd.getSrc(), vadd.getDst());
      }
    }
    llvm::outs() << "Replaced all vadd with vsub\n";
  } else if (crudMode == "roundtrip") {
    llvm::outs() << "=== Before ===\n";
    editor.printSummary(llvm::outs());

    if (removeGMTrips > 0) {
      editor.removeRedundantLoadStorePair(removeGMTrips);
      llvm::outs() << "Removed " << removeGMTrips << " redundant GM trips\n";
    }

    std::vector<Operation *> allVAddOps;
    collectOpsByName(editor, "vadd", allVAddOps);
    
    llvm::outs() << "Found " << allVAddOps.size() << " vadd ops to replace\n";
    for (auto op : allVAddOps) {
      auto vadd = dyn_cast<VAddOp>(op);
      if (vadd) {
        editor.replaceOpWith<VAddOp, VSubOp>(vadd, vadd.getSrc(), vadd.getDst());
      }
    }

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
