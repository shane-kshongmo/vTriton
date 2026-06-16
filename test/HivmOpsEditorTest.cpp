//===- HivmOpsEditorTest.cpp - HIVM Ops Editor Test Case ------------------===//
//
// Simple test case for HivmOpsEditor basic operations.
// Tests: loadFromFile and exportToFile.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Transforms/HivmOpsEditor.h"

#ifdef TRITONSIM_HAS_BISHENGIR_HIVM

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace mlir;
using namespace mlir::ascend;

int main(int argc, char** argv) {
  if (argc < 2) {
    llvm::errs() << "Usage: " << argv[0] << " <kernel.hivm.milr>\n";
    return 1;
  }
  
  std::string kernelPath = argv[1];
  
  if (!llvm::sys::fs::exists(kernelPath)) {
    llvm::errs() << "Error: Kernel file not found: " << kernelPath << "\n";
    return 1;
  }
  
  llvm::outs() << "=== HIVM Ops Editor Test ===\n";
  llvm::outs() << "Input: " << kernelPath << "\n";
  
  MLIRContext ctx;
  
  llvm::outs() << "Loading module...\n";
  auto moduleRef = HivmOpsEditor::loadFromFile(ctx, kernelPath);
  if (!moduleRef) {
    llvm::errs() << "Error: Failed to load module\n";
    return 1;
  }
  
  HivmOpsEditor editor(*moduleRef);
  
  auto ops = editor.listOps();
  llvm::outs() << "Loaded " << ops.size() << " operations\n";
  
  std::string outputPath = kernelPath + ".out.mlir";
  llvm::outs() << "Exporting to: " << outputPath << "\n";
  
  if (failed(editor.exportToFile(outputPath))) {
    llvm::errs() << "Error: Failed to export module\n";
    return 1;
  }
  
  llvm::outs() << "=== Test completed successfully ===\n";
  return 0;
}

#else

#include <iostream>

int main() {
  std::cerr << "Error: TRITONSIM_HAS_BISHENGIR_HIVM not defined.\n";
  return 1;
}

#endif // TRITONSIM_HAS_BISHENGIR_HIVM
