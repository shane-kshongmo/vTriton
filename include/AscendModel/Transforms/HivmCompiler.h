//===- HivmCompiler.h - HIVM MLIR → ELF compilation wrapper -----*- C++ -*-===//
//
// Provides a C++ API for compiling an HIVM MLIR module into a kernel object
// file (ELF) by delegating to the external `bishengir-compile` binary.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_TRANSFORMS_HIVMCOMPILER_H
#define ASCENDMODEL_TRANSFORMS_HIVMCOMPILER_H

#ifdef TRITONSIM_HAS_BISHENGIR_HIVM

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

#include <string>
#include <vector>

namespace mlir {
namespace ascend {

class HivmCompiler {
public:
  struct Options {
    bool enableHIVMCompile = true;
    bool enableTritonKernelCompile = true;
    bool enableAutoMultiBuffer = true;
    std::vector<std::string> extraArgs;

    std::vector<std::string> toArgs() const;
  };

  /// Compile an HIVM MLIR file to a kernel object file.
  static LogicalResult compileFile(llvm::StringRef inputPath,
                                   llvm::StringRef outputPath,
                                   const Options &opts = Options());

  /// Compile an HIVM MLIR module to a kernel object file.
  /// The module is first serialised to a temporary file.
  static LogicalResult compileModule(ModuleOp module,
                                     llvm::StringRef outputPath,
                                     const Options &opts = Options());
};

} // namespace ascend
} // namespace mlir

#endif // TRITONSIM_HAS_BISHENGIR_HIVM
#endif // ASCENDMODEL_TRANSFORMS_HIVMCOMPILER_H
