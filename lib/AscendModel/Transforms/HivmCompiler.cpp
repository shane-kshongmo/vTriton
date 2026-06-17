//===- HivmCompiler.cpp - HIVM MLIR → ELF compilation wrapper ------------===//
//
// Implements HivmCompiler by delegating to the external `bishengir-compile`
// binary with the standard NPU backend pipeline flags.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Transforms/HivmCompiler.h"

#ifdef TRITONSIM_HAS_BISHENGIR_HIVM

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

using namespace mlir;
using namespace mlir::ascend;

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

std::vector<std::string> HivmCompilerOptions::toArgs() const {
  std::vector<std::string> args;
  auto add = [&](const char *flag, bool value) {
    args.push_back((llvm::Twine("--") + flag + "=" + (value ? "true" : "false")).str());
  };
  add("enable-hivm-compile", enableHIVMCompile);
  add("enable-triton-kernel-compile", enableTritonKernelCompile);
  add("enable-auto-multi-buffer", enableAutoMultiBuffer);
  args.insert(args.end(), extraArgs.begin(), extraArgs.end());
  return args;
}

// ---------------------------------------------------------------------------
// Binary discovery
// ---------------------------------------------------------------------------

static std::string findBishengirCompile() {
  // 1. Search BISHENG_INSTALL_PATH first.
  if (const char *installPath = std::getenv("BISHENG_INSTALL_PATH")) {
    if (*installPath) {
      auto result = llvm::sys::findProgramByName("bishengir-compile",
                                                  {llvm::StringRef(installPath)});
      if (result)
        return *result;
    }
  }
  // 2. Fall back to PATH.
  auto result = llvm::sys::findProgramByName("bishengir-compile");
  if (result)
    return *result;
  return {};
}

// ---------------------------------------------------------------------------
// compileFile
// ---------------------------------------------------------------------------

LogicalResult HivmCompiler::compileFile(llvm::StringRef inputPath,
                                        llvm::StringRef outputPath,
                                        const HivmCompilerOptions &opts) {
  std::string binary = findBishengirCompile();
  if (binary.empty()) {
    llvm::errs() << "[HivmCompiler] Cannot find bishengir-compile in "
                    "BISHENG_INSTALL_PATH or $PATH\n";
    return failure();
  }

  std::vector<std::string> flagArgs = opts.toArgs();
  llvm::SmallVector<llvm::StringRef, 16> args;
  args.reserve(4 + flagArgs.size());
  args.push_back(binary);
  for (const auto &a : flagArgs)
    args.push_back(a);
  args.push_back(inputPath);
  args.push_back("-o");
  args.push_back(outputPath);

  std::string errMsg;
  int rc = llvm::sys::ExecuteAndWait(binary, args, /*Env=*/std::nullopt,
                                     /*Redirects=*/{}, /*SecondsToWait=*/0,
                                     /*MemoryLimit=*/0, &errMsg);
  if (rc != 0) {
    llvm::errs() << "[HivmCompiler] bishengir-compile failed (exit " << rc
                 << "): " << errMsg << "\n";
    return failure();
  }
  return success();
}

// ---------------------------------------------------------------------------
// compileModule
// ---------------------------------------------------------------------------

LogicalResult HivmCompiler::compileModule(ModuleOp module,
                                          llvm::StringRef outputPath,
                                          const HivmCompilerOptions &opts) {
  // Create a temporary file for the MLIR module.
  int fd;
  llvm::SmallString<128> tempPath;
  if (std::error_code ec = llvm::sys::fs::createTemporaryFile(
          "hivm-module", "mlir", fd, tempPath)) {
    llvm::errs() << "[HivmCompiler] Failed to create temp file: "
                 << ec.message() << "\n";
    return failure();
  }

  llvm::raw_fd_ostream os(fd, /*shouldClose=*/true);
  module.print(os);
  os.close();

  LogicalResult result = compileFile(tempPath, outputPath, opts);

  // Clean up the temporary file.
  llvm::sys::fs::remove(tempPath);

  return result;
}

#endif // TRITONSIM_HAS_BISHENGIR_HIVM
