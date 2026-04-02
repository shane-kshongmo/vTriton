//===- tritonsim-hivm.cpp - HIVM-based TritonSim entrypoint --------------===//
//
// Direct `.npuir.mlir` analysis plus optional Triton-DSL-to-HIVM workflow
// through triton-ascend's compile-only dump path.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Analysis/HIVMAnalysis.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

using namespace mlir::ascend;

namespace {

llvm::cl::opt<std::string> npuirFile(
    "npuir-file", llvm::cl::desc("Path to a dumped kernel.npuir.mlir file"),
    llvm::cl::value_desc("path"));

llvm::cl::opt<std::string> tritonScript(
    "triton-script",
    llvm::cl::desc("Path to a Triton Python script to compile in dump mode"),
    llvm::cl::value_desc("path"));

llvm::cl::opt<std::string> pythonExe(
    "python", llvm::cl::desc("Python executable for Triton-DSL mode"),
    llvm::cl::init("python3"), llvm::cl::value_desc("path"));

llvm::cl::opt<std::string> tritonAscendRoot(
    "triton-ascend-root",
    llvm::cl::desc("Optional triton-ascend source root kept for compatibility; the launcher uses the installed wheel"),
    llvm::cl::value_desc("path"));

llvm::cl::opt<std::string> tritonAscendArch(
    "triton-ascend-arch",
    llvm::cl::desc("Ascend target architecture for compile-only lowering"),
    llvm::cl::init("Ascend910_9599"), llvm::cl::value_desc("arch"));

llvm::cl::list<std::string> scriptArgs(
    "script-arg", llvm::cl::desc("Extra argument passed through to --triton-script"),
    llvm::cl::ZeroOrMore, llvm::cl::value_desc("arg"));

llvm::cl::opt<std::string> hardwareConfig(
    "hardware-config",
    llvm::cl::desc("Path to hardware configuration JSON file"),
    llvm::cl::init(""));

llvm::cl::opt<std::string> argBindings(
    "arg-bindings",
    llvm::cl::desc("Bindings for dynamic values used by HIVM loop bounds, e.g. arg10=128,arg9=256"),
    llvm::cl::init(""));

llvm::cl::opt<std::string> schedulerMode(
    "scheduler",
    llvm::cl::desc("HIVM scheduler backend: static or des"),
    llvm::cl::init("static"));

llvm::cl::opt<bool> keepDumpDir(
    "keep-dump-dir",
    llvm::cl::desc("Keep the temporary Triton dump directory after analysis"),
    llvm::cl::init(false));

llvm::cl::opt<std::string> perfettoTraceFile(
    "perfetto-trace-file",
    llvm::cl::desc("Write a Perfetto-compatible JSON trace for the scheduled HIVM ops"),
    llvm::cl::init(""), llvm::cl::value_desc("path"));

llvm::cl::opt<std::string> desGraphFile(
    "des-graph-file",
    llvm::cl::desc("Export the parsed operation graph as JSON for external DES simulation"),
    llvm::cl::init(""), llvm::cl::value_desc("path"));

std::optional<std::string> findLatestNpuirUnder(llvm::StringRef root) {
  std::error_code ec;
  llvm::sys::fs::recursive_directory_iterator it(root, ec), end;
  if (ec)
    return std::nullopt;

  std::string bestPath;
  llvm::sys::TimePoint<> bestTime;
  bool found = false;
  for (; it != end && !ec; it.increment(ec)) {
    if (!llvm::sys::fs::is_regular_file(it->path()))
      continue;
    if (!llvm::StringRef(it->path()).ends_with(".npuir.mlir"))
      continue;

    llvm::sys::fs::file_status status;
    if (llvm::sys::fs::status(it->path(), status))
      continue;

    if (!found || status.getLastModificationTime() > bestTime) {
      bestPath = it->path();
      bestTime = status.getLastModificationTime();
      found = true;
    }
  }
  if (!found)
    return std::nullopt;
  return bestPath;
}

std::optional<HIVMSchedulerMode> parseSchedulerMode(llvm::StringRef mode) {
  if (mode.empty() || mode == "static")
    return HIVMSchedulerMode::Static;
  if (mode == "des")
    return HIVMSchedulerMode::DES;
  return std::nullopt;
}

std::optional<std::string> detectAscendHome() {
  if (const char *envPath = std::getenv("ASCEND_HOME_PATH")) {
    if (*envPath && llvm::sys::fs::exists(envPath))
      return std::string(envPath);
  }

  constexpr llvm::StringLiteral candidates[] = {
      "/home/shane/Ascend/cann",
      "/home/shane/Ascend/cann-8.5.0",
      "/home/shane/Ascend/cann-9.0.0-beta.1",
      "/usr/local/Ascend/ascend-toolkit/latest",
  };
  for (llvm::StringLiteral candidate : candidates) {
    if (llvm::sys::fs::exists(candidate))
      return candidate.str();
  }
  return std::nullopt;
}

std::string prependEnvValue(llvm::StringRef prefix, const char *existing) {
  if (!existing || !*existing)
    return prefix.str();
  return (prefix + ":" + existing).str();
}

bool runPythonPreflight(const llvm::SmallVectorImpl<llvm::StringRef> &envRefs,
                        llvm::StringRef resolvedPython,
                        std::string &error) {
  llvm::SmallVector<llvm::StringRef, 8> args;
  args.push_back(resolvedPython);
  args.push_back("-c");
  args.push_back(
      "import importlib.util;"
      "mods=['triton','torch','torch_npu'];"
      "missing=[m for m in mods if importlib.util.find_spec(m) is None];"
      "raise SystemExit(0 if not missing else 1)");

  std::string errMsg;
  int rc = llvm::sys::ExecuteAndWait(resolvedPython, args, envRefs, {}, 0, 0, &errMsg);
  if (rc != 0) {
    error = "Triton-DSL mode requires a working python environment with modules: "
            "triton, torch, torch_npu";
    if (!errMsg.empty())
      error += ". Preflight error: " + errMsg;
    if (!tritonAscendRoot.empty())
      error += ". PYTHONPATH was prepared from --triton-ascend-root, but the runtime "
               "dependencies are still unavailable.";
    return false;
  }

  return true;
}

bool runTritonScriptToDump(std::string &outNpuirPath, std::string &tempDir,
                           std::string &error) {
  if (tritonScript.empty()) {
    error = "missing --triton-script";
    return false;
  }
  if (!llvm::sys::fs::exists(tritonScript)) {
    error = "triton script does not exist: " + tritonScript;
    return false;
  }
  auto resolvedProgram = llvm::sys::findProgramByName(pythonExe);
  if (!resolvedProgram) {
      error = "python executable not found: " + pythonExe;
      return false;
  }
  llvm::StringRef resolvedPython = *resolvedProgram;

  llvm::SmallString<128> uniqueDir;
  if (std::error_code ec =
          llvm::sys::fs::createUniqueDirectory("tritonsim-hivm", uniqueDir)) {
    error = "failed to create temporary dump directory: " + ec.message();
    return false;
  }
  tempDir = uniqueDir.str().str();

  llvm::SmallVector<llvm::StringRef, 16> args;
  args.push_back(resolvedPython);
  args.push_back(tritonScript);
  for (const std::string &arg : scriptArgs)
    args.push_back(arg);

  std::vector<std::string> envStorage;
  auto pushEnv = [&](llvm::StringRef key, llvm::StringRef value) {
    envStorage.push_back((key + "=" + value).str());
  };

  pushEnv("TRITON_ALWAYS_COMPILE", "1");
  pushEnv("TRITON_COMPILE_ONLY", "1");
  pushEnv("TRITON_KERNEL_DUMP", "1");
  pushEnv("TRITON_DEBUG", "1");
  pushEnv("TRITON_DUMP_DIR", tempDir);
  pushEnv("TRITON_CACHE_DIR", tempDir + "/cache");
  pushEnv("TRITON_ASCEND_ARCH", tritonAscendArch);
  pushEnv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0");

  if (llvm::sys::fs::create_directories(tempDir + "/cache")) {
    error = "failed to create Triton cache directory under " + tempDir;
    return false;
  }

  if (std::optional<std::string> ascendHome = detectAscendHome()) {
    pushEnv("ASCEND_HOME_PATH", *ascendHome);
    pushEnv("ASCEND_TOOLKIT_HOME", *ascendHome);
    pushEnv("ASCEND_AICPU_PATH", *ascendHome);
    pushEnv("ASCEND_OPP_PATH", *ascendHome + "/opp");
    pushEnv("TOOLCHAIN_HOME", *ascendHome + "/toolkit");
    pushEnv("PATH",
            prependEnvValue(*ascendHome + "/bin", std::getenv("PATH")));
    pushEnv("LD_LIBRARY_PATH",
            prependEnvValue(*ascendHome + "/lib64:" + *ascendHome + "/devlib",
                            std::getenv("LD_LIBRARY_PATH")));
  }

  const char *existingPythonPath = std::getenv("PYTHONPATH");
  if (existingPythonPath && *existingPythonPath) {
    pushEnv("PYTHONPATH", existingPythonPath);
  }

  llvm::SmallVector<llvm::StringRef, 16> envRefs;
  for (const std::string &entry : envStorage)
    envRefs.push_back(entry);

  if (!runPythonPreflight(envRefs, resolvedPython, error))
    return false;

  std::string errMsg;
  int rc = llvm::sys::ExecuteAndWait(resolvedPython, args, envRefs, {}, 0, 0, &errMsg);
  if (rc != 0) {
    error = "triton script execution failed with exit code " + std::to_string(rc);
    if (!errMsg.empty())
      error += ": " + errMsg;
    return false;
  }

  auto latest = findLatestNpuirUnder(tempDir);
  if (!latest) {
    error = "triton script completed but no .npuir.mlir dump was found under " + tempDir;
    return false;
  }

  outNpuirPath = *latest;
  return true;
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "TritonSim HIVM analysis tool\n");

  if (npuirFile.empty() == tritonScript.empty()) {
    llvm::errs() << "Exactly one of --npuir-file or --triton-script must be provided.\n";
    return 1;
  }

  if (!hardwareConfig.empty()) {
    std::string loadError;
    if (!loadHardwareConfigFromFile(hardwareConfig, loadError)) {
      llvm::errs() << loadError << "\n";
      return 1;
    }
  }

  std::string npuirPath = npuirFile;
  std::string tempDumpDir;
  if (!tritonScript.empty()) {
    std::string dumpError;
    if (!runTritonScriptToDump(npuirPath, tempDumpDir, dumpError)) {
      llvm::errs() << dumpError << "\n";
      return 1;
    }
  }

  auto parsedSchedulerMode = parseSchedulerMode(schedulerMode);
  if (!parsedSchedulerMode) {
    llvm::errs() << "invalid --scheduler value `" << schedulerMode
                 << "`; expected `static` or `des`\n";
    return 1;
  }

  HIVMAnalyzer analyzer(getHardwareConfig(), argBindings, *parsedSchedulerMode);
  HIVMAnalysisReport report;
  std::string error;
  if (!analyzer.analyzeFile(npuirPath, report, error)) {
    llvm::errs() << error << "\n";
    return 1;
  }

  if (!tritonScript.empty()) {
    report.sourceMode = "triton-dsl";
    report.sourcePath = npuirPath;
  }

  report.print(llvm::outs(), getHardwareConfig());

  if (!perfettoTraceFile.empty()) {
    std::error_code ec;
    llvm::raw_fd_ostream traceOS(perfettoTraceFile, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      llvm::errs() << "failed to open Perfetto trace file " << perfettoTraceFile
                   << ": " << ec.message() << "\n";
      return 1;
    }
    report.emitPerfettoTrace(traceOS, getHardwareConfig());
    llvm::outs() << "\nPerfetto trace: " << perfettoTraceFile << "\n";
  }

  if (!desGraphFile.empty()) {
    std::error_code ec;
    llvm::raw_fd_ostream graphOS(desGraphFile, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      llvm::errs() << "failed to open DES graph file " << desGraphFile
                   << ": " << ec.message() << "\n";
      return 1;
    }
    report.emitDESGraph(graphOS, getHardwareConfig());
    llvm::outs() << "\nDES graph: " << desGraphFile << "\n";
  }

  if (!tempDumpDir.empty() && keepDumpDir) {
    llvm::outs() << "\nKept dump directory: " << tempDumpDir << "\n";
  } else if (!tempDumpDir.empty()) {
    std::error_code ec = llvm::sys::fs::remove_directories(tempDumpDir);
    if (ec)
      llvm::errs() << "Warning: failed to remove temporary dump dir " << tempDumpDir
                   << ": " << ec.message() << "\n";
  }

  return 0;
}
