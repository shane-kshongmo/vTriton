//===- PassRegistration.cpp - Register AscendModel passes ----------------===//
//
// This file registers the AscendModel pipeline with configurable options.
//
// Usage (options are space-separated, NOT comma-separated):
//   tritonsim-opt -ascend-perf-model="arg-bindings=arg2=100,pid_x=0" input.mlir
//   tritonsim-opt -ascend-perf-model="loop-trip-counts=4,6588" input.mlir
//   tritonsim-opt -ascend-perf-model="hardware-config=path/to/config.json" input.mlir
//   tritonsim-opt -ascend-perf-model="hardware-config=config.json arg-bindings=pid_x=0" input.mlir
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Transforms/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir {
namespace ascend {

//===----------------------------------------------------------------------===//
// Pipeline Options
//===----------------------------------------------------------------------===//

/// Options for the ascend-perf-model pipeline.
struct AscendPerfModelPipelineOptions
    : public PassPipelineOptions<AscendPerfModelPipelineOptions> {
  
  /// Bindings for function arguments and program_ids.
  /// Format: "arg2=100,arg3=200,pid_x=0"
  PassOptions::Option<std::string> argBindings{
      *this, "arg-bindings",
      llvm::cl::desc("Bindings for args and program_ids (e.g., 'arg2=100,pid_x=0')"),
      llvm::cl::init("")};
  
  /// Direct loop trip count overrides.
  /// Format: "4,6588" (first loop=4, second loop=6588)
  PassOptions::Option<std::string> loopTripCounts{
      *this, "loop-trip-counts",
      llvm::cl::desc("Direct loop trip count overrides in order (e.g., '4,6588')"),
      llvm::cl::init("")};
  
  /// Hardware config JSON file path.
  PassOptions::Option<std::string> hardwareConfig{
      *this, "hardware-config",
      llvm::cl::desc("Path to hardware configuration JSON file"),
      llvm::cl::init("")};
  
  /// Enable tiling optimization search.
  PassOptions::Option<bool> optimizeTiling{
      *this, "optimize-tiling",
      llvm::cl::desc("Enable tiling parameter optimization search"),
      llvm::cl::init(false)};
  
  /// Tiling search strategy.
  PassOptions::Option<std::string> tilingStrategy{
      *this, "tiling-strategy",
      llvm::cl::desc("Tiling search strategy: exhaustive, heuristic, or genetic"),
      llvm::cl::init("heuristic")};
};

//===----------------------------------------------------------------------===//
// Pipeline Registration
//===----------------------------------------------------------------------===//

void registerAscendModelPipeline() {
  PassPipelineRegistration<AscendPerfModelPipelineOptions>(
      "ascend-perf-model",
      "Run the full Ascend 910B performance modeling pipeline",
      [](OpPassManager &pm, const AscendPerfModelPipelineOptions &options) {
        // Step 1: Convert Triton IR to AscendModel IR
        pm.addPass(createConvertTritonToAscendPass());
        
        // Step 2: Insert data transfers between Cube and Vector paths
        pm.addPass(createInsertDataTransfersPass());
        
        // Step 3: Assign unique IDs to all operations
        pm.addPass(createAssignOpIDsPass());
        
        // Step 4: Estimate execution cycles based on hardware config
        // Create options and pass to the pass
        {
          EstimateCyclesPassOptions estimateOpts;
          estimateOpts.argBindingsStr = options.argBindings;
          estimateOpts.loopTripCountsStr = options.loopTripCounts;
          estimateOpts.hardwareConfigPath = options.hardwareConfig;
          pm.addPass(createEstimateCyclesPass(estimateOpts));
        }
        
        // Step 5: Perform pipeline scheduling analysis
        {
          PipelineAnalysisPassOptions pipelineOpts;
          pipelineOpts.argBindingsStr = options.argBindings;
          pipelineOpts.loopTripCountsStr = options.loopTripCounts;
          pipelineOpts.hardwareConfigPath = options.hardwareConfig;
          pm.addPass(createPipelineAnalysisPass(pipelineOpts));
        }
        
        // Step 6 (optional): Tiling optimization search
        if (options.optimizeTiling) {
          TilingOptimizationPassOptions tilingOpts;
          tilingOpts.strategy = options.tilingStrategy;
          tilingOpts.hardwareConfigPath = options.hardwareConfig;
          tilingOpts.verbose = true;
          pm.addPass(createTilingOptimizationPass(tilingOpts));
        }
        
        // Step 7: Generate performance report
        pm.addPass(createPerfReportPass());
      });
}

} // namespace ascend
} // namespace mlir
