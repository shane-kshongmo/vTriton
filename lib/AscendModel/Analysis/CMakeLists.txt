add_mlir_library(AscendModelAnalysis
  PARTIAL_SOURCES_INTENDED
  HardwareConfig.cpp
  PipelineAnalysis.cpp
  UnifiedTilingCostModel.cpp
  MemoryTilingOptimizer.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/AscendModel/Analysis
  
  DEPENDS
  AscendModelOpsIncGen
  
  LINK_LIBS PUBLIC
  AscendModelIR
  MLIRIR
  MLIRSupport
  LLVMSupport
)
