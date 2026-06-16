add_mlir_library(AscendModelTransforms
  PARTIAL_SOURCES_INTENDED
  ConvertTritonToAscend.cpp
  InsertDataTransfers.cpp
  AssignOpIDs.cpp
  EstimateCycles.cpp
  ExtractTTIRInfo.cpp
  HIVMAnalysisPass.cpp
  PipelineAnalysisPass.cpp
  PerfReportPass.cpp
  PassRegistration.cpp
  TilingOptimizationPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/AscendModel/Transforms

  DEPENDS
  AscendModelOpsIncGen
  AscendModelTransformsPassIncGen

  LINK_LIBS PUBLIC
  AscendModelIR
  AscendModelAnalysis
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  MLIRSCFDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRTensorDialect
)

if(TRITONSIM_HAS_BISHENGIR_HIVM)
  target_sources(obj.AscendModelTransforms PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/HivmOpsEditor.cpp
  )
  target_include_directories(obj.AscendModelTransforms SYSTEM PUBLIC
    ${TRITONSIM_BISHENGIR_INCLUDE_DIRS}
  )
  target_link_libraries(obj.AscendModelTransforms PUBLIC
    ${TRITONSIM_BISHENGIR_HIVM_LIBS}
    MLIRMemRefDialect
    MLIRFuncDialect
    MLIRDestinationStyleOpInterface
    MLIRSideEffectInterfaces
    MLIRInferTypeOpInterface
    MLIRBufferizationDialect
    MLIRDLTIDialect
    MLIRBytecodeOpInterface
    MLIRFunctionInterfaces
    MLIRControlFlowInterfaces
    MLIRDataLayoutInterfaces
  )
endif()

if(TRITONSIM_HAS_TRITON AND TRITON_DIALECT_LIB)
  add_dependencies(obj.AscendModelTransforms TritonDialectIncGen)
  target_link_libraries(obj.AscendModelTransforms PRIVATE ${TRITON_DIALECT_LIB})
endif()
