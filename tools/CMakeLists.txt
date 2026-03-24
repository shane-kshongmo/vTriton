#===----------------------------------------------------------------------===//
# tools/CMakeLists.txt - Build tritonsim-opt and ascend-tiling-opt
#
# Two tools sharing the same modeling library:
# - tritonsim-opt: Performance modeling (reads MLIR, outputs cycle estimates)  
# - ascend-tiling-opt: Tiling optimization (finds optimal tile configurations)
#===----------------------------------------------------------------------===//

#===----------------------------------------------------------------------===//
# tritonsim-opt - Performance Modeling Tool
#===----------------------------------------------------------------------===//

add_llvm_executable(tritonsim-opt
  tritonsim-opt/tritonsim-opt.cpp
)

target_link_libraries(tritonsim-opt
  PRIVATE
  AscendModelIR
  AscendModelTransforms
  AscendModelAnalysis

  MLIROptLib
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRTensorDialect
  LLVMSupport
)

if(TRITONSIM_HAS_TRITON)
  target_link_libraries(tritonsim-opt PRIVATE ${TRITON_DIALECT_LIB})
endif()

mlir_check_all_link_libraries(tritonsim-opt)

#===----------------------------------------------------------------------===//
# ascend-tiling-opt - Tiling Optimization Tool
#===----------------------------------------------------------------------===//

add_llvm_executable(ascend-tiling-opt
  ascend-tiling-opt/ascend-tiling-opt.cpp
)

target_link_libraries(ascend-tiling-opt
  PRIVATE
  AscendModelIR
  AscendModelTransforms
  AscendModelAnalysis

  MLIROptLib
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRArithDialect
  MLIRMathDialect
)

mlir_check_all_link_libraries(ascend-tiling-opt)

#===----------------------------------------------------------------------===//
# Install
#===----------------------------------------------------------------------===//

install(TARGETS tritonsim-opt ascend-tiling-opt
  RUNTIME DESTINATION bin
)
