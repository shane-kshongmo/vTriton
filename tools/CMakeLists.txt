#===----------------------------------------------------------------------===//
# tools/CMakeLists.txt - Build tritonsim-opt and ascend-tiling-opt
#
# Two tools sharing the same modeling library:
# - tritonsim-opt: Performance modeling (reads MLIR, outputs cycle estimates)  
# - ascend-tiling-opt: Tiling optimization (finds optimal tile configurations)
#===----------------------------------------------------------------------===//

set(_installed_tools tritonsim-hivm)

#===----------------------------------------------------------------------===//
# tritonsim-opt - Performance Modeling Tool
#===----------------------------------------------------------------------===//

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tritonsim-opt/tritonsim-opt.cpp")
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
  list(APPEND _installed_tools tritonsim-opt)
endif()

#===----------------------------------------------------------------------===//
# tritonsim-hivm - HIVM-based Performance Modeling Tool
#===----------------------------------------------------------------------===//

add_llvm_executable(tritonsim-hivm
  tritonsim-hivm/tritonsim-hivm.cpp
)

target_link_libraries(tritonsim-hivm
  PRIVATE
  AscendModelAnalysis
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
  LLVMSupport
)

mlir_check_all_link_libraries(tritonsim-hivm)

#===----------------------------------------------------------------------===//
# ascend-tiling-opt - Tiling Optimization Tool
#===----------------------------------------------------------------------===//

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ascend-tiling-opt/ascend-tiling-opt.cpp")
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
  list(APPEND _installed_tools ascend-tiling-opt)
endif()

#===----------------------------------------------------------------------===//
# Install
#===----------------------------------------------------------------------===//

install(TARGETS ${_installed_tools}
  RUNTIME DESTINATION bin
)
