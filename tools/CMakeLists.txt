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

  if(TRITONSIM_HAS_BISHENGIR_HIVM)
    target_link_libraries(tritonsim-opt PRIVATE ${TRITONSIM_BISHENGIR_HIVM_LIBS})
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
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
  LLVMSupport
)

if(TRITONSIM_HAS_BISHENGIR_HIVM)
  target_link_libraries(tritonsim-hivm
    PRIVATE
    ${TRITONSIM_BISHENGIR_HIVM_LIBS}
  )
  target_include_directories(tritonsim-hivm
    PRIVATE
    ${TRITONSIM_BISHENGIR_INCLUDE_DIRS}
  )
endif()

mlir_check_all_link_libraries(tritonsim-hivm)

#===----------------------------------------------------------------------===//
# hivm-crud - HIVM CRUD Tool (Create/Read/Update/Delete HIVM ops via C++ API)
#===----------------------------------------------------------------------===//

if(TRITONSIM_HAS_BISHENGIR_HIVM AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/hivm-crud/hivm-crud.cpp")
  add_llvm_executable(hivm-crud
    hivm-crud/hivm-crud.cpp
  )

  target_link_libraries(hivm-crud
    PRIVATE
    AscendModelTransforms
    ${TRITONSIM_BISHENGIR_HIVM_LIBS}
    MLIRIR
    MLIRParser
    MLIRSupport
    LLVMSupport
  )

  target_include_directories(hivm-crud
    PRIVATE
    ${TRITONSIM_BISHENGIR_INCLUDE_DIRS}
  )

  mlir_check_all_link_libraries(hivm-crud)
  list(APPEND _installed_tools hivm-crud)
endif()

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
