#===- CMakeLists.txt - HIVM Tests ----------------------------------------===//

add_executable(HivmOpsEditorTest
  HivmOpsEditorTest.cpp
)

if(TRITONSIM_HAS_BISHENGIR_HIVM)
  target_link_libraries(HivmOpsEditorTest
    PRIVATE
    # 先链接 MLIR 基础库
    MLIRIR
    MLIRParser
    MLIRSupport
    LLVMSupport
    # 再链接 BiShengIR 库
    ${TRITONSIM_BISHENGIR_HIVM_LIBS}
    # 最后链接 vTriton 库
    AscendModelTransforms
    AscendModelAnalysis
    AscendModelIR
  )
  
  add_test(NAME HivmOpsEditorTest
    COMMAND HivmOpsEditorTest
      ${CMAKE_SOURCE_DIR}/kernel.hivm.milr
  )
  
  set_tests_properties(HivmOpsEditorTest PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
else()
  target_link_libraries(HivmOpsEditorTest
    PRIVATE
    AscendModelTransforms
    AscendModelAnalysis
    AscendModelIR
    MLIRIR
    MLIRParser
    MLIRSupport
    LLVMSupport
  )
endif()
