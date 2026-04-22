#===----------------------------------------------------------------------===//
# Triton Dialect Library (Built from Triton-Ascend Source)
#
# Only builds the core Triton dialect (not TritonGPU/TritonNvidiaGPU).
# Traits.cpp is excluded because it depends on TritonGPU headers which are
# incompatible with LLVM 20 and not needed for the modelling pipeline.
#
# Mode A (recommended): reuse pre-generated .inc (needs TRITON_BUILD_DIR)
# Mode B: run mlir-tblgen (needs only TRITON_SRC_DIR)
#===----------------------------------------------------------------------===//

message(STATUS "=== Configuring Triton Dialects ===")

set(TRITON_IR_INC_DIR "${TRITON_INCLUDE_DIR}/triton/Dialect/Triton/IR")
set(TRITON_IR_LIB_DIR "${TRITON_SRC_DIR}/lib/Dialect/Triton/IR")

#===----------------------------------------------------------------------===//
# .inc file source
#===----------------------------------------------------------------------===//

set(TRITON_GEN_INC_DIR "")

if(TRITON_BUILD_DIR)
  set(_pre "${TRITON_BUILD_DIR}/include")
  if(EXISTS "${_pre}/triton/Dialect/Triton/IR/Dialect.h.inc")
    set(TRITON_GEN_INC_DIR "${_pre}")
    message(STATUS "  Mode A: Reusing pre-generated .inc from TRITON_BUILD_DIR")
    add_custom_target(TritonDialectIncGen)
  endif()
endif()

if(NOT TRITON_GEN_INC_DIR)
  message(STATUS "  Mode B: Running mlir-tblgen")
  set(TRITON_GEN_INC_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")

  find_program(MLIR_TABLEGEN_EXE "mlir-tblgen"
    HINTS "${LLVM_TOOLS_BINARY_DIR}" "${MLIR_DIR}/../../../bin")
  if(NOT MLIR_TABLEGEN_EXE)
    message(FATAL_ERROR "mlir-tblgen not found")
  endif()

  set(_I -I${TRITON_INCLUDE_DIR} -I${MLIR_INCLUDE_DIRS} -I${LLVM_INCLUDE_DIRS})

  macro(add_tblgen_rule _dir _td _file)
    set(_flags ${ARGN})
    add_custom_command(OUTPUT ${_dir}/${_file}
      COMMAND ${MLIR_TABLEGEN_EXE} ${_td} ${_flags} ${_I} -o ${_dir}/${_file}
      DEPENDS ${_td}
      COMMENT "[Triton-tblgen] ${_file}")
  endmacro()

  set(_out "${TRITON_GEN_INC_DIR}/triton/Dialect/Triton/IR")
  file(MAKE_DIRECTORY ${_out})
  set(_td "${TRITON_IR_INC_DIR}")
  set(_tt_gen "")

  add_tblgen_rule(${_out} "${_td}/TritonOps.td" "Ops.h.inc"       -gen-op-decls)
  add_tblgen_rule(${_out} "${_td}/TritonOps.td" "Ops.cpp.inc"     -gen-op-defs)
  add_tblgen_rule(${_out} "${_td}/TritonOps.td" "OpsEnums.h.inc"  -gen-enum-decls)
  add_tblgen_rule(${_out} "${_td}/TritonOps.td" "OpsEnums.cpp.inc" -gen-enum-defs)
  list(APPEND _tt_gen ${_out}/Ops.h.inc ${_out}/Ops.cpp.inc
    ${_out}/OpsEnums.h.inc ${_out}/OpsEnums.cpp.inc)

  add_tblgen_rule(${_out} "${_td}/TritonDialect.td" "Dialect.h.inc"   -gen-dialect-decls)
  add_tblgen_rule(${_out} "${_td}/TritonDialect.td" "Dialect.cpp.inc" -gen-dialect-defs)
  list(APPEND _tt_gen ${_out}/Dialect.h.inc ${_out}/Dialect.cpp.inc)

  add_tblgen_rule(${_out} "${_td}/TritonTypes.td" "Types.h.inc"   -gen-typedef-decls)
  add_tblgen_rule(${_out} "${_td}/TritonTypes.td" "Types.cpp.inc" -gen-typedef-defs)
  list(APPEND _tt_gen ${_out}/Types.h.inc ${_out}/Types.cpp.inc)

  add_tblgen_rule(${_out} "${_td}/TritonInterfaces.td"
    "AttrInterfaces.h.inc"   -gen-attr-interface-decls)
  add_tblgen_rule(${_out} "${_td}/TritonInterfaces.td"
    "AttrInterfaces.cpp.inc" -gen-attr-interface-defs)
  list(APPEND _tt_gen ${_out}/AttrInterfaces.h.inc ${_out}/AttrInterfaces.cpp.inc)

  add_tblgen_rule(${_out} "${_td}/TritonTypeInterfaces.td"
    "TritonTypeInterfaces.h.inc"   -gen-type-interface-decls)
  add_tblgen_rule(${_out} "${_td}/TritonTypeInterfaces.td"
    "TritonTypeInterfaces.cpp.inc" -gen-type-interface-defs)
  list(APPEND _tt_gen
    ${_out}/TritonTypeInterfaces.h.inc ${_out}/TritonTypeInterfaces.cpp.inc)

  add_tblgen_rule(${_out} "${_td}/TritonOpInterfaces.td"
    "OpInterfaces.h.inc"   -gen-op-interface-decls)
  add_tblgen_rule(${_out} "${_td}/TritonOpInterfaces.td"
    "OpInterfaces.cpp.inc" -gen-op-interface-defs)
  list(APPEND _tt_gen ${_out}/OpInterfaces.h.inc ${_out}/OpInterfaces.cpp.inc)

  add_custom_target(TritonDialectIncGen DEPENDS ${_tt_gen})
endif()

#===----------------------------------------------------------------------===//
# TritonDialectIR (core only, no Traits.cpp/TritonGPU)
#===----------------------------------------------------------------------===//

add_library(TritonDialectIR STATIC
  ${TRITON_IR_LIB_DIR}/Dialect.cpp
  ${TRITON_IR_LIB_DIR}/Ops.cpp
  ${TRITON_IR_LIB_DIR}/Types.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TraitsStub.cpp
)

add_dependencies(TritonDialectIR TritonDialectIncGen)
llvm_update_compile_flags(TritonDialectIR)

target_include_directories(TritonDialectIR PUBLIC
  ${TRITON_INCLUDE_DIR}
  ${TRITON_GEN_INC_DIR}
)

target_link_libraries(TritonDialectIR PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)

# Export
set(TRITON_GEN_INC_DIR ${TRITON_GEN_INC_DIR} PARENT_SCOPE)

message(STATUS "  TritonDialectIR: Dialect.cpp Ops.cpp Types.cpp")
message(STATUS "=== Triton Dialects configured ===")
