#===----------------------------------------------------------------------===//
# Triton Dialect Libraries (Built from Triton-Ascend Source)
#
# 编译两个静态库:
#   TritonDialectIR   <- lib/Dialect/Triton/IR/   (4 files)
#   TritonGPUDialectIR <- lib/Dialect/TritonGPU/IR/ (4 files)
#
# Traits.cpp (in TritonIR) depends on TritonGPU's BlockedEncodingAttr,
# CTALayoutAttr etc, so both must be compiled.
#
# 模式A (推荐): 复用已有 .inc (需要 TRITON_BUILD_DIR)
# 模式B: 自行 mlir-tblgen (只需 TRITON_SRC_DIR)
#===----------------------------------------------------------------------===//

message(STATUS "=== Configuring Triton Dialects ===")

set(TRITON_IR_INC_DIR "${TRITON_INCLUDE_DIR}/triton/Dialect/Triton/IR")
set(TRITON_IR_LIB_DIR "${TRITON_SRC_DIR}/lib/Dialect/Triton/IR")
set(TRITON_GPU_INC_DIR "${TRITON_INCLUDE_DIR}/triton/Dialect/TritonGPU/IR")
set(TRITON_GPU_LIB_DIR "${TRITON_SRC_DIR}/lib/Dialect/TritonGPU/IR")

#===----------------------------------------------------------------------===//
# .inc 文件来源
#===----------------------------------------------------------------------===//

set(TRITON_GEN_INC_DIR "")

if(TRITON_BUILD_DIR)
  set(_pre "${TRITON_BUILD_DIR}/include")
  if(EXISTS "${_pre}/triton/Dialect/Triton/IR/Dialect.h.inc" AND
     EXISTS "${_pre}/triton/Dialect/TritonGPU/IR/Dialect.h.inc")
    set(TRITON_GEN_INC_DIR "${_pre}")
    message(STATUS "  Mode A: Reusing pre-generated .inc from TRITON_BUILD_DIR")
    add_custom_target(TritonDialectIncGen)
    add_custom_target(TritonGPUIncGen)
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

  # Helper macro: add_tblgen_rule(output_dir, td_file, out_file, flags...)
  macro(add_tblgen_rule _dir _td _file)
    set(_flags ${ARGN})
    add_custom_command(OUTPUT ${_dir}/${_file}
      COMMAND ${MLIR_TABLEGEN_EXE} ${_td} ${_flags} ${_I} -o ${_dir}/${_file}
      DEPENDS ${_td}
      COMMENT "[Triton-tblgen] ${_file}")
  endmacro()

  #--- Triton IR (14 .inc files) ---
  set(_out "${TRITON_GEN_INC_DIR}/triton/Dialect/Triton/IR")
  file(MAKE_DIRECTORY ${_out})
  set(_td "${TRITON_IR_INC_DIR}")
  set(_tt_gen "")

  # TritonOps.td -> 4 files
  add_tblgen_rule(${_out} "${_td}/TritonOps.td" "Ops.h.inc"       -gen-op-decls)
  add_tblgen_rule(${_out} "${_td}/TritonOps.td" "Ops.cpp.inc"     -gen-op-defs)
  add_tblgen_rule(${_out} "${_td}/TritonOps.td" "OpsEnums.h.inc"  -gen-enum-decls)
  add_tblgen_rule(${_out} "${_td}/TritonOps.td" "OpsEnums.cpp.inc" -gen-enum-defs)
  list(APPEND _tt_gen ${_out}/Ops.h.inc ${_out}/Ops.cpp.inc
    ${_out}/OpsEnums.h.inc ${_out}/OpsEnums.cpp.inc)

  # TritonDialect.td -> 2 files
  add_tblgen_rule(${_out} "${_td}/TritonDialect.td" "Dialect.h.inc"   -gen-dialect-decls)
  add_tblgen_rule(${_out} "${_td}/TritonDialect.td" "Dialect.cpp.inc" -gen-dialect-defs)
  list(APPEND _tt_gen ${_out}/Dialect.h.inc ${_out}/Dialect.cpp.inc)

  # TritonTypes.td -> 2 files
  add_tblgen_rule(${_out} "${_td}/TritonTypes.td" "Types.h.inc"   -gen-typedef-decls)
  add_tblgen_rule(${_out} "${_td}/TritonTypes.td" "Types.cpp.inc" -gen-typedef-defs)
  list(APPEND _tt_gen ${_out}/Types.h.inc ${_out}/Types.cpp.inc)

  # TritonInterfaces.td -> AttrInterfaces 2 files
  add_tblgen_rule(${_out} "${_td}/TritonInterfaces.td"
    "AttrInterfaces.h.inc"   -gen-attr-interface-decls)
  add_tblgen_rule(${_out} "${_td}/TritonInterfaces.td"
    "AttrInterfaces.cpp.inc" -gen-attr-interface-defs)
  list(APPEND _tt_gen ${_out}/AttrInterfaces.h.inc ${_out}/AttrInterfaces.cpp.inc)

  # TritonTypeInterfaces.td -> 2 files
  add_tblgen_rule(${_out} "${_td}/TritonTypeInterfaces.td"
    "TritonTypeInterfaces.h.inc"   -gen-type-interface-decls)
  add_tblgen_rule(${_out} "${_td}/TritonTypeInterfaces.td"
    "TritonTypeInterfaces.cpp.inc" -gen-type-interface-defs)
  list(APPEND _tt_gen
    ${_out}/TritonTypeInterfaces.h.inc ${_out}/TritonTypeInterfaces.cpp.inc)

  # TritonOpInterfaces.td -> 2 files
  add_tblgen_rule(${_out} "${_td}/TritonOpInterfaces.td"
    "OpInterfaces.h.inc"   -gen-op-interface-decls)
  add_tblgen_rule(${_out} "${_td}/TritonOpInterfaces.td"
    "OpInterfaces.cpp.inc" -gen-op-interface-defs)
  list(APPEND _tt_gen ${_out}/OpInterfaces.h.inc ${_out}/OpInterfaces.cpp.inc)

  add_custom_target(TritonDialectIncGen DEPENDS ${_tt_gen})

  #--- TritonGPU IR (12 .inc files) ---
  set(_gout "${TRITON_GEN_INC_DIR}/triton/Dialect/TritonGPU/IR")
  file(MAKE_DIRECTORY ${_gout})
  set(_gtd "${TRITON_GPU_INC_DIR}")
  set(_ttg_gen "")

  # TritonGPUOps.td -> 6 files
  add_tblgen_rule(${_gout} "${_gtd}/TritonGPUOps.td"
    "Dialect.h.inc"   -gen-dialect-decls -dialect=triton_gpu)
  add_tblgen_rule(${_gout} "${_gtd}/TritonGPUOps.td"
    "Dialect.cpp.inc" -gen-dialect-defs  -dialect=triton_gpu)
  add_tblgen_rule(${_gout} "${_gtd}/TritonGPUOps.td"
    "Ops.h.inc"   -gen-op-decls)
  add_tblgen_rule(${_gout} "${_gtd}/TritonGPUOps.td"
    "Ops.cpp.inc" -gen-op-defs)
  add_tblgen_rule(${_gout} "${_gtd}/TritonGPUOps.td"
    "Types.h.inc"   -gen-typedef-decls -typedefs-dialect=triton_gpu)
  add_tblgen_rule(${_gout} "${_gtd}/TritonGPUOps.td"
    "Types.cpp.inc" -gen-typedef-defs  -typedefs-dialect=triton_gpu)
  list(APPEND _ttg_gen
    ${_gout}/Dialect.h.inc ${_gout}/Dialect.cpp.inc
    ${_gout}/Ops.h.inc ${_gout}/Ops.cpp.inc
    ${_gout}/Types.h.inc ${_gout}/Types.cpp.inc)

  # TritonGPUAttrDefs.td -> 6 files
  add_tblgen_rule(${_gout} "${_gtd}/TritonGPUAttrDefs.td"
    "TritonGPUAttrInterfaces.h.inc"   -gen-attr-interface-decls)
  add_tblgen_rule(${_gout} "${_gtd}/TritonGPUAttrDefs.td"
    "TritonGPUAttrInterfaces.cpp.inc" -gen-attr-interface-defs)
  add_tblgen_rule(${_gout} "${_gtd}/TritonGPUAttrDefs.td"
    "TritonGPUAttrDefs.h.inc"   -gen-attrdef-decls)
  add_tblgen_rule(${_gout} "${_gtd}/TritonGPUAttrDefs.td"
    "TritonGPUAttrDefs.cpp.inc" -gen-attrdef-defs)
  add_tblgen_rule(${_gout} "${_gtd}/TritonGPUAttrDefs.td"
    "OpsEnums.h.inc"  -gen-enum-decls)
  add_tblgen_rule(${_gout} "${_gtd}/TritonGPUAttrDefs.td"
    "OpsEnums.cpp.inc" -gen-enum-defs)
  list(APPEND _ttg_gen
    ${_gout}/TritonGPUAttrInterfaces.h.inc ${_gout}/TritonGPUAttrInterfaces.cpp.inc
    ${_gout}/TritonGPUAttrDefs.h.inc ${_gout}/TritonGPUAttrDefs.cpp.inc
    ${_gout}/OpsEnums.h.inc ${_gout}/OpsEnums.cpp.inc)

  add_custom_target(TritonGPUIncGen DEPENDS ${_ttg_gen})
endif()

#===----------------------------------------------------------------------===//
# TritonGPUDialectIR (must be built first — TritonIR depends on it)
#===----------------------------------------------------------------------===//

add_library(TritonGPUDialectIR STATIC
  ${TRITON_GPU_LIB_DIR}/Dialect.cpp
  ${TRITON_GPU_LIB_DIR}/LinearLayoutConversions.cpp
  ${TRITON_GPU_LIB_DIR}/Ops.cpp
  ${TRITON_GPU_LIB_DIR}/Types.cpp
)

add_dependencies(TritonGPUDialectIR TritonDialectIncGen TritonGPUIncGen)
llvm_update_compile_flags(TritonGPUDialectIR)

target_include_directories(TritonGPUDialectIR PUBLIC
  ${TRITON_INCLUDE_DIR}
  ${TRITON_GEN_INC_DIR}
)

# 与 Triton 原始 CMake 一致: LINK_LIBS PUBLIC MLIRGPUDialect TritonIR TritonTools
# TritonIR 会产生循环依赖，所以这里只链接 MLIR 库
# TritonTools 的功能我们通过链接更多 MLIR 库来补偿
target_link_libraries(TritonGPUDialectIR PUBLIC
  MLIRIR
  MLIRSupport
  MLIRGPUDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
  LLVMSupport
)

#===----------------------------------------------------------------------===//
# TritonDialectIR
#===----------------------------------------------------------------------===//

add_library(TritonDialectIR STATIC
  ${TRITON_IR_LIB_DIR}/Dialect.cpp
  ${TRITON_IR_LIB_DIR}/Ops.cpp
  ${TRITON_IR_LIB_DIR}/Types.cpp
  ${TRITON_IR_LIB_DIR}/Traits.cpp
)

add_dependencies(TritonDialectIR TritonDialectIncGen TritonGPUIncGen)
llvm_update_compile_flags(TritonDialectIR)

target_include_directories(TritonDialectIR PUBLIC
  ${TRITON_INCLUDE_DIR}
  ${TRITON_GEN_INC_DIR}
)

# 与 Triton 原始 CMake 一致: LINK_LIBS PUBLIC MLIRIR MLIRArithDialect MLIRMathDialect MLIRSCFDialect
# 加上 TritonGPUDialectIR 解决 Traits.cpp 的依赖
target_link_libraries(TritonDialectIR PUBLIC
  TritonGPUDialectIR
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)

# 导出
set(TRITON_GEN_INC_DIR ${TRITON_GEN_INC_DIR} PARENT_SCOPE)

message(STATUS "  TritonDialectIR:    Dialect.cpp Ops.cpp Types.cpp Traits.cpp")
message(STATUS "  TritonGPUDialectIR: Dialect.cpp LinearLayoutConversions.cpp Ops.cpp Types.cpp")
message(STATUS "=== Triton Dialects configured ===")
