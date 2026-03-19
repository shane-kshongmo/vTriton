cmake_minimum_required(VERSION 3.20)
project(AscendPerfModel LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#===----------------------------------------------------------------------===//
# Build Options
#===----------------------------------------------------------------------===//

option(TRITONSIM_ENABLE_TRITON "Enable Triton dialect support" ON)
option(TRITONSIM_AUTO_BUILD_LLVM
  "Auto-build LLVM/MLIR from submodule if not found" ON)

set(TRITON_SRC_DIR "" CACHE PATH "Path to triton-ascend source root")
set(TRITON_BUILD_DIR "" CACHE PATH
    "Path to triton cmake build dir (optional, reuses generated .inc files)")

#===----------------------------------------------------------------------===//
# LLVM/MLIR Discovery
#
# 优先级:
#   1. 用户指定 -DMLIR_DIR          → 直接使用
#   2. submodule 已构建             → 自动检测
#   3. TRITONSIM_AUTO_BUILD_LLVM=ON → 自动构建 submodule
#   4. 报错，提示操作方法
#===----------------------------------------------------------------------===//

set(LLVM_SUBMODULE_SRC "${CMAKE_SOURCE_DIR}/thirdparty/llvm-project")
set(LLVM_SUBMODULE_BUILD "${LLVM_SUBMODULE_SRC}/build")
set(LLVM_SUBMODULE_INSTALL "${LLVM_SUBMODULE_BUILD}/install")

if(NOT MLIR_DIR)
  # ── 自动检测: submodule 是否已构建? ──
  if(EXISTS "${LLVM_SUBMODULE_INSTALL}/lib/cmake/mlir/MLIRConfig.cmake")
    set(MLIR_DIR "${LLVM_SUBMODULE_INSTALL}/lib/cmake/mlir" CACHE PATH "" FORCE)
    message(STATUS "[auto] Found MLIR from submodule: ${MLIR_DIR}")

  # ── 自动构建 ──
  elseif(TRITONSIM_AUTO_BUILD_LLVM)
    # 检查 submodule 是否存在
    if(NOT EXISTS "${LLVM_SUBMODULE_SRC}/llvm/CMakeLists.txt")
      message(FATAL_ERROR
        "LLVM submodule not initialized.\n"
        "Run:\n"
        "  git submodule update --init thirdparty/llvm-project\n"
        "Or provide:\n"
        "  cmake -DMLIR_DIR=<path> ..")
    endif()

    message(STATUS "")
    message(STATUS "============================================================")
    message(STATUS "  Building LLVM/MLIR from submodule (first time only)")
    message(STATUS "  This may take 30-60 minutes...")
    message(STATUS "  To skip: cmake -DMLIR_DIR=<path> ..")
    message(STATUS "  Or run manually: scripts/build_llvm.sh")
    message(STATUS "============================================================")
    message(STATUS "")

    # 检测并行度
    include(ProcessorCount)
    ProcessorCount(NPROC)
    if(NPROC EQUAL 0)
      set(NPROC 4)
    endif()

    # Configure LLVM
    execute_process(
      COMMAND ${CMAKE_COMMAND}
        -G Ninja
        -DCMAKE_BUILD_TYPE=Release
        -DLLVM_ENABLE_ASSERTIONS=ON
        "-DLLVM_ENABLE_PROJECTS=mlir;llvm"
        "-DLLVM_TARGETS_TO_BUILD=host;NVPTX;AMDGPU"
        -DCMAKE_INSTALL_PREFIX=${LLVM_SUBMODULE_INSTALL}
        -S ${LLVM_SUBMODULE_SRC}/llvm
        -B ${LLVM_SUBMODULE_BUILD}
      RESULT_VARIABLE _rc
    )
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR
        "LLVM cmake configure failed (rc=${_rc}).\n"
        "Check output above for details.\n"
        "You can also build manually: scripts/build_llvm.sh")
    endif()

    # Build + Install
    execute_process(
      COMMAND ${CMAKE_COMMAND} --build ${LLVM_SUBMODULE_BUILD}
        --target install -- -j${NPROC}
      RESULT_VARIABLE _rc
    )
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR
        "LLVM build failed (rc=${_rc}).\n"
        "You can also build manually: scripts/build_llvm.sh")
    endif()

    set(MLIR_DIR "${LLVM_SUBMODULE_INSTALL}/lib/cmake/mlir" CACHE PATH "" FORCE)
    message(STATUS "LLVM/MLIR built successfully: ${MLIR_DIR}")

  # ── 无法找到，报错 ──
  else()
    message(FATAL_ERROR
      "MLIR_DIR not set and auto-build disabled.\n"
      "Options:\n"
      "  1) cmake -DMLIR_DIR=<path> ..\n"
      "  2) cmake -DTRITONSIM_AUTO_BUILD_LLVM=ON ..\n"
      "  3) scripts/build_llvm.sh  (then re-run cmake)")
  endif()
endif()

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM version: ${LLVM_PACKAGE_VERSION}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

#===----------------------------------------------------------------------===//
# Triton Integration (Optional)
#
# 优先级:
#   1. 用户指定 -DTRITON_SRC_DIR             → 直接使用
#   2. submodule thirdparty/triton-ascend 存在 → 自动检测
#   3. 不启用 Triton (仅解析 ascend dialect)
#
# Triton dialect 从源码编译为静态库 (thirdparty/triton-dialect/)，
# 不链接 libtriton.so。
#===----------------------------------------------------------------------===//

set(TRITON_SUBMODULE_SRC "${CMAKE_SOURCE_DIR}/thirdparty/triton-ascend")

set(TRITONSIM_HAS_TRITON OFF)
set(TRITON_DIALECT_LIB "")

if(TRITONSIM_ENABLE_TRITON)
  # ── 自动检测 Triton 源码 ──
  if(NOT TRITON_SRC_DIR)
    if(EXISTS "${TRITON_SUBMODULE_SRC}")
      set(TRITON_SRC_DIR "${TRITON_SUBMODULE_SRC}" CACHE PATH "" FORCE)
      message(STATUS "[auto] Found Triton source from submodule: ${TRITON_SRC_DIR}")
    endif()
  endif()

  if(TRITON_SRC_DIR)
    # 查找 Triton 头文件
    set(_triton_inc_candidates
      "${TRITON_SRC_DIR}/include"
      "${TRITON_SRC_DIR}/triton/include"
    )

    set(TRITON_INCLUDE_DIR "")
    foreach(_candidate ${_triton_inc_candidates})
      if(EXISTS "${_candidate}/triton/Dialect/Triton/IR/Dialect.h")
        set(TRITON_INCLUDE_DIR "${_candidate}")
        break()
      endif()
    endforeach()

    if(NOT TRITON_INCLUDE_DIR)
      message(WARNING
        "Triton headers not found under TRITON_SRC_DIR=${TRITON_SRC_DIR}\n"
        "Looked for: triton/Dialect/Triton/IR/Dialect.h\n"
        "If using submodule, ensure it is initialized:\n"
        "  git submodule update --init --recursive thirdparty/triton-ascend\n"
        "Triton support DISABLED.")
    else()
      message(STATUS "Found Triton headers: ${TRITON_INCLUDE_DIR}")

      # 从源码编译 Triton Dialect
      add_subdirectory(thirdparty/triton-dialect)

      set(TRITONSIM_HAS_TRITON ON)
      set(TRITON_DIALECT_LIB TritonDialectIR)

      include_directories(${TRITON_INCLUDE_DIR})
      if(TRITON_GEN_INC_DIR)
        include_directories(${TRITON_GEN_INC_DIR})
      endif()

      add_definitions(-DTRITONSIM_HAS_TRITON=1)
      message(STATUS "=== Triton Integration ENABLED ===")
    endif()
  else()
    message(STATUS
      "Triton support requested but source not found.\n"
      "Building WITHOUT Triton. Tool will only parse ascend dialect.\n"
      "To enable:\n"
      "  git submodule update --init thirdparty/triton-ascend\n"
      "  Or: cmake -DTRITON_SRC_DIR=<path> ..")
  endif()
endif()

message(STATUS "TRITONSIM_HAS_TRITON: ${TRITONSIM_HAS_TRITON}")

#===----------------------------------------------------------------------===//
# Ascend910B Hardware Parameters
#===----------------------------------------------------------------------===//

set(ASCEND_CUBE_TFLOPS 320 CACHE STRING "Cube Core FP16 TFLOPS")
set(ASCEND_CUBE_M 16 CACHE STRING "Cube Core M dimension")
set(ASCEND_CUBE_N 16 CACHE STRING "Cube Core N dimension")
set(ASCEND_CUBE_K 16 CACHE STRING "Cube Core K dimension")
set(ASCEND_VECTOR_TFLOPS 10 CACHE STRING "Vector Core FP32 TFLOPS")
set(ASCEND_VECTOR_WIDTH 256 CACHE STRING "Vector width in bits")
set(ASCEND_HBM_BW 1555 CACHE STRING "HBM Bandwidth GB/s")
set(ASCEND_L2_SIZE 192 CACHE STRING "L2 Cache Size MB")
set(ASCEND_L1_SIZE 1024 CACHE STRING "L1 Buffer Size KB")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/AscendModel/HardwareParams.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/AscendModel/HardwareParams.h
)

#===----------------------------------------------------------------------===//
# Subdirectories
#===----------------------------------------------------------------------===//

set(ASCEND_MODEL_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(ASCEND_MODEL_BINARY_DIR ${PROJECT_BINARY_DIR})

add_subdirectory(include/AscendModel)
add_subdirectory(lib)
add_subdirectory(tools)

#===----------------------------------------------------------------------===//
# Testing
#===----------------------------------------------------------------------===//

option(ASCEND_MODEL_ENABLE_TESTS "Enable building tests" OFF)

if(ASCEND_MODEL_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

#===----------------------------------------------------------------------===//
# Summary
#===----------------------------------------------------------------------===//

message(STATUS "")
message(STATUS "======= AscendPerfModel Build Summary =======")
message(STATUS "  LLVM version:     ${LLVM_PACKAGE_VERSION}")
message(STATUS "  MLIR_DIR:         ${MLIR_DIR}")
message(STATUS "  Triton support:   ${TRITONSIM_HAS_TRITON}")
if(TRITONSIM_HAS_TRITON)
  message(STATUS "  Triton source:    ${TRITON_SRC_DIR}")
  if(TRITON_BUILD_DIR)
    message(STATUS "  Triton .inc:      ${TRITON_BUILD_DIR} (Mode A)")
  else()
    message(STATUS "  Triton .inc:      mlir-tblgen (Mode B)")
  endif()
endif()
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "=============================================")
message(STATUS "")
