cmake_minimum_required(VERSION 3.20)
project(AscendPerfModel LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#===----------------------------------------------------------------------===//
# MLIR/LLVM Configuration
#===----------------------------------------------------------------------===//

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
# 从 Triton-Ascend 源码直接编译 Triton Dialect 为静态库。
# 避免链接 libtriton.so 的复杂依赖问题。
#
# 配置:
#   -DTRITON_SRC_DIR=<triton-ascend 源码根目录>
#   -DTRITON_BUILD_DIR=<triton cmake 构建目录> (可选，加速：复用已生成的.inc)
#
# 禁用: -DTRITONSIM_ENABLE_TRITON=OFF
#===----------------------------------------------------------------------===//

option(TRITONSIM_ENABLE_TRITON "Enable Triton dialect support" ON)
option(TRITONSIM_ENABLE_BISHENGIR_HIVM
  "Enable typed BiShengIR HIVM dialect ingestion when local triton-ascend artifacts are available"
  ON)

# Default to submodule path if not explicitly set
set(_default_triton_src "")
if(EXISTS "${PROJECT_SOURCE_DIR}/thirdparty/triton-ascend/CMakeLists.txt")
  set(_default_triton_src "${PROJECT_SOURCE_DIR}/thirdparty/triton-ascend")
endif()
set(TRITON_SRC_DIR "${_default_triton_src}" CACHE PATH "Path to triton-ascend source root")
set(TRITON_BUILD_DIR "" CACHE PATH
    "Path to triton CMake build dir (optional, reuses generated .inc files)")

set(TRITONSIM_HAS_TRITON OFF)
set(TRITON_DIALECT_LIB "")

if(TRITONSIM_ENABLE_TRITON AND TRITON_SRC_DIR)
  # 查找 Triton 源码头文件
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
      "Triton support DISABLED.")
  else()
    message(STATUS "Found Triton headers: ${TRITON_INCLUDE_DIR}")
    
    # 从源码编译 Triton Dialect (thirdparty/triton-dialect/CMakeLists.txt)
    add_subdirectory(thirdparty/triton-dialect)
    
    set(TRITONSIM_HAS_TRITON ON)
    set(TRITON_DIALECT_LIB TritonDialectIR)
    
    # TritonDialectIR 的 target_include_directories 已经设置了
    # 这里只需把 include 传播到全局，让其他源码也能 #include triton 头文件
    include_directories(${TRITON_INCLUDE_DIR})
    if(TRITON_GEN_INC_DIR)
      include_directories(${TRITON_GEN_INC_DIR})
    endif()
    
    add_definitions(-DTRITONSIM_HAS_TRITON=1)
    message(STATUS "=== Triton Integration ENABLED ===")
  endif()
elseif(TRITONSIM_ENABLE_TRITON)
  message(STATUS
    "Triton support requested but TRITON_SRC_DIR not set.\n"
    "Building WITHOUT Triton. Tool will only parse ascend dialect.\n"
    "To enable: -DTRITON_SRC_DIR=<path>")
endif()

message(STATUS "TRITONSIM_HAS_TRITON: ${TRITONSIM_HAS_TRITON}")

#===----------------------------------------------------------------------===//
# Optional BiShengIR HIVM integration
#===----------------------------------------------------------------------===//

set(TRITONSIM_HAS_BISHENGIR_HIVM OFF)
set(TRITONSIM_BISHENGIR_INCLUDE_DIRS "")
set(TRITONSIM_BISHENGIR_HIVM_LIBS "")

set(_default_bishengir_src "")
if(EXISTS "/mnt/d/work/git/triton-ascend/third_party/ascend/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVM.h")
  set(_default_bishengir_src
      "/mnt/d/work/git/triton-ascend/third_party/ascend/AscendNPU-IR/bishengir")
endif()

set(_default_bishengir_build "")
if(EXISTS "/mnt/d/work/git/triton-ascend/python/build/cmake.linux-x86_64-cpython-3.12/third_party/ascend/AscendNPU-IR/bishengir/lib/Dialect/HIVM/IR/libBiShengIRHIVMDialect.a")
  set(_default_bishengir_build
      "/mnt/d/work/git/triton-ascend/python/build/cmake.linux-x86_64-cpython-3.12/third_party/ascend/AscendNPU-IR/bishengir")
endif()

set(TRITONSIM_BISHENGIR_SRC_DIR "${_default_bishengir_src}" CACHE PATH
    "Path to BiShengIR source root (contains include/bishengir/...)")
set(TRITONSIM_BISHENGIR_BUILD_DIR "${_default_bishengir_build}" CACHE PATH
    "Path to BiShengIR build root (contains generated include/bishengir/... and lib/...)")

if(TRITONSIM_ENABLE_BISHENGIR_HIVM)
  set(_bishengir_src_inc "${TRITONSIM_BISHENGIR_SRC_DIR}/include")
  set(_bishengir_build_inc "${TRITONSIM_BISHENGIR_BUILD_DIR}/include")
  set(_bishengir_hivm_lib
      "${TRITONSIM_BISHENGIR_BUILD_DIR}/lib/Dialect/HIVM/IR/libBiShengIRHIVMDialect.a")

  if(EXISTS "${_bishengir_src_inc}/bishengir/Dialect/HIVM/IR/HIVM.h" AND
     EXISTS "${_bishengir_build_inc}/bishengir/Dialect/HIVM/IR/HIVMOps.h.inc" AND
     EXISTS "${_bishengir_hivm_lib}")
    file(GLOB_RECURSE _bishengir_all_libs
      "${TRITONSIM_BISHENGIR_BUILD_DIR}/lib/libBiShengIR*.a"
      "${TRITONSIM_BISHENGIR_BUILD_DIR}/lib/**/*.a")
    list(SORT _bishengir_all_libs)
    set(_bishengir_preferred_libs
      "${TRITONSIM_BISHENGIR_BUILD_DIR}/lib/Dialect/HIVM/IR/libBiShengIRHIVMDialect.a"
      "${TRITONSIM_BISHENGIR_BUILD_DIR}/lib/Dialect/HIVM/Utils/libBiShengIRHIVMUtils.a"
      "${TRITONSIM_BISHENGIR_BUILD_DIR}/lib/Dialect/HACC/IR/libBiShengIRHACCDialect.a"
      "${TRITONSIM_BISHENGIR_BUILD_DIR}/lib/Dialect/HACC/Utils/libBiShengIRHACCUtils.a"
      "${TRITONSIM_BISHENGIR_BUILD_DIR}/lib/Dialect/Annotation/IR/libBiShengIRAnnotationDialect.a"
      "${TRITONSIM_BISHENGIR_BUILD_DIR}/lib/Dialect/HFusion/IR/libBiShengIRHFusionDialect.a"
      "${TRITONSIM_BISHENGIR_BUILD_DIR}/lib/Dialect/MemRef/IR/libBiShengIRMemRefDialect.a"
      "${TRITONSIM_BISHENGIR_BUILD_DIR}/lib/Dialect/MemRefExt/IR/libBiShengIRMemRefExtDialect.a"
      "${TRITONSIM_BISHENGIR_BUILD_DIR}/lib/Dialect/Tensor/IR/libBiShengIRTensorDialect.a"
      "${TRITONSIM_BISHENGIR_BUILD_DIR}/lib/Dialect/Scope/IR/libBiShengIRScopeDialect.a"
      "${TRITONSIM_BISHENGIR_BUILD_DIR}/lib/Dialect/Symbol/IR/libBiShengIRSymbolDialect.a"
      "${TRITONSIM_BISHENGIR_BUILD_DIR}/lib/Dialect/Math/IR/libBiShengIRMathExtDialect.a"
      "${TRITONSIM_BISHENGIR_BUILD_DIR}/lib/Dialect/Utils/libBiShengIRDialectUtils.a"
      "${TRITONSIM_BISHENGIR_BUILD_DIR}/lib/Dialect/HACC/Transforms/libBiShengIRHACCTransforms.a")
    set(TRITONSIM_BISHENGIR_HIVM_LIBS "")
    foreach(_lib IN LISTS _bishengir_preferred_libs)
      if(EXISTS "${_lib}")
        list(APPEND TRITONSIM_BISHENGIR_HIVM_LIBS "${_lib}")
        list(REMOVE_ITEM _bishengir_all_libs "${_lib}")
      endif()
    endforeach()
    list(APPEND TRITONSIM_BISHENGIR_HIVM_LIBS ${_bishengir_all_libs})
    list(APPEND TRITONSIM_BISHENGIR_HIVM_LIBS ${TRITONSIM_BISHENGIR_HIVM_LIBS})
    set(TRITONSIM_BISHENGIR_INCLUDE_DIRS
      "${_bishengir_src_inc}"
      "${_bishengir_build_inc}")
    set(TRITONSIM_HAS_BISHENGIR_HIVM ON)
    include_directories(${TRITONSIM_BISHENGIR_INCLUDE_DIRS})
    add_definitions(-DTRITONSIM_HAS_BISHENGIR_HIVM=1)
    message(STATUS "=== BiShengIR HIVM integration ENABLED ===")
    message(STATUS "BiShengIR source: ${TRITONSIM_BISHENGIR_SRC_DIR}")
    message(STATUS "BiShengIR build:  ${TRITONSIM_BISHENGIR_BUILD_DIR}")
  else()
    message(STATUS
      "BiShengIR HIVM support requested but local headers/libs were not found.\n"
      "Looked for:\n"
      "  ${_bishengir_src_inc}/bishengir/Dialect/HIVM/IR/HIVM.h\n"
      "  ${_bishengir_build_inc}/bishengir/Dialect/HIVM/IR/HIVMOps.h.inc\n"
      "  ${_bishengir_hivm_lib}\n"
      "Continuing with generic/unregistered HIVM parsing.")
  endif()
endif()

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
message(STATUS "  Triton support:   ${TRITONSIM_HAS_TRITON}")
message(STATUS "  BiShengIR HIVM:   ${TRITONSIM_HAS_BISHENGIR_HIVM}")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "=============================================")
message(STATUS "")
