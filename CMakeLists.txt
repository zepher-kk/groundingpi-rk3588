cmake_minimum_required(VERSION 3.16)
project(groundingpi_rk3588 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -Wall -Wextra -Wno-unused-parameter)

# ---------------------------------------------------------------------------
# Vendor SDKs are NOT redistributed with this repository.
# Point these at your own Rockchip SDK copies:
#   RKLLM_SDK_DIR : contains include/rkllm.h and lib/librkllmrt.so
#   RKNN_SDK_DIR  : contains include/rknn_api.h and lib/librknnrt.so
# ---------------------------------------------------------------------------
set(RKLLM_SDK_DIR "" CACHE PATH "Directory with include/rkllm.h and lib/librkllmrt.so")
set(RKNN_SDK_DIR "" CACHE PATH "Directory with include/rknn_api.h and lib/librknnrt.so")

option(ENABLE_EXPERIMENTS "Build experiment probes and batch tools" OFF)
option(ENABLE_OPENCV_EXPERIMENTS "Build OpenCV-based experiment tools" OFF)
option(ENABLE_GROUNDINGPI "Build the original GroundingPI-4B entry" ON)
option(ENABLE_STUDENT "Build the 3s student entry" ON)

function(gpi_require_sdk name dir header_name)
  if(dir STREQUAL "")
    message(FATAL_ERROR
      "${name} is not set. Pass -D${name}=/path/to/sdk (needs include/${header_name}).")
  endif()
  if(NOT EXISTS "${dir}/include/${header_name}")
    message(FATAL_ERROR "${name}=${dir} does not contain include/${header_name}")
  endif()
endfunction()

gpi_require_sdk(RKLLM_SDK_DIR "${RKLLM_SDK_DIR}" "rkllm.h")
gpi_require_sdk(RKNN_SDK_DIR "${RKNN_SDK_DIR}" "rknn_api.h")

find_library(RKLLM_LIB NAMES rkllmrt PATHS "${RKLLM_SDK_DIR}/lib" NO_DEFAULT_PATH)
find_library(RKNN_LIB NAMES rknnrt PATHS "${RKNN_SDK_DIR}/lib" NO_DEFAULT_PATH)
if(NOT RKLLM_LIB)
  message(FATAL_ERROR "librkllmrt.so not found under ${RKLLM_SDK_DIR}/lib")
endif()
if(NOT RKNN_LIB)
  message(FATAL_ERROR "librknnrt.so not found under ${RKNN_SDK_DIR}/lib")
endif()
message(STATUS "rkllm: ${RKLLM_LIB}")
message(STATUS "rknn : ${RKNN_LIB}")

include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${RKLLM_SDK_DIR}/include
  ${RKNN_SDK_DIR}/include
)

set(GPI_CORE_SRC
  src/common.cc src/npy.cc src/tokenizer.cc src/embed.cc
  src/vision.cc src/llm.cc src/protocol.cc src/preprocess.cc)

# RPATH: let the runtime library be found from the SDK dir at run time, without
# baking a machine path into the binary (override with GPI_RPATH if needed).
set(GPI_RPATH "${RKLLM_SDK_DIR}/lib;${RKNN_SDK_DIR}/lib")
if(DEFINED ENV{GPI_RPATH})
  set(GPI_RPATH "$ENV{GPI_RPATH}")
endif()

# ------------------------------ production apps ----------------------------
if(ENABLE_GROUNDINGPI)
  add_executable(groundingpi apps/baseline/main.cc ${GPI_CORE_SRC})
  target_link_libraries(groundingpi ${RKNN_LIB} ${RKLLM_LIB} pthread)
  set_target_properties(groundingpi PROPERTIES BUILD_RPATH "${GPI_RPATH}")
endif()

if(ENABLE_STUDENT)
  add_executable(student_v1 apps/student/main.cc
    src/common.cc src/tokenizer.cc src/embed.cc src/vision.cc src/llm.cc)
  target_link_libraries(student_v1 ${RKNN_LIB} ${RKLLM_LIB} pthread)
  set_target_properties(student_v1 PROPERTIES BUILD_RPATH "${GPI_RPATH}")
endif()

# ------------------------------ experiments --------------------------------
if(ENABLE_EXPERIMENTS)
  add_executable(protocol_probe experiments/protocol_probe.cc src/common.cc src/npy.cc src/vision.cc src/llm.cc)
  target_link_libraries(protocol_probe ${RKNN_LIB} ${RKLLM_LIB} pthread)
  set_target_properties(protocol_probe PROPERTIES BUILD_RPATH "${GPI_RPATH}")

  add_executable(student_coexist experiments/student_coexist.cc src/common.cc src/npy.cc src/vision.cc src/llm.cc)
  target_link_libraries(student_coexist ${RKNN_LIB} ${RKLLM_LIB} pthread)
  set_target_properties(student_coexist PROPERTIES BUILD_RPATH "${GPI_RPATH}")

  add_executable(coexist_probe experiments/coexist_probe.cc ${GPI_CORE_SRC})
  target_link_libraries(coexist_probe ${RKNN_LIB} ${RKLLM_LIB} pthread)
  set_target_properties(coexist_probe PROPERTIES BUILD_RPATH "${GPI_RPATH}")

  add_executable(pipeline_bench experiments/pipeline_bench.cc ${GPI_CORE_SRC})
  target_link_libraries(pipeline_bench ${RKNN_LIB} ${RKLLM_LIB} pthread)
  set_target_properties(pipeline_bench PROPERTIES BUILD_RPATH "${GPI_RPATH}")

  add_executable(resident_service experiments/resident_service.cc ${GPI_CORE_SRC})
  target_link_libraries(resident_service ${RKNN_LIB} ${RKLLM_LIB} pthread)
  set_target_properties(resident_service PROPERTIES BUILD_RPATH "${GPI_RPATH}")

  add_executable(probe experiments/probe.cc)
  target_link_libraries(probe ${RKNN_LIB} pthread)

  add_executable(student_batch_probe experiments/student_batch_probe.cc)
  target_link_libraries(student_batch_probe ${RKLLM_LIB} pthread)
  set_target_properties(student_batch_probe PROPERTIES BUILD_RPATH "${GPI_RPATH}")
endif()

if(ENABLE_EXPERIMENTS AND ENABLE_OPENCV_EXPERIMENTS)
  find_package(OpenCV REQUIRED)

  add_executable(student_batch_prep experiments/student_batch_prep.cc
    src/common.cc src/tokenizer.cc src/embed.cc src/vision.cc)
  target_link_libraries(student_batch_prep ${RKNN_LIB} ${OpenCV_LIBS} pthread)
  target_include_directories(student_batch_prep PRIVATE ${OpenCV_INCLUDE_DIRS})
  set_target_properties(student_batch_prep PROPERTIES BUILD_RPATH "${GPI_RPATH}")

  add_executable(student_batch_pair experiments/student_batch_pair.cc
    src/common.cc src/tokenizer.cc src/embed.cc src/vision.cc)
  target_link_libraries(student_batch_pair ${RKNN_LIB} ${RKLLM_LIB} ${OpenCV_LIBS} pthread)
  target_include_directories(student_batch_pair PRIVATE ${OpenCV_INCLUDE_DIRS})
  set_target_properties(student_batch_pair PROPERTIES BUILD_RPATH "${GPI_RPATH}")

  add_executable(rga_probe experiments/rga_probe.cc)
  target_link_libraries(rga_probe rga pthread)

  add_executable(rga_probe2 experiments/rga_probe2.cc)
  target_link_libraries(rga_probe2 rga ${RKNN_LIB} pthread)
endif()

install(TARGETS groundingpi student_v1
  RUNTIME DESTINATION bin OPTIONAL)
