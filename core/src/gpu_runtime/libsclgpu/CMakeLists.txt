cmake_minimum_required(VERSION 3.25.2)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_CUDA_HOST_COMPILER /usr/bin/clang++)
#set(CMAKE_CXX_COMPILER /usr/bin/clang++)
#set(CMAKE_CXX_COMPILER /home/paulbib/Builds/clang/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04/bin/clang++)
#set(CMAKE_CUDA_HOST_COMPILER /home/paulbib/Builds/clang/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04/bin/clang++)

# verbose compilation
#set(CMAKE_VERBOSE_MAKEFILE on)


if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "--expt-relaxed-constexpr -G")
    add_definitions(-DSAFETY=1)
else()
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "--expt-relaxed-constexpr --generate-line-info")
    add_definitions(-DSAFETY=0)
endif()

set(OBJECTS_NAME sclgpu_objs)
set(LIBRARY_NAME sclgpu)
set(TESTS_NAME sclgpu_tests)

project(SclGpu LANGUAGES CXX CUDA)

#message(CMAKE_CXX_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}")
#message(CMAKE_CXX_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}")

# add objects target
add_library(${OBJECTS_NAME} OBJECT 
  alloc.h
  bindings.h
  dataflow.h
  device_vec.h
  expr.h
  filter.h
  find.h
  flame.h
  intersect.h
  join.h
  join_gather/join_gather.h
  merge.h
  normalize.h
  overwrite_one.h
  product.h
  project.h
  provenance.h
  remove_nan.h
  table.h
  table_index.h
  utils.h
  project/project_constant.h
  project/project_interpret.h
  project/project_permute.h

  alloc.cu
  bindings.cu
  dataflow.cu
  expr.cu
  filter.cu
  find.cu
  intersect.cu
  join.cu
  join_gather/join_gather.cu
  merge.cu
  utils.cu
  normalize.cu
  overwrite_one.cu
  product.cu
  project.cu
  provenance.cu
  remove_nan.cu
  table.cu
  table_sorted_until.cu
  table_index.cu
  specialize_normalize.cu
  project/project_constant.cu
  project/project_constant_dtkp.cu
  project/project_interpret.cu
  project/project_permute.cu

  #specializations/normalize1.cu
  #specializations/normalize2.cu
  #specializations/normalize3.cu
  #specializations/normalize4.cu
  #specializations/normalize4s0.cu
  #specializations/normalize4s1.cu
  #specializations/normalize4s2.cu
  #specializations/normalize4s3.cu
  #specializations/normalize4s4.cu
  #specializations/normalize4s5.cu
  #specializations/normalize4s6.cu
  #specializations/normalize4s7.cu
  #specializations/normalize4s8.cu
    generated/merge.cu_spec_0.cu
    generated/merge.cu_spec_1.cu
    generated/merge.cu_spec_2.cu
    generated/merge.cu_spec_3.cu
    generated/merge.cu_spec_4.cu
    generated/merge.cu_spec_5.cu
    generated/merge.cu_spec_6.cu
    generated/merge.cu_spec_7.cu
    generated/merge.cu_spec_8.cu
    generated/merge.cu_spec_9.cu
    generated/merge.cu_spec_10.cu
    generated/merge.cu_spec_11.cu
    generated/merge.cu_spec_12.cu
    generated/merge.cu_spec_13.cu
    generated/merge.cu_spec_14.cu
    generated/merge.cu_spec_15.cu
    generated/merge.cu_spec_16.cu
    generated/merge.cu_spec_17.cu
    generated/merge.cu_spec_18.cu
    generated/merge.cu_spec_19.cu
    generated/merge.cu_spec_20.cu
    generated/merge.cu_spec_21.cu
    generated/merge.cu_spec_22.cu
    generated/merge.cu_spec_23.cu
    generated/merge.cu_spec_24.cu
    generated/merge.cu_spec_25.cu
    generated/merge.cu_spec_26.cu
    generated/merge.cu_spec_27.cu
    generated/merge.cu_spec_28.cu
    generated/merge.cu_spec_29.cu
    generated/merge.cu_spec_30.cu
    generated/merge.cu_spec_31.cu
    generated/merge.cu_spec_32.cu
    generated/merge.cu_spec_33.cu
    generated/merge.cu_spec_34.cu
    generated/merge.cu_spec_35.cu
    generated/merge.cu_spec_36.cu
    generated/merge.cu_spec_37.cu
    generated/merge.cu_spec_38.cu
    generated/merge.cu_spec_39.cu
    generated/normalize.cu_spec_0.cu
    generated/normalize.cu_spec_1.cu
    generated/normalize.cu_spec_2.cu
    generated/normalize.cu_spec_3.cu
    generated/normalize.cu_spec_4.cu
    generated/normalize.cu_spec_5.cu
    generated/normalize.cu_spec_6.cu
    generated/normalize.cu_spec_7.cu
    generated/normalize.cu_spec_8.cu
    generated/normalize.cu_spec_9.cu
    generated/normalize.cu_spec_10.cu
    generated/normalize.cu_spec_11.cu
    generated/normalize.cu_spec_12.cu
    generated/normalize.cu_spec_13.cu
    generated/normalize.cu_spec_14.cu
    generated/normalize.cu_spec_15.cu
    generated/normalize.cu_spec_16.cu
    generated/normalize.cu_spec_17.cu
    generated/normalize.cu_spec_18.cu
    generated/normalize.cu_spec_19.cu
    generated/normalize.cu_spec_20.cu
    generated/normalize.cu_spec_21.cu
    generated/normalize.cu_spec_22.cu
    generated/normalize.cu_spec_23.cu
    generated/normalize.cu_spec_24.cu
    generated/normalize.cu_spec_25.cu
    generated/normalize.cu_spec_26.cu
    generated/normalize.cu_spec_27.cu
    generated/normalize.cu_spec_28.cu
    generated/normalize.cu_spec_29.cu
    generated/normalize.cu_spec_30.cu
    generated/normalize.cu_spec_31.cu
    generated/normalize.cu_spec_32.cu
    generated/normalize.cu_spec_33.cu
    generated/normalize.cu_spec_34.cu
    generated/normalize.cu_spec_35.cu
    generated/normalize.cu_spec_36.cu
    generated/normalize.cu_spec_37.cu
    generated/normalize.cu_spec_38.cu
    generated/normalize.cu_spec_39.cu
    generated/normalize.cu_spec_40.cu
    generated/normalize.cu_spec_41.cu
    generated/normalize.cu_spec_42.cu
    generated/normalize.cu_spec_43.cu
    generated/normalize.cu_spec_44.cu
)

# add archive target
add_library(${LIBRARY_NAME} STATIC
  lib.cu
  $<TARGET_OBJECTS:${OBJECTS_NAME}>
  )
target_compile_options(${LIBRARY_NAME} PRIVATE -Wall -Wextra --generate-line-info)

# add test target
add_executable(
  ${TESTS_NAME}
  test/run_tests.cu
  $<TARGET_OBJECTS:${OBJECTS_NAME}>
)
target_compile_options(${TESTS_NAME} PRIVATE -Wall -Wextra)

# ==================== Link Libraries =====================

# CATCH2
Include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0
)
FetchContent_MakeAvailable(Catch2)

set(USE_CCCL 1)
if (DEFINED USE_CCCL)
  # CCCL
  add_executable(
    cccl_ex
    test/cccl.cu
  )
  add_library(CCCL::CCCL INTERFACE IMPORTED GLOBAL)

  #find_package(CUB REQUIRED CONFIG NO_DEFAULT_PATH HINTS "${CMAKE_CURRENT_LIST_DIR}/cccl/cub/" )
  find_package(CUB REQUIRED CONFIG NO_DEFAULT_PATH HINTS "${CMAKE_CURRENT_LIST_DIR}/cccl/lib/cmake/cub/" )
  add_library(CCCL::CUB ALIAS _CUB_CUB)
  target_link_libraries(CCCL::CCCL INTERFACE CCCL::CUB)

  find_package(libcudacxx CONFIG NO_DEFAULT_PATH HINTS "${CMAKE_CURRENT_LIST_DIR}/cccl/lib/cmake/libcudacxx/")
  add_library(CCCL::libcudacxx ALIAS _libcudacxx_libcudacxx)
  target_link_libraries(CCCL::CCCL INTERFACE CCCL::libcudacxx)

  find_package(Thrust CONFIG NO_DEFAULT_PATH HINTS "${CMAKE_CURRENT_LIST_DIR}/cccl/lib/cmake/thrust/")
  thrust_create_target(CCCL::Thrust FROM_OPTIONS
    HOST_OPTION cpp
    DEVICE_OPTION cuda
  )
  target_link_libraries(CCCL::CCCL INTERFACE CCCL::Thrust)
  
  target_link_libraries(cccl_ex PRIVATE CCCL::CCCL)
  target_link_libraries(${OBJECTS_NAME} CCCL::CCCL)
  target_link_libraries(${LIBRARY_NAME} CCCL::CCCL)
  target_link_libraries(${TESTS_NAME} PRIVATE CCCL::CCCL Catch2::Catch2WithMain )
else()
  # Thrust
  set(Thrust_DIR /usr/local/cuda-12/targets/x86_64-linux/lib/cmake/thrust)
  find_package(Thrust REQUIRED CONFIG)
  thrust_create_target(Thrust)
  target_link_libraries(${OBJECTS_NAME} Thrust)
  target_link_libraries(${LIBRARY_NAME} Thrust)
  target_link_libraries(${TESTS_NAME} PRIVATE Thrust Catch2::Catch2WithMain )
endif()

# ==================== Compile options ====================

# add timing info
set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
set_property(TARGET ${OBJECTS_NAME} PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")

set_target_properties(
  ${LIBRARY_NAME} ${TESTS_NAME} ${OBJECTS_NAME}
    PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES native

      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED YES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES

      CXX_EXTENSIONS NO
)

install (TARGETS ${LIBRARY_NAME} DESTINATION .)

# ==================== Configure tests ====================

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(${TESTS_NAME})
