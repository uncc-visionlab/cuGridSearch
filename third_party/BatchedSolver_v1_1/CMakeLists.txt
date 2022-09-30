cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(batched_solver LANGUAGES CUDA)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.5")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.5/bin/nvcc")
#set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_DEFAULT_CUDA_ARCHITECTURES 50 52 60 61 75 86)

set( CMAKE_VERBOSE_MAKEFILE ON)
set( CMAKE_BUILD_TYPE Debug)

find_package(CUDA)

set(CUDA_NVCC_FLAGS -Xcompiler -no-pie --ptxas-options=-v -G -g -O3)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_DEFAULT_CUDA_ARCHITECTURES})
endif()

include_directories(
        ../../../include
        ${CUDA_INCLUDE_DIRS}
        ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
        ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include/
)

add_executable(example_batch_solver
        example.c solve.cu inverse.cu)
#target_compile_options(example_batch_solver PUBLIC --extended-lambda --expt-relaxed-constexpr)
target_link_libraries(example_batch_solver ${CUDA_LIBRARIES})