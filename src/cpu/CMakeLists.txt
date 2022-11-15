cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cpuRegistration LANGUAGES CXX CUDA)

set( CMAKE_VERBOSE_MAKEFILE ON)
set( CMAKE_BUILD_TYPE Debug)

#set(CUDA_NVCC_FLAGS -Xcompiler -fPIC -use_fast_math -gencode=arch=compute_20,code="sm_20,compute_20" --ptxas-options=-v -DMATLAB_MEX_FILE -G -g -O0)
#set(CUDA_NVCC_FLAGS -Xcompiler -no-pie --ptxas-options=-v -DMATLAB_MEX_FILE -G -g -O0)
# suppress multiple "set but never used" variable warnings due to using NVCC to compile code using stb_image.h

add_executable(cpuNonLinearOptimizer
        cpuNonLinearOptimizer.cpp)
#target_compile_options(cpuNonLinearOptimizer PUBLIC --extended-lambda --expt-relaxed-constexpr)
target_link_libraries(cpuNonLinearOptimizer -no-pie Eigen3::Eigen)

include_directories(
        ../../../include
        ${CUDA_INCLUDE_DIRS}
        ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
        ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include/
)

set(CMAKE_DEFAULT_CUDA_ARCHITECTURES 50 52 60 61 75 86)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_DEFAULT_CUDA_ARCHITECTURES})
endif()

add_executable(multispectral_ImageMatcher
        multispectral_ImageMatcher.cu)
#target_compile_options(cpuNonLinearOptimizer PUBLIC --extended-lambda --expt-relaxed-constexpr)
target_link_libraries(multispectral_ImageMatcher -no-pie Eigen3::Eigen cugridsearch)
target_compile_options(multispectral_ImageMatcher PRIVATE -Xcudafe --diag_suppress=20012)