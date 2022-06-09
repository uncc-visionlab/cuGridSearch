cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cudaGridSearch LANGUAGES CXX CUDA)

set( CMAKE_VERBOSE_MAKEFILE ON)
set( CMAKE_BUILD_TYPE Debug)

find_package(CUDA)

#set(CUDA_NVCC_FLAGS -Xcompiler -fPIC -use_fast_math -gencode=arch=compute_20,code="sm_20,compute_20" --ptxas-options=-v -DMATLAB_MEX_FILE -G -g -O0)
set(CUDA_NVCC_FLAGS -Xcompiler -no-pie --ptxas-options=-v -DMATLAB_MEX_FILE -G -g -O0)

include_directories(
    ../../../include
    ${CUDA_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
    ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include/
)

add_executable(${PROJECT_NAME}_main
        cudaGridSearch_main.cu
        cudaImageOps.cu cudaErrorFunctions.cuh)
target_compile_options(${PROJECT_NAME}_main PUBLIC --extended-lambda)
target_link_libraries(${PROJECT_NAME}_main -no-pie ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})

add_executable(cudaGridSearch_example1
        cudaGridSearch_example1.cu
        cudaImageOps.cu cudaErrorFunctions.cuh)
target_compile_options(cudaGridSearch_example1 PUBLIC --extended-lambda)
target_link_libraries(cudaGridSearch_example1 ${CUDA_LIBRARIES})

add_executable(cudaGridSearch_example2
        cudaGridSearch_example2.cu
        cudaImageOps.cu cudaErrorFunctions.cuh)
target_compile_options(cudaGridSearch_example2 PUBLIC --extended-lambda)
target_link_libraries(cudaGridSearch_example2 ${CUDA_LIBRARIES})

if (Matlab_FOUND)
    include_directories(
        ${Matlab_INCLUDE_DIRS}
        ${Matlab_ROOT_DIR}/toolbox/parallel/gpu/extern/include
        ${Matlab_ROOT_DIR}/sys/cuda/glnxa64/cuda/include
    )

    add_library(${PROJECT_NAME}_mex SHARED cudaGridSearch_mex.cu ${MATLAB_PROJECT_DIR}/Matlabdef.def)
    target_link_libraries(${PROJECT_NAME}_mex ${MATLAB_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
    set_target_properties(${PROJECT_NAME}_mex PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)

    # 32-bit or 64-bit mex
    if(WIN32)
        if (CMAKE_CL_64)
          set_target_properties(${PROJECT_NAME}_mex PROPERTIES SUFFIX .mexw64)
        else(CMAKE_CL_64)
          set_target_properties(${PROJECT_NAME}_mex PROPERTIES SUFFIX .mexw32)
        endif(CMAKE_CL_64)
    else(WIN32)
        if (CMAKE_SIZEOF_VOID_P MATCHES "8")
          set_target_properties(${PROJECT_NAME}_mex PROPERTIES SUFFIX .mexa64 PREFIX "")
        else(CMAKE_SIZEOF_VOID_P MATCHES "8")
          set_target_properties(${PROJECT_NAME}_mex PROPERTIES SUFFIX .mexglx PREFIX "")
        endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
    endif(WIN32)
endif (Matlab_FOUND)
