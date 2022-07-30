cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

find_package(Matlab)
find_package(CUDA)

if (MATLAB_FOUND)
    add_definitions(/DMATLAB_MEX_FILE) #define matlab macros
    add_definitions(/DMX_COMPAT_64)
    add_definitions(/DMATLAB_DEFAULT_RELEASE=R2019b)
    add_definitions(/DUSE_MEX_CMD)
endif (MATLAB_FOUND)

add_subdirectory(cpu)

if (CUDA_FOUND)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_DEFAULT_CUDA_ARCHITECTURES})
    endif()

    add_subdirectory(gpu)
endif (CUDA_FOUND)
