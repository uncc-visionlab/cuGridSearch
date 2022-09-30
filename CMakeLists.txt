cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cuGridSearch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
#set (CMAKE_CXX_STANDARD 11)

set( CMAKE_VERBOSE_MAKEFILE ON)

#set( CMAKE_BUILD_TYPE Debug)
#set( CMAKE_BUILD_TYPE Release)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake_modules" CACHE STRING "Modules for CMake" FORCE)

#set( Matlab_ROOT_DIR "/usr/local/bin/matlab/R2020a")
set( Matlab_ROOT_DIR "/usr/local/bin/matlab/R2021a")
set(MATLAB_PROJECT_DIR "${CMAKE_CURRENT_LIST_DIR}")

# Code should build on CUDA version 11.0 - 11.5 not currently compatible with CUDA 11.6
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.5")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.5/bin/nvcc")
#set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_DEFAULT_CUDA_ARCHITECTURES 50 52 60 61 75 86)

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

# Build example executables
option(BUILD_EXAMPLES "Build library examples." ON)
if (BUILD_EXAMPLES)
    message("STATUS cuGridSearch::Build library examples = YES")
else()
    message("STATUS cuGridSearch::Build library examples = NO")
endif()

include_directories(
        include
        include/third_party
)

add_subdirectory(third_party)

add_subdirectory(src)