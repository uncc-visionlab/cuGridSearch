/************************************************************************
 Sample CUDA MEX code written by Fang Liu (leoliuf@gmail.com).
 ************************************************************************/

/* system header */
#include <cmath>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

/* nVIDIA CUDA header */
#include <cuda.h>

#include <cxxopts.hpp>

//#define CUDAFUNCTION
#define CUDAFUNCTION __host__ __device__

#include "helper_functions.h"
#include "helper_cuda.h"
#include "cudaImage.cuh"
#include "cudaGridSearch.cuh"
#include "cudaErrorFunctions.cuh"

#define grid_dimension 2        // the dimension of the grid, e.g., 1 => 1D grid, 2 => 2D grid, 3=> 3D grid, etc.
typedef float grid_precision;   // the type of values in the grid, e.g., float, double, int, etc.
typedef float func_precision;   // the type of values taken by the error function, e.g., float, double, int, etc.
typedef float pixel_precision; // the type of values in the image, e.g., float, double, int, etc.

typedef func_byvalue_t<func_precision, grid_precision, grid_dimension, CudaMatrix<pixel_precision>, CudaMatrix<pixel_precision> > image_err_func_byvalue;

// create device function pointer for by-value kernel function here
__device__ image_err_func_byvalue dev_func_byvalue_ptr = averageAbsoluteDifference<func_precision, grid_precision, grid_dimension, pixel_precision>;

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

void cxxopts_integration(cxxopts::Options &options) {

    options.add_options()
            ("i,input", "Input file", cxxopts::value<std::string>())
            //("f,format", "Data format {GOTCHA, Sandia, <auto>}", cxxopts::value<std::string>()->default_value("auto"))
            //("p,polarity", "Polarity {HH,HV,VH,VV,<any>}", cxxopts::value<std::string>()->default_value("any"))
            ("d,debug", "Enable debugging", cxxopts::value<bool>()->default_value("false"))
            ("r,dynrange", "Dynamic Range (dB) <70 dB>", cxxopts::value<float>()->default_value("70"))
            ("o,output", "Output file <sar_image.bmp>", cxxopts::value<std::string>()->default_value("sar_image.bmp"))
            ("h,help", "Print usage");
}

void printMatrix(double **matrix, int ROWS, int COLUMNS) {

    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLUMNS; c++) {
            std::cout << matrix[r][c] << " ";
        }
        std::cout << std::endl;
    }
}

// test grid search
// classes typically store images in column major format so the images
// stored are the transpose of that shown in initialization below
pixel_precision imageA_data[6 * 6] = {0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0,
                                      0, 0, 1, 1, 0, 0,
                                      0, 0, 1, 1, 0, 0,
                                      0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0};

pixel_precision imageB_data[6 * 6] = {0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 1, 1,
                                      0, 0, 0, 0, 1, 1};

pixel_precision imageC_data[6 * 6] = {1, 1, 0, 0, 0, 0,
                                      1, 1, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0
};

int main(int argc, char **argv) {
    cxxopts::Options options("cuda_gridsearch", "UNC Charlotte Machine Vision Lab CUDA-accelerated gridsearch code.");
    cxxopts_integration(options);

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    /* set GPU grid & block configuration */
    image_err_func_byvalue host_func_byval_ptr;
    int cuda_device = 0;
    cudaDeviceProp deviceProp;

    cuda_device = findCudaDevice(0, nullptr);
    checkCudaErrors(cudaGetDevice(&cuda_device));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));
//    cudaCheckErrors("copyImageXYToDevice::cudaMalloc image.values() failed.");
//    cudaCheckErrors("copyImageXYToDevice::cudaMemcpy image.values() failed.");

    CudaMatrix<pixel_precision> m1(6, 6);
    CudaMatrix<pixel_precision> m2(6, 6);

    ck(cudaMalloc(&m1._data, m1.bytesSize()));
    ck(cudaMalloc(&m2._data, m2.bytesSize()));

    // Test here

    m1.setValuesFromVector(std::vector<pixel_precision>(imageA_data, imageA_data + 6 * 6));
    m2.setValuesFromVector(std::vector<pixel_precision>(imageA_data, imageA_data + 6 * 6));
//    m1.fill(5);
//    m2.fill(10);

    m1.display("m1");
    m2.display("m2");

    //m1 *= m2;
    //m1.display("m1 * m2");

    std::vector<grid_precision> start_point = {(grid_precision) -m2._width / 2, (grid_precision) -m2._height / 2};
    std::vector<grid_precision> end_point = {(grid_precision) std::abs(m1._width - (m2._width / 2)),
                                             (grid_precision) std::abs(m1._height - (m2._height / 2))};
    std::vector<grid_precision> resolution = {(grid_precision) 1.0f, (grid_precision) 1.0f};

    CudaGrid<grid_precision> translation_xy_grid(grid_dimension);
    ck(cudaMalloc(&translation_xy_grid.data(), translation_xy_grid.bytesSize()));

    translation_xy_grid.setStartPoint(start_point);
    translation_xy_grid.setEndPoint(end_point);
    translation_xy_grid.setResolution(resolution);
    translation_xy_grid.display("translation_xy_grid");

    // first template argument is the error function return type
    // second template argument is the grid point value type
    CudaGridSearcher<func_precision, grid_precision> translation_xy_gridsearcher(translation_xy_grid);

    // Copy device function pointer for the function having by-value parameters to host side
    cudaMemcpyFromSymbol(&host_func_byval_ptr, dev_func_byvalue_ptr,
                         sizeof(image_err_func_byvalue));

    //translation_xy_gridsearcher.search(host_func_byval_ptr, m1, m2);
    translation_xy_gridsearcher.search_by_value(host_func_byval_ptr, m1, m2);

    // Clean memory
    ck(cudaFree(m1._data));
    ck(cudaFree(m2._data));
    ck(cudaFree(translation_xy_grid.data()));

    return EXIT_SUCCESS;
}
