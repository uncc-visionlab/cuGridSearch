/*
 * Copyright (C) 2022 Andrew R. Willis
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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

#include "cudaTensor.cuh"

#include "cudaGridSearch.cuh"
#include "cudaErrorFunctions.cuh"
#include "cudaErrorFunction_mi.cuh"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#define STBI_NO_FAILURE_STRINGS

#include "stb_image.h"

#define PI 3.14159265

#define grid_dimension 4        // the dimension of the grid, e.g., 1 => 1D grid, 2 => 2D grid, 3=> 3D grid, etc.
typedef float grid_precision;   // the type of values in the grid, e.g., float, double, int, etc.
typedef float func_precision;   // the type of values taken by the error function, e.g., float, double, int, etc.
typedef unsigned char pixel_precision; // the type of values in the image, e.g., float, double, int, etc.

// typedef func_byvalue_t<func_precision, grid_precision, grid_dimension, CudaImage<pixel_precision>, CudaImage<pixel_precision> > image_err_func_byvalue;

// create device function pointer for by-value kernel function here
// __device__ image_err_func_byvalue dev_func_byvalue_ptr = averageAbsoluteDifference<func_precision, grid_precision, grid_dimension, pixel_precision>;
//__device__ image_err_func_byvalue dev_func_byvalue_ptr = sumOfAbsoluteDifferences<func_precision, grid_precision, grid_dimension, pixel_precision>;

// grid_mi
typedef func_byvalue_t<func_precision, grid_precision, grid_dimension,
        unsigned char *, unsigned char *, int, int, int, int> image_err_func_byvalue;
__device__ image_err_func_byvalue dev_func_byvalue_ptr = grid_mi<func_precision, grid_precision,
        grid_dimension, pixel_precision>;

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
            ("i_ref", "Reference Image (image in the reference coordinate frame)", cxxopts::value<std::string>())
            ("i_mov", "Moved Image (image in the measured coordinate frame)", cxxopts::value<std::string>())
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
    cxxopts::Options options("cuda_gridsearch", "UNC Charlotte Machine Vision Lab CUDA-accelerated grid search code.");
    cxxopts_integration(options);

    auto result = options.parse(argc, argv);

    std::string img_fixed_filename, img_moved_filename;
    if (result.count("i_ref")) {
        img_fixed_filename = result["i_ref"].as<std::string>();
    } else {
        std::cerr << "No input reference image was provided. Exiting.." << std::endl;
        return EXIT_FAILURE;
    }
    if (result.count("i_mov")) {
        img_moved_filename = result["i_mov"].as<std::string>();
    } else {
        std::cerr << "No input moving image was provided. Exiting.." << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    /* set GPU grid & block configuration */
    image_err_func_byvalue host_func_byval_ptr;
    int cuda_device = 0;
    cudaDeviceProp deviceProp;

    cuda_device = findCudaDevice(0, nullptr);
    checkCudaErrors(cudaGetDevice(&cuda_device));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

    // Original

    // CudaImage<pixel_precision> m1(6, 6);
    // CudaImage<pixel_precision> m2(6, 6);

    // ck(cudaMalloc(&m1.data(), m1.bytesSize()));
    // ck(cudaMalloc(&m2.data(), m2.bytesSize()));

    // m1.setValuesFromVector(std::vector<pixel_precision>(imageA_data, imageA_data + 6 * 6));
    // m2.setValuesFromVector(std::vector<pixel_precision>(imageA_data, imageA_data + 6 * 6));

    // m1.display("m1");
    // m2.display("m2");

    // std::vector<grid_precision> start_point = {(grid_precision) -m2.width() / 2, (grid_precision) -m2.height() / 2};
    // std::vector<grid_precision> end_point = {(grid_precision) std::abs(m1.width() - (m2.width() / 2)),
    //                                          (grid_precision) std::abs(m1.height() - (m2.height() / 2))};
    // std::vector<grid_precision> num_samples = {(grid_precision) 10000, (grid_precision) 10000};

    // Mutual Information

    if (argc < 3) {
        printf("Args: FixedImage MovingImage\n");
        return 1;
    }

    int xf, yf, nf;

    unsigned char *dataf = stbi_load(img_fixed_filename.c_str(), &xf, &yf, &nf, 1);
    if (dataf == NULL) {
        std::cerr << "Reference image " + img_fixed_filename + " failed to load!" << std::endl;
        return EXIT_FAILURE;
    }

    int xm, ym, nm;

    unsigned char *datam = stbi_load(img_moved_filename.c_str(), &xm, &ym, &nm, 1);
    if (datam == NULL) {
        std::cerr << "Moving image " + img_moved_filename + " failed to load!" << std::endl;
        return EXIT_FAILURE;
    }

    unsigned char *imagef;
    checkCudaErrors(cudaMalloc(&imagef, xf * yf * sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(imagef, dataf, xf * yf * sizeof(unsigned char), cudaMemcpyHostToDevice));

    unsigned char *imagem;
    checkCudaErrors(cudaMalloc(&imagem, xm * ym * sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(imagem, datam, xm * ym * sizeof(unsigned char), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1 << 30));

    // Example MI
    std::vector<grid_precision> start_point = {(grid_precision) -xm / 2, (grid_precision) -ym / 2, (grid_precision) 0,
                                               (grid_precision) 1};
    std::vector<grid_precision> end_point = {(grid_precision) xf - xm / 2, (grid_precision) yf - ym / 2,
                                             (grid_precision) 0, (grid_precision) 1};
    std::vector<grid_precision> num_samples = {(grid_precision) (xf + 1), (grid_precision) (yf + 1), (grid_precision) 1,
                                               (grid_precision) 1};

    CudaGrid<grid_precision, grid_dimension> translation_xy_grid;
    ck(cudaMalloc(&translation_xy_grid.data(), translation_xy_grid.bytesSize()));

    translation_xy_grid.setStartPoint(start_point);
    translation_xy_grid.setEndPoint(end_point);
    translation_xy_grid.setNumSamples(num_samples);
    translation_xy_grid.display("translation_xy_grid");

    grid_precision axis_sample_counts[grid_dimension];
    translation_xy_grid.getAxisSampleCounts(axis_sample_counts);

    CudaTensor<func_precision, grid_dimension> func_values(axis_sample_counts);
    ck(cudaMalloc(&func_values.data(), func_values.bytesSize()));
    //func_values.fill(0);

    // first template argument is the error function return type
    // second template argument is the grid point value type
    CudaGridSearcher<func_precision, grid_precision, grid_dimension> translation_xy_gridsearcher(translation_xy_grid,
                                                                                                 func_values);

    // Copy device function pointer for the function having by-value parameters to host side
    cudaMemcpyFromSymbol(&host_func_byval_ptr, dev_func_byvalue_ptr,
                         sizeof(image_err_func_byvalue));

    //translation_xy_gridsearcher.search(host_func_byval_ptr, m1, m2);
    // translation_xy_gridsearcher.search_by_value(host_func_byval_ptr, m1, m2);
    translation_xy_gridsearcher.search_by_value_stream(host_func_byval_ptr, 10000, 1, imagem, imagef, xm, ym, xf, yf);

//    func_values.display();

    func_precision min_value;
    int32_t min_value_index1d;
    func_values.find_extrema(min_value, min_value_index1d);

    grid_precision min_grid_point[grid_dimension];
    translation_xy_grid.getGridPoint(min_grid_point, min_value_index1d);
    std::cout << "Minimum found at point p = { ";
    for (int d = 0; d < grid_dimension; d++) {
        std::cout << min_grid_point[d] << ((d < grid_dimension - 1) ? ", " : " ");
    }
    std::cout << "}" << std::endl;

    // Clean memory
    // ck(cudaFree(m1.data()));
    // ck(cudaFree(m2.data()));
    ck(cudaFree(imagef));
    ck(cudaFree(imagem));
    ck(cudaFree(translation_xy_grid.data()));
    ck(cudaFree(func_values.data()));
    stbi_image_free(dataf);
    stbi_image_free(datam);
    return EXIT_SUCCESS;
}
