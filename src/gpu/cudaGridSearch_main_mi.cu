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


//#include <Eigen/Dense>

//#include <unsupported/Eigen/NonLinearOptimization>
//#include <unsupported/Eigen/NumericalDiff>

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

#include <cudaTensor.cuh>

#include "cudaGridSearch.cuh"
#include "cudaErrorFunctions.cuh"
#include "cudaErrorFunction_mi.cuh"
#include "cudaErrorFunctionsStreams.cuh"
#include "cudaErrorFunction_miStreams.cuh"
#include "cudaImageFunctions.cuh"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#define STBI_NO_FAILURE_STRINGS

#include "stb_image.h"

#define PI 3.14159265

#define grid_dimension 8        // the dimension of the grid, e.g., 1 => 1D grid, 2 => 2D grid, 3=> 3D grid, etc.
#define CHANNELS 3
typedef float grid_precision;   // the type of values in the grid, e.g., float, double, int, etc.
typedef float func_precision;   // the type of values taken by the error function, e.g., float, double, int, etc.
typedef uint8_t pixel_precision; // the type of values in the image, e.g., float, double, int, etc.

#define DEPTH 1

// typedef func_byvalue_t<func_precision, grid_precision, grid_dimension,
//         CudaImage<pixel_precision, CHANNELS>, CudaImage<pixel_precision, CHANNELS> > image_err_func_byvalue;

// create device function pointer for by-value kernel function here
// __device__ image_err_func_byvalue dev_func_byvalue_ptr = averageAbsoluteDifference<func_precision, grid_precision, grid_dimension, CHANNELS, pixel_precision>;
//__device__ image_err_func_byvalue dev_func_byvalue_ptr = sumOfAbsoluteDifferences<func_precision, grid_precision, grid_dimension, pixel_precision>;

// SQD/NCC/MI
typedef func_byvalue_t<func_precision, grid_precision, grid_dimension,
       CudaImage<pixel_precision, CHANNELS>, CudaImage<pixel_precision, CHANNELS> > image_err_func_byvalue;
__device__ image_err_func_byvalue dev_func_byvalue_ptr = calcNCCAlt<func_precision, grid_precision,
       grid_dimension, CHANNELS, pixel_precision>;

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
            ("d,debug", "Enable debugging", cxxopts::value<bool>()->default_value("false"))
            ("o,output", "Output file <output_image.png>",
             cxxopts::value<std::string>()->default_value("output_image.png"))
            ("f,fusedoutput", "Fused output file <output_image_fused.png>",
             cxxopts::value<std::string>()->default_value("output_image_fused.png"))
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

// Generic functor
//template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
//struct Functor
//{
//    typedef _Scalar Scalar;
//    enum {
//        InputsAtCompileTime = NX,
//        ValuesAtCompileTime = NY
//    };
//    typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
//    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
//    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;
//
//    int m_inputs, m_values;
//
//    Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
//    Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}
//
//    int inputs() const { return m_inputs; }
//    int values() const { return m_values; }
//
//};

CudaImage<uint8_t, CHANNELS> *image_fix_test;
CudaImage<uint8_t, CHANNELS> *image_mov_test;
//struct my_functor : Functor<float>
//{
//    my_functor(void): Functor<float>(grid_dimension,grid_dimension) {}
//    int operator()(const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const
//    {
//        float minParams[grid_dimension] = {0};
//        for (int i = 0; i < grid_dimension; i++)
//            minParams[i] = x(i);
//        nv_ext::Vec<float, grid_dimension> minParamsVec(minParams);
//
//        fvec(0) = sqrt(calcNCCAlt<func_precision, grid_precision, grid_dimension, CHANNELS, pixel_precision>(minParamsVec, *image_fix_test, *image_mov_test));
//        for (int i = 1; i < grid_dimension; i++)
//            fvec(i) = 0;
//        return 0;
//    }
//};

// test grid search
// classes typically store images in column major format so the images
// stored are the transpose of that shown in initialization below
pixel_precision imageA_data[6 * 6] = {0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0,
                                      0, 0, 255, 255, 0, 0,
                                      0, 0, 255, 255, 0, 0,
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

    // Argument parsing
    cxxopts::Options options("cuda_gridsearch", "UNC Charlotte Machine Vision Lab CUDA-accelerated grid search code.");
    cxxopts_integration(options);
    auto result = options.parse(argc, argv);
    std::string img_fixed_filename, img_moved_filename, img_out_filename, img_fused_filename;
    if (result.count("i_ref")) {
        img_fixed_filename = result["i_ref"].as<std::string>();
    } else {
        std::cerr << "No input reference image filename was provided. Exiting.." << std::endl;
        return EXIT_FAILURE;
    }
    if (result.count("i_mov")) {
        img_moved_filename = result["i_mov"].as<std::string>();
    } else {
        std::cerr << "No input moving image filename was provided. Exiting.." << std::endl;
        return EXIT_FAILURE;
    }
    img_out_filename = result["output"].as<std::string>();
    img_fused_filename = result["fusedoutput"].as<std::string>();
    std::cerr << "Output image filename is " << img_out_filename << "." << std::endl;
    std::cerr << "Fused Output image filename is " << img_fused_filename << "." << std::endl;
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

    // Load input images from disk
    int xf, yf, nf;
    uint8_t *dataf = stbi_load(img_fixed_filename.c_str(), &xf, &yf, &nf, CHANNELS);
    if (dataf == NULL) {
        std::cerr << "Reference image " + img_fixed_filename + " failed to load!" << std::endl;
        return EXIT_FAILURE;
    }
    int xm, ym, nm;
    uint8_t *datam = stbi_load(img_moved_filename.c_str(), &xm, &ym, &nm, CHANNELS);
    if (datam == NULL) {
        std::cerr << "Moving image " + img_moved_filename + " failed to load!" << std::endl;
        return EXIT_FAILURE;
    }

    // number of components must be equal on construction
    // assert(nf == CHANNELS && nm == CHANNELS); // Does not work if using gray scale, nf/nm are based on original channels

    CudaImage<uint8_t, CHANNELS> image_fix(yf, xf);
    CudaImage<uint8_t, CHANNELS> image_mov(ym, xm);

    checkCudaErrors(cudaMalloc(&image_fix.data(), image_fix.bytesSize()));
    checkCudaErrors(cudaMemcpy(image_fix.data(), dataf, image_fix.bytesSize(), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&image_mov.data(), image_mov.bytesSize()));
    checkCudaErrors(cudaMemcpy(image_mov.data(), datam, image_mov.bytesSize(), cudaMemcpyHostToDevice));

    CudaImage<uint8_t, CHANNELS> image_fix2(yf, xf);
    CudaImage<uint8_t, CHANNELS> image_mov2(ym, xm);

    image_fix2.data() = dataf;
    image_mov2.data() = datam;

    image_fix_test = &image_fix2;
    image_mov_test = &image_mov2;

    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1 << 30));

    //    linear interpolation in homography / affine matrix space
    //    https://math.stackexchange.com/questions/612006/decomposing-an-affine-transformation
    //    using the parameterization described by Stephane Laurent
    float theta = 5.0 * PI / 180.0; // range [0, 2*PI]
    float scaleX = 1.5;  // // range [1, 2]
    float scaleY = 1.5;  // // range [1, 2]
    float shearXY = 0.2; // range [-0.2, 0.2]
    float translateX = -10; // range [-image.width()/2, image.width()/2]
    float translateY = -30; // range [-image.height()/2, image.height()/2]
    float keystoneX = 0.0; // range [-0.1, 0.1]
    float keystoneY = 0.0; // range [-0.1, 0.1]
    // Transform does scale, shear, rotate, then translate and finally perspective project per the website
    // https://math.stackexchange.com/questions/612006/decomposing-an-affine-transformation
    //float initialH[] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};

    float initialH[] = {scaleX * cos(theta), scaleY * shearXY * cos(theta) - scaleY * sin(theta), scaleX * translateX,
                        scaleX * sin(theta), scaleY * shearXY * sin(theta) + scaleY * cos(theta), scaleY * translateY,
                        keystoneX, keystoneY};

    // Example MI
    scaleX = 1;
    scaleY = 1;
    std::vector<grid_precision> start_point = {(grid_precision) 0, // theta
                                               (grid_precision) 1.5, // scaleX
                                               (grid_precision) 1.5, // scaleY
                                               (grid_precision) -0.4,  // shearXY
                                               (grid_precision) -xm / 2,  // translateX
                                               (grid_precision) -ym / 2,  // translateY
                                               (grid_precision) 0, // keystoneX
                                               (grid_precision) 0  // keystoneY
                                              };
    std::vector<grid_precision> end_point = {(grid_precision) PI,
                                             (grid_precision) 5,
                                             (grid_precision) 5,
                                             (grid_precision) 0.2,
                                             (grid_precision) xf - xm/2,
                                             (grid_precision) yf - ym/2,
                                             (grid_precision) 0,
                                             (grid_precision) 0
                                            };
                                             
    std::vector<grid_precision> num_samples = {(grid_precision) 8,
                                               (grid_precision) 8,
                                               (grid_precision) 8,
                                               (grid_precision) 5,
                                               (grid_precision) (xf + 1) / 10,
                                               (grid_precision) (yf + 1) / 10,
                                               (grid_precision) 1,
                                               (grid_precision) 1
                                            };
    
    float minParams[grid_dimension] = {0};

    for (int iii = 0; iii < DEPTH; iii++) {
        CudaGrid<grid_precision, grid_dimension> translation_xy_grid;
        checkCudaErrors(cudaMalloc(&translation_xy_grid.data(), translation_xy_grid.bytesSize()));

        translation_xy_grid.setStartPoint(start_point);
        translation_xy_grid.setEndPoint(end_point);
        translation_xy_grid.setNumSamples(num_samples);
        translation_xy_grid.display("translation_xy_grid");

        grid_precision axis_sample_counts[grid_dimension];
        translation_xy_grid.getAxisSampleCounts(axis_sample_counts);

        CudaTensor<func_precision, grid_dimension> func_values(axis_sample_counts);
        checkCudaErrors(cudaMalloc(&func_values.data(), func_values.bytesSize()));
        //func_values.fill(0);

        // first template argument is the error function return type
        // second template argument is the grid point value type
        CudaGridSearcher<func_precision, grid_precision, grid_dimension> translation_xy_gridsearcher(translation_xy_grid,
                                                                                                    func_values);

        // Mutual Information
        // Copy device function pointer for the function having by-value parameters to host side
        cudaMemcpyFromSymbol(&host_func_byval_ptr, dev_func_byvalue_ptr,
                            sizeof(image_err_func_byvalue));

        //translation_xy_gridsearcher.search(host_func_byval_ptr, m1, m2);
        translation_xy_gridsearcher.search_by_value(host_func_byval_ptr, image_mov, image_fix);
        // translation_xy_gridsearcher.search_by_value_stream(host_func_byval_ptr, 10000, 1, image_mov, image_fix);
        // translation_xy_gridsearcher.search_by_value_stream(host_func_byval_ptr, 10000, image_fix.height(), image_mov, image_fix);

    //    func_values.display();

        func_precision min_value;
        int32_t min_value_index1d;
        func_values.find_extrema(min_value, min_value_index1d);

        grid_precision min_grid_point[grid_dimension];
        translation_xy_grid.getGridPoint(min_grid_point, min_value_index1d);
        std::cout << "Minimum found at point p = { ";
        for (int d = 0; d < grid_dimension; d++) {
            std::cout << min_grid_point[d] << ((d < grid_dimension - 1) ? ", " : " ");
            minParams[d] = min_grid_point[d];

            start_point[d] = min_grid_point[d] - (end_point[d] - start_point[d]) / 4;
            end_point[d] = min_grid_point[d] + (end_point[d] - start_point[d]) / 4;
        }
        std::cout << "}" << std::endl;

        checkCudaErrors(cudaFree(translation_xy_grid.data()));
        checkCudaErrors(cudaFree(func_values.data()));
    }

    // Non-linear optimizer
//    Eigen::VectorXf x(grid_dimension);
//    for(int i = 0; i < grid_dimension; i++)
//        x(i) = minParams[i];
    // std::cout << "x: " << x << std::endl;

//    my_functor functor;
//    Eigen::NumericalDiff<my_functor> numDiff(functor);
//    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<my_functor>,float> lm(numDiff);
//    lm.parameters.maxfev = 2000;
//    lm.parameters.xtol = 1.0e-10;

    // int ret = lm.minimize(x);
    // std::cout << "Iterations: " << lm.iter << ", Return code: " << ret << std::endl;

    // std::cout << "x that minimizes the function: " << x << std::endl;

    // Convert min Values to H
    // for(int i = 0; i < grid_dimension; i++)
    //     minParams[i] = x(i);

    float h11 = 0, h12 = 0, h13 = 0, h21 = 0, h22 = 0, h23 = 0, h31 = 0, h32 = 0, cx = (float)xm/2, cy = (float)ym/2;
    nv_ext::Vec<float, grid_dimension> minParamsVec(minParams);
    std::cout << "Min Value check: " << calcNCCAlt<func_precision, grid_precision, grid_dimension, CHANNELS, pixel_precision>(minParamsVec, *image_fix_test, *image_mov_test) << std::endl;
    parametersToHomography<grid_precision,grid_dimension>(minParamsVec, cx, cy,
        h11, h12, h13,
        h21, h22, h23,
        h31, h32);
    float minH[] = {h11, h12, h13,
                    h21, h22, h23,
                    h31, h32};
//    nv_ext::Vec<float, 8> H(initialH);
    nv_ext::Vec<float, 8> H(minH);

    // Write an output image to disk
    writeTransformedImageToDisk<uint8_t, CHANNELS>(image_mov, yf, xf, H, img_out_filename);

    // Write aligned and fused output image to disk
    writeAlignedAndFusedImageToDisk<uint8_t, CHANNELS>(image_fix, image_mov, H, H, img_fused_filename);

    // Clean memory
    checkCudaErrors(cudaFree(image_fix.data()));
    checkCudaErrors(cudaFree(image_mov.data()));
    
    
    stbi_image_free(dataf);
    stbi_image_free(datam);
    return EXIT_SUCCESS;
}
