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

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Dense>

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

#include <cmath>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <chrono>

/* nVIDIA CUDA header */

#include "third_party/cxxopts.hpp"

#include "cuGridSearch.cuh"

#include "../gpu/cudaErrorFunction_mi.cuh"
#include "../gpu/cudaErrorFunctions.cuh"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#define STBI_NO_FAILURE_STRINGS

#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include "stb_image_resize.h"

#define PI 3.14159265

#define DEBUG true

#define grid_dimension 8        // the dimension of the grid, e.g., 1 => 1D grid, 2 => 2D grid, 3=> 3D grid, etc.
#define CHANNELS 1
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
__device__ image_err_func_byvalue dev_func_byvalue_ptr = calcMIAlt<func_precision, grid_precision,
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
            ("m,mod", "Intensity modifier", cxxopts::value<float>()->default_value("0"))
            ("gt", "Ground Truth CSV file", cxxopts::value<std::string>())
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

template<typename pixType, uint8_t D, uint8_t CHANNEL>
__global__ void fuseAlignedImages_local(nv_ext::Vec<float, D> estimatedH,
                                  nv_ext::Vec<float, D> trueH,
                                  CudaImage<pixType, CHANNEL> imageReference,
                                  CudaImage<pixType, CHANNEL> imageMoving,
                                  CudaImage<pixType, 3> imageFused,
                                  int fusedChannel = 0) {
    int columns = imageFused.width();
    int rows = imageFused.height();
    float identityH_vals[] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    nv_ext::Vec<float, D> identityH(identityH_vals);
    nv_ext::Vec<float, D> H;
    // Transform the image
    for (int c = 0; c < 3; c++) {
        switch (c) {
            case 0:
                H = estimatedH;
                break;
            case 1: //GREEN
                H = identityH;
                break;
            case 2:
                H = trueH;
                break;
        }
        float &h11 = H[0], &h12 = H[1], &h13 = H[2], &h21 = H[3], &h22 = H[4], &h23 = H[5], &h31 = H[6], &h32 = H[7];
        // for (int x = 0; x < columns; x++) {
        //     for (int y = 0; y < rows; y++) {
        for (float x = -0.5; x < 0.5; x += 1/((float)columns)) {
            for (float y = -0.5; y < 0.5; y += 1/((float)rows)) {
                //float denom = (h11*h22 - h12*h21 - h11*h23*h32 + h12*h23*h31 + h13*h21*h32 - h13*h22*h31);
                float new_w = ((h21 * h32 - h22 * h31) * x + (h12 * h31 - h11 * h32) * y + h11 * h22 - h12 * h21);
                float new_x = ((h22 - h23 * h32) * x + (h13 * h32 - h12) * y + h12 * h23 - h13 * h22) / new_w;
                float new_y = ((h23 * h31 - h21) * x + (h11 - h13 * h31) * y + h13 * h21 - h11 * h23) / new_w;
                
                new_x = (new_x + 0.5) * imageMoving.width();
                new_y = (new_y + 0.5) * imageMoving.height();

                int temp_x = round(columns * (x+0.5));
                int temp_y = round(rows * (y+0.5));

                // printf("new_x = %f, new_y = %f, temp_x = %d, temp_y = %d\n", new_x, new_y, temp_x, temp_y);
                switch (c) {
                    case 1:
                        if (imageReference.inImage(new_y, new_x)) {
                            // imageFused.template at<float>(y, x, c) = imageReference.valueAt_bilinear(new_y, new_x,
                            //                                                                          fusedChannel);
                            imageFused.template at<float>(temp_y, temp_x, c) = imageReference.valueAt_bilinear(new_y, new_x,
                                                                                                     fusedChannel);
                        }
                        break;
                    case 0:
                    case 2:
                        if (imageMoving.inImage(new_y, new_x)) {
                            // imageFused.template at<float>(y, x, c) = imageMoving.valueAt_bilinear(new_y, new_x,
                            //                                                                       fusedChannel);
                            imageFused.template at<float>(temp_y, temp_x, c) = imageMoving.valueAt_bilinear(new_y, new_x,
                                                                                                  fusedChannel);
                        }
                        break;
                }
            }
        }
    }
}

template<typename pixType, uint8_t D, uint8_t CHANNEL>
__global__ void transformImage_local(nv_ext::Vec<float, D> H, int rowsm, int colsm,
                               CudaImage<pixType, CHANNEL> imageIn,
                               CudaImage<pixType, CHANNEL> imageOut) {
    // int colsm = imageOut.width();
    // int rowsm = imageOut.height();

    // Transform the image
    float &h11 = H[0], &h12 = H[1], &h13 = H[2], &h21 = H[3], &h22 = H[4], &h23 = H[5], &h31 = H[6], &h32 = H[7];
    for (int x = 0; x < colsm; x++) {
        for (int y = 0; y < rowsm; y++) {
            //float denom = (h11*h22 - h12*h21 - h11*h23*h32 + h12*h23*h31 + h13*h21*h32 - h13*h22*h31);
            float new_w = ((h21 * h32 - h22 * h31) * x + (h12 * h31 - h11 * h32) * y + h11 * h22 - h12 * h21);
            float new_x = ((h22 - h23 * h32) * x + (h13 * h32 - h12) * y + h12 * h23 - h13 * h22) / new_w;
            float new_y = ((h23 * h31 - h21) * x + (h11 - h13 * h31) * y + h13 * h21 - h11 * h23) / new_w;
            if (imageIn.inImage(new_y, new_x)) {
                for (int c = 0; c < CHANNEL; c++) {
                    imageOut.template at<float>(y, x, c) = imageIn.valueAt_bilinear(new_y, new_x, c);
                }
            }
        }
    }
}

template<typename pixType, uint8_t CHANNEL>
void
writeTransformedImageToDisk_local(CudaImage<pixType, CHANNEL> image, int rowsf, int colsf, nv_ext::Vec<float, 8> H,
                            std::string img_out_filename) {
    CudaImage<uint8_t, CHANNEL> image_out(rowsf, colsf);
    checkCudaErrors(cudaMalloc(&image_out.data(), image_out.bytesSize()));
    checkCudaErrors(cudaMemset(image_out.data(), 0, image_out.bytesSize()));
    transformImage_local<uint8_t, 8, CHANNELS><<<1, 1>>>(H, rowsf, colsf, image, image_out);

    uint8_t *hostValues;
    checkCudaErrors(cudaMallocHost(&hostValues, image_out.bytesSize()));
    checkCudaErrors(cudaMemcpy(hostValues, image_out.data(), image_out.bytesSize(), cudaMemcpyDeviceToHost));
    if (endsWithCaseInsensitive(img_out_filename, ".png")) {
        stbi_write_png(img_out_filename.c_str(), image_out.width(), image_out.height(), CHANNELS, hostValues,
                       image.width() * sizeof(pixType) * CHANNELS);
        // You have to use 3 comp for complete jpg file. If not, the image will be grayscale or nothing.
    } else if (endsWithCaseInsensitive(img_out_filename, ".jpg")) {
        stbi_write_jpg(img_out_filename.c_str(), image_out.width(), image_out.height(), CHANNELS, hostValues, 95);
    } else {
        std::cout << "Filename suffix has image format not recognized." << std::endl;
    }
    cudaFreeHost(hostValues);
    checkCudaErrors(cudaFree(image_out.data()));
}

template<typename pixType, uint8_t CHANNEL>
void writeAlignedAndFusedImageToDisk_local(CudaImage<pixType, CHANNEL> image_fix,
                                     CudaImage<pixType, CHANNEL> image_mov,
                                     nv_ext::Vec<float, 8> estimatedH,
                                     nv_ext::Vec<float, 8> trueH,
                                     std::string img_fused_filename) {
    CudaImage<uint8_t, 3> image_fused(image_fix.height(), image_fix.width());
    checkCudaErrors(cudaMalloc(&image_fused.data(), image_fused.bytesSize()));
    checkCudaErrors(cudaMemset(image_fused.data(), 0, image_fused.bytesSize()));
    fuseAlignedImages_local<uint8_t, 8, CHANNELS><<<1, 1>>>(estimatedH, trueH, image_fix, image_mov, image_fused, 0);

    uint8_t *hostValues_fused;
    checkCudaErrors(cudaMallocHost(&hostValues_fused, image_fused.bytesSize()));
    checkCudaErrors(cudaMemcpy(hostValues_fused, image_fused.data(), image_fused.bytesSize(), cudaMemcpyDeviceToHost));
    if (endsWithCaseInsensitive(img_fused_filename, ".png")) {
        stbi_write_png(img_fused_filename.c_str(), image_fused.width(), image_fused.height(), 3, hostValues_fused,
                       image_fix.width() * sizeof(uint8_t) * 3);
        // You have to use 3 comp for complete jpg file. If not, the image will be grayscale or nothing.
    } else if (endsWithCaseInsensitive(img_fused_filename, ".jpg")) {
        stbi_write_jpg(img_fused_filename.c_str(), image_fused.width(), image_fused.height(), 3, hostValues_fused, 95);
    } else {
        std::cout << "Filename suffix has image format not recognized." << std::endl;
    }
    cudaFreeHost(hostValues_fused);
    checkCudaErrors(cudaFree(image_fused.data()));
}

// Generic functor
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{
    typedef _Scalar Scalar;
    enum {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };
    typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

    int m_inputs, m_values;

    Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
    Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

    int inputs() const { return m_inputs; }
    int values() const { return m_values; }

};

CudaImage<uint8_t, CHANNELS> *image_fix_test;
CudaImage<uint8_t, CHANNELS> *image_mov_test;

struct my_functor : Functor<float>
{
    my_functor(void): Functor<float>(grid_dimension,grid_dimension) {}
    int operator()(const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const
    {
        float minParams[grid_dimension] = {0};
        for (int i = 0; i < grid_dimension; i++)
            minParams[i] = x(i);
        nv_ext::Vec<float, grid_dimension> minParamsVec(minParams);

        fvec(0) = calcMIAlt<func_precision, grid_precision, grid_dimension, CHANNELS, pixel_precision>(minParamsVec, *image_fix_test, *image_mov_test);
        for (int i = 1; i < grid_dimension; i++)
            fvec(i) = 0;
        return 0;
    }
};


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

float delta_filter_data[1]={1.0f};

#define F_D5x5 25.0f
float avg_filter_5x5_data[5 * 5] = {1.0f / F_D5x5, 1.0f / F_D5x5, 1.0f / F_D5x5, 1.0f / F_D5x5, 1.0f / F_D5x5,
                                    1.0f / F_D5x5, 1.0f / F_D5x5, 1.0f / F_D5x5, 1.0f / F_D5x5, 1.0f / F_D5x5,
                                    1.0f / F_D5x5, 1.0f / F_D5x5, 1.0f / F_D5x5, 1.0f / F_D5x5, 1.0f / F_D5x5,
                                    1.0f / F_D5x5, 1.0f / F_D5x5, 1.0f / F_D5x5, 1.0f / F_D5x5, 1.0f / F_D5x5,
                                    1.0f / F_D5x5, 1.0f / F_D5x5, 1.0f / F_D5x5, 1.0f / F_D5x5, 1.0f / F_D5x5,
};

float central_diff_5[5] = {
        1.0f / 12.0f, -2.0f / 3.0f, 0.0f , 2.0f / 3.0f, -1.0f / 12.0f
};

#define F_D10x10 100.0f
float avg_filter_10x10_data[10 * 10] = {1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10,
                                        1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10,
                                        1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10,
                                        1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10,
                                        1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10,
                                        1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10,
                                        1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10,
                                        1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10,
                                        1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10,
                                        1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10,
                                        1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10,
                                        1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10,
                                        1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10,
                                        1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10,
                                        1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10,
                                        1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10,
                                        1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10,
                                        1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10,
                                        1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10,
                                        1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10, 1.0f / F_D10x10
};

void display_data(float *a, int ax, int ay) {
    for (int y = 0; y < ay; y++) {
        for (int x = 0; x < ax; x++) {
            std::cout << a[x + y * ax] << ", ";
        }
        std::cout << std::endl;
    }
}

void conv2_data(float *a, int ax, int ay, float *h, int hx, int hy, float *c) {
    int ax0, ay0, hx0, hy0;
    for (ax0 = 0; ax0 < ax; ax0++) {
        for (ay0 = 0; ay0 < ay; ay0++) {
            for (hx0 = 0; hx0 < hx; hx0++) {
                for (hy0 = 0; hy0 < hy; hy0++) {
                    if (ax0 - hx0 >= 0 && ax0 - hx0 < ax && ay0 - hy0 >= 0 && ay0 - hy0 < ay) {
                        c[ax0 + ay0 * ax] += h[hx0 + hy0 * hx] * a[(ax0 - hx0) + (ay0 - hy0) * ax];
                    }
                }
            }
        }
    }
}

template<typename pixType>
__global__ void transformVerify(CudaImage<pixType, CHANNELS> img, float mod) {
    int xIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int yIdx = blockDim.y * blockIdx.y + threadIdx.y;

    if(xIdx > img.width() || yIdx > img.height()) return;

    for(int c = 0; c < CHANNELS; c++) {
        // char pixel = img.at(yIdx, xIdx, c);
        char pixel = img._data[img.toIndex(yIdx, xIdx, c)];
        int temp = (int)pixel;
        temp /= mod;
        temp *= mod;
        if(xIdx == 60 && yIdx == 50) {
            printf("Pixel Value = %d\nChanged Value = %d\n", pixel, temp);
        }
    }
}

template<typename pixType>
__global__ void transformPixel(CudaImage<pixType, CHANNELS> img, float mod) {
    int xIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int yIdx = blockDim.y * blockIdx.y + threadIdx.y;

    if(xIdx > img.width() || yIdx > img.height()) return;


    // Grouping
    for(int c = 0; c < CHANNELS; c++) {
        // char pixel = img.at(yIdx, xIdx, c);
        char pixel = img._data[img.toIndex(yIdx, xIdx, c)];
        int temp = (int)pixel;
        temp /= mod;
        temp *= mod;
        img._data[img.toIndex(yIdx, xIdx, c)] = (pixType)temp;
    }

    // // Sigmoid func
    // for(int c = 0; c < CHANNELS; c++) {
    //     char pixel = img.at(yIdx, xIdx, c);
    //     float temp = (float)pixel;
    //     temp /= 255.0f;
    //     temp *= 12.0f;
    //     temp = 1.0f / (1.0f + exp(-1 * mod * (temp - 6.0f)));
    //     temp *= 255.0f;
    //     img.template at<pixType>(yIdx, xIdx, c) = temp;
    // }

    // // Arcsine
    // for(int c = 0; c < CHANNELS; c++) {
    //     char pixel = img.at(yIdx, xIdx, c);
    //     float temp = (float)pixel;
    //     temp /= 255.0f;
    //     temp *= 2.0f;
    //     temp = (asin(mod*(temp - 1.0f)) + PI/2) / PI;
    //     temp *= 255.0f;
    //     img.template at<pixType>(yIdx, xIdx, c) = temp;
    // }
}

int main(int argc, char **argv) {

    // Argument parsing
    cxxopts::Options options("cuda_gridsearch", "UNC Charlotte Machine Vision Lab CUDA-accelerated grid search code.");
    cxxopts_integration(options);
    auto result = options.parse(argc, argv);
    std::string gt_filename, img_fixed_filename, img_moved_filename, img_out_filename, img_fused_filename;
    if (result.count("gt")) {
        gt_filename = result["gt"].as<std::string>();
    }
    if (result.count("i_ref")) {
        img_fixed_filename = result["i_ref"].as<std::string>();
    } else {
        std::cerr << "No input reference image filename was provided. Exiting.." << std::endl;
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }
    if (result.count("i_mov")) {
        img_moved_filename = result["i_mov"].as<std::string>();
    } else {
        std::cerr << "No input moving image filename was provided. Exiting.." << std::endl;
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }
    img_out_filename = result["output"].as<std::string>();
    img_fused_filename = result["fusedoutput"].as<std::string>();
    float intensityModifier = result["mod"].as<float>();

    std::cerr << "Output image filename is " << img_out_filename << "." << std::endl;
    std::cerr << "Fused Output image filename is " << img_fused_filename << "." << std::endl;
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    std::ofstream outfile;
    outfile.open("imageMatcherResults.txt", std::ios_base::app);
    outfile << "input files," << img_fixed_filename.c_str() << "," << img_moved_filename.c_str() << ",";
    outfile << "output fused image," << img_fused_filename << ",";
    outfile << "DEPTH," << DEPTH << ",";

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

    float MAX_SIZE_DISCREPANCY = 1.5;
    int x_new, y_new;
    uint8_t *data_new;
    float scale_factor_x = 1;
    float scale_factor_y = 1;
    /**/
    if (xf * yf > MAX_SIZE_DISCREPANCY * xm * ym) { // resize fixed image
        x_new = xm;
        y_new = ym;
        scale_factor_x = (float)xm / (float)xf;
        scale_factor_y = (float)ym / (float)yf;

        std::cerr << "Rescaling " + img_fixed_filename + " from " << "(" << xf << "," << yf << ")" << " to "
                  << "(" << x_new << "," << y_new << ")" << std::endl;
        data_new = (uint8_t *) malloc(x_new * y_new * CHANNELS);
        stbir_resize_uint8(dataf, xf, yf, 0, data_new, x_new, y_new, 0, CHANNELS);
        if (data_new == NULL) {
            std::cerr << "Image resize " + img_fixed_filename + " failed!" << std::endl;
            return EXIT_FAILURE;
        }
        xf = x_new;
        yf = y_new;
        stbi_image_free(dataf);
        dataf = data_new;
        stbi_write_png("image_fixed_resized.png", x_new, y_new, CHANNELS, data_new,
                       x_new * sizeof(uint8_t) * CHANNELS);
    } else if (xm * ym > MAX_SIZE_DISCREPANCY * xf * yf) {
        x_new = xf;
        y_new = yf;
        scale_factor_x = (float)xf / (float)xm;
        scale_factor_y = (float)yf / (float)ym;
        std::cerr << "Rescaling " + img_moved_filename + " from " << "(" << xm << "," << ym << ")" << " to "
                  << "(" << x_new << "," << y_new << ")" << std::endl;
        data_new = (uint8_t *) malloc(x_new * y_new * CHANNELS);
        stbir_resize_uint8(datam, xm, ym, 0, data_new, x_new, y_new, 0, CHANNELS);
        if (data_new == NULL) {
            std::cerr << "Image resize " + img_moved_filename + " failed!" << std::endl;
            return EXIT_FAILURE;
        }
        xm = x_new;
        ym = y_new;
        stbi_image_free(datam);
        datam = data_new;
        stbi_write_png("image_moved_resized.png", x_new, y_new, CHANNELS, data_new,
                       x_new * sizeof(uint8_t) * CHANNELS);
    }
    /**/
    // number of components must be equal on construction
    // assert(nf == CHANNELS && nm == CHANNELS); // Does not work if using gray scale, nf/nm are based on original channels
    outfile << "Scale Factor," << scale_factor_x << "," << scale_factor_y << ",";
    CudaImage<uint8_t, CHANNELS> image_fix(yf, xf);
    CudaImage<uint8_t, CHANNELS> image_mov(ym, xm);

    checkCudaErrors(cudaMalloc(&image_fix.data(), image_fix.bytesSize()));
    checkCudaErrors(cudaMemcpy(image_fix.data(), dataf, image_fix.bytesSize(), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&image_mov.data(), image_mov.bytesSize()));
    checkCudaErrors(cudaMemcpy(image_mov.data(), datam, image_mov.bytesSize(), cudaMemcpyHostToDevice));

    CudaImage<float> delta_filter(1, 1);
    checkCudaErrors(cudaMalloc(&delta_filter.data(), delta_filter.bytesSize()));
    delta_filter.setValuesFromVector(std::vector<float>(delta_filter_data, delta_filter_data + 1 * 1));

    CudaImage<float> avg_filter_5x5(5, 5);
    checkCudaErrors(cudaMalloc(&avg_filter_5x5.data(), avg_filter_5x5.bytesSize()));
    avg_filter_5x5.setValuesFromVector(std::vector<float>(avg_filter_5x5_data, avg_filter_5x5_data + 5 * 5));
    // avg_filter_5x5.display("avg_filter_5x5");

    CudaImage<float> avg_filter_10x10(10, 10);
    checkCudaErrors(cudaMalloc(&avg_filter_10x10.data(), avg_filter_10x10.bytesSize()));
    avg_filter_10x10.setValuesFromVector(std::vector<float>(avg_filter_10x10_data, avg_filter_10x10_data + 10 * 10));

    float zeros_5x5[5*5] =  { 0 };
    float sobel_5x5[5*5] =  { 0 };
    // set element [2,0] to 1
    zeros_5x5[2*5+0] = 1.0f;
    display_data(zeros_5x5, 5, 5);
    conv2_data(zeros_5x5, 5, 5, central_diff_5, 5, 1, sobel_5x5);
    display_data(sobel_5x5, 5, 5);
    zeros_5x5[2*5+0] = 0.0f;
    zeros_5x5[0*5+2] = 1.0f;
    display_data(zeros_5x5, 5, 5);
    conv2_data(zeros_5x5, 5, 5, central_diff_5, 1, 5, sobel_5x5);
    zeros_5x5[0*5+2] = 0.0f;
    display_data(sobel_5x5, 5, 5);
    for (int idx=0; idx < 5; idx++) {
        sobel_5x5[idx + idx * 5] += central_diff_5[idx];
    }
    display_data(sobel_5x5, 5, 5);
    CudaImage<float> sobel_filter_5x5(10, 10);
    checkCudaErrors(cudaMalloc(&sobel_filter_5x5.data(), sobel_filter_5x5.bytesSize()));
    sobel_filter_5x5.setValuesFromVector(std::vector<float>(sobel_5x5, sobel_5x5 + 5 * 5));

    CudaImage<uint8_t, CHANNELS> image_fix_filtered(yf, xf);
    checkCudaErrors(cudaMalloc(&image_fix_filtered.data(), image_fix_filtered.bytesSize()));
    CHANNEL_ACTION actions[CHANNELS] {FILTER};

    // image_fix.filter(delta_filter, image_fix_filtered, actions);
    // image_fix.filter(avg_filter_5x5, image_fix_filtered, actions);
    image_fix.filter(avg_filter_10x10, image_fix_filtered, actions);
    // image_fix.filter(sobel_filter_5x5, image_fix_filtered, actions);

    if(intensityModifier > 0.0f) {
        dim3 bSize(32,32);
        dim3 gSize(image_mov.width()/32 + 1,image_mov.height()/32 + 1);
        transformPixel<uint8_t><<<gSize,bSize>>>(image_mov, intensityModifier);
        cudaDeviceSynchronize();
    }

    if (DEBUG) {
        // Write an output image to disk
        float identityH_data[] = {1.0f, 0.0f, 0.0f,
                                  0.0f, 1.0f, 0.0f,
                                  0.0f, 0.0f};
        nv_ext::Vec<float, 8> identityH(identityH_data);
        writeTransformedImageToDisk<uint8_t, CHANNELS>(image_fix, yf, xf, identityH, "image_fixed.png");
        writeTransformedImageToDisk<uint8_t, CHANNELS>(image_fix_filtered, yf, xf, identityH,
                                                       "image_fixed_filtered.png");
    }

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
    float MAX_NONOVERLAPPING_PCT = 0.5f;

    std::vector<grid_precision> num_samples = {(grid_precision) 32,
                                               (grid_precision) 16,
                                               (grid_precision) 16,
                                               (grid_precision) 5,
                                               (grid_precision) (xf + 1) / (20 * scale_factor_x),
                                               (grid_precision) (yf + 1) / (20 * scale_factor_y),
                                               (grid_precision) 1,
                                               (grid_precision) 1
    };

    std::vector<grid_precision> start_point = {(grid_precision) 0, // theta
                                               (grid_precision) 1.5 * scale_factor_x, // scaleX
                                               (grid_precision) 1.5 * scale_factor_y, // scaleY
                                               (grid_precision) -0.4,  // shearXY
                                               (grid_precision) -xm * MAX_NONOVERLAPPING_PCT,  // translateX
                                               (grid_precision) -ym * MAX_NONOVERLAPPING_PCT,  // translateY
                                               (grid_precision) 0, // keystoneX
                                               (grid_precision) 0  // keystoneY
    };

    std::vector<grid_precision> end_point = {static_cast<float>((grid_precision) 2 * PI - PI / num_samples[0]),
                                             (grid_precision) 5 * scale_factor_x,
                                             (grid_precision) 5 * scale_factor_y,
                                             (grid_precision) 0.2,
                                             (grid_precision) xf - xm * MAX_NONOVERLAPPING_PCT,
                                             (grid_precision) yf - ym * MAX_NONOVERLAPPING_PCT,
                                             (grid_precision) 0,
                                             (grid_precision) 0
    };

    outfile << "start_point,";
    for (int i = 0; i < grid_dimension; i++) outfile << start_point[i] << ",";
    outfile << "end_point,";
    for (int i = 0; i < grid_dimension; i++) outfile << end_point[i] << ",";
    outfile << "num_samples,";
    for (int i = 0; i < grid_dimension; i++) outfile << num_samples[i] << ",";
    float minParams[grid_dimension] = {0};

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
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
        CudaGridSearcher<func_precision, grid_precision, grid_dimension> translation_xy_gridsearcher(
                translation_xy_grid,
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
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    printf("Total Time Taken: %f\n", elapsed_seconds.count());

    outfile << "Total Time," << elapsed_seconds.count() << ",";

    outfile << "MinParams,";
    for (int i = 0; i < grid_dimension; i++) outfile << minParams[i] << ",";

    // Covariance calculation
    grid_precision *covar_matrix = new grid_precision[(grid_dimension+1)*(grid_dimension+1)];
    num_samples = {(grid_precision) 3,
                    (grid_precision) 3,
                    (grid_precision) 3,
                    (grid_precision) 3,
                    (grid_precision) 3,
                    (grid_precision) 3,
                    (grid_precision) 1,
                    (grid_precision) 1
    };

    start_point = {static_cast<float>((grid_precision) minParams[0] - 3 * PI / 32), // theta
                                               (grid_precision) (minParams[1] - 1.0f), // scaleX
                                               (grid_precision) (minParams[2] - 1.0f), // scaleY
                                               (grid_precision) (minParams[3] - 0.05),  // shearXY
                                               (grid_precision) (minParams[4] - 5.0f),  // translateX
                                               (grid_precision) (minParams[5] - 5.0f),  // translateY
                                               (grid_precision) 0, // keystoneX
                                               (grid_precision) 0  // keystoneY
    };

    end_point = {static_cast<float>((grid_precision) minParams[0] + 3 * PI / 32),
                                             (grid_precision) (minParams[1] + 1.0f),
                                             (grid_precision) (minParams[2] + 1.0f),
                                             (grid_precision) (minParams[3] + 0.05),
                                             (grid_precision) (minParams[4] + 5.0f),
                                             (grid_precision) (minParams[4] + 5.0f),
                                             (grid_precision) 0,
                                             (grid_precision) 0
    };

    {
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
        CudaGridSearcher<func_precision, grid_precision, grid_dimension> translation_xy_gridsearcher(
                translation_xy_grid,
                func_values);

        // Mutual Information
        // Copy device function pointer for the function having by-value parameters to host side
        cudaMemcpyFromSymbol(&host_func_byval_ptr, dev_func_byvalue_ptr,
                             sizeof(image_err_func_byvalue));

        //translation_xy_gridsearcher.search(host_func_byval_ptr, m1, m2);
        translation_xy_gridsearcher.search_by_value(host_func_byval_ptr, image_mov, image_fix);
        // translation_xy_gridsearcher.search_by_value_stream(host_func_byval_ptr, 10000, 1, image_mov, image_fix);
        // translation_xy_gridsearcher.search_by_value_stream(host_func_byval_ptr, 10000, image_fix.height(), image_mov, image_fix);

        // func_values.display();

        
        for(int i = 0; i < (grid_dimension+1)*(grid_dimension+1); i++) {
            covar_matrix[i] = 0;
        }
        // grid_precision covar_matrix[(grid_dimension+1)*(grid_dimension+1)] = {0};
        translation_xy_gridsearcher.covariance_by_value(covar_matrix);

        printf("Covariance matrix:\n");
        for(int i = 0; i < grid_dimension+1; i++) {
            printf("[");
            for(int d = 0; d < grid_dimension+1; d++) {
                printf("%e, ", covar_matrix[(i * (grid_dimension+1)) + d]);
            }
            printf("]\n");
        }
        
        // func_precision min_value;
        // int32_t min_value_index1d;
        // func_values.find_extrema(min_value, min_value_index1d);

        // grid_precision min_grid_point[grid_dimension];
        // translation_xy_grid.getGridPoint(min_grid_point, min_value_index1d);
        // std::cout << "Minimum found at point p = { ";
        // for (int d = 0; d < grid_dimension; d++) {
        //     std::cout << min_grid_point[d] << ((d < grid_dimension - 1) ? ", " : " ");
        //     minParams[d] = min_grid_point[d];

        //     start_point[d] = min_grid_point[d] - (end_point[d] - start_point[d]) / 4;
        //     end_point[d] = min_grid_point[d] + (end_point[d] - start_point[d]) / 4;
        // }
        // std::cout << "}" << std::endl;

        checkCudaErrors(cudaFree(translation_xy_grid.data()));
        checkCudaErrors(cudaFree(func_values.data()));
    }

    // Non-linear optimizer
    Eigen::VectorXf x(grid_dimension);
    for(int i = 0; i < grid_dimension; i++)
        x(i) = minParams[i];
    std::cout << "x: " << x << std::endl;

    my_functor functor;
    Eigen::NumericalDiff<my_functor> numDiff(functor);
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<my_functor>,float> lm(numDiff);
    lm.parameters.maxfev = 2000;
    lm.parameters.xtol = 1.0e-10;

    // TODO: Uncomment this later
    // int ret = lm.minimize(x);
    // std::cout << "Iterations: " << lm.iter << ", Return code: " << ret << std::endl;

    std::cout << "x that minimizes the function: " << x << std::endl;

    // Convert min Values to H
    for(int i = 0; i < grid_dimension; i++)
        minParams[i] = x(i);

    float h11 = 0, h12 = 0, h13 = 0, h21 = 0, h22 = 0, h23 = 0, h31 = 0, h32 = 0;
    float cx = (float) xm / 2, cy = (float) ym / 2;
    nv_ext::Vec<float, grid_dimension> minParamsVec(minParams);
    //std::cout << "Min Value check: " << calcMIAlt<func_precision, grid_precision, grid_dimension, CHANNELS, pixel_precision>(minParamsVec, *image_fix_test, *image_mov_test) << std::endl;
    parametersToHomographyNorm<grid_precision, grid_dimension>(minParamsVec, 
                                                           xm, ym, xf, yf,
                                                           h11, h12, h13,
                                                           h21, h22, h23,
                                                           h31, h32);
    // centerHomography<grid_precision, grid_dimension>(cx, cy,
    //     h11, h12, h13,
    //     h21, h22, h23,
    //     h31, h32);
    float minH[] = {h11, h12, h13,
                    h21, h22, h23,
                    h31, h32};
    outfile << "Homography,";
    std::cout << "Answer[";
    for (int i = 0; i < grid_dimension; i++) {
        outfile << minH[i] << ",";
        std::cout << minH[i] << ((i < grid_dimension - 1) ? "," : "");
    }
    std::cout << "]" << std::endl;

    std::cout << "Scale_factor[" << scale_factor_x << "," << scale_factor_y << "]" << std::endl;

    std::cout << "Homography_to_parameters[";
    // uncenterHomography<grid_precision, grid_dimension>(cx, cy,
    //     h11, h12, h13,
    //     h21, h22, h23,
    //     h31, h32);
    homographyToParameters<grid_precision, grid_dimension>(h11, h12, h13,
                           h21, h22, h23,
                           h31, h32,
                           minParamsVec);
    // centerHomography<grid_precision, grid_dimension>(cx, cy,
    //     h11, h12, h13,
    //     h21, h22, h23,
    //     h31, h32);
    
    for (int i = 0; i < grid_dimension; i++) {
        std::cout << minParamsVec[i] << ((i < grid_dimension - 1) ? "," : "");
    }
    std::cout << "]" << std::endl;

    std::cout << "covar_matrix[";
    for (int i = 0; i < (grid_dimension + 1) * (grid_dimension + 1); i++) {
        std::cout << covar_matrix[i] << ((i < (((grid_dimension + 1) * (grid_dimension + 1)) - 1)) ? "," : "");
    }
    std::cout << "]" << std::endl;
    float tempGTH[] = {h11, h12, h13, h21, h22, h23, h31, h32};
    std::ifstream gtFile(gt_filename.c_str());
    // Check if gtFile exists
    if(gtFile.is_open()) {
        // Extract image number from filename (should be the final part of the name)
        char imgNumChar[5] = {' '};
        size_t numIdx = img_fixed_filename.find_last_not_of("0123456789.png");
        numIdx += 1;
        int imgNumCharIdx = 0;
        while(img_fixed_filename[numIdx] != '.')
            imgNumChar[imgNumCharIdx++] = img_fixed_filename[numIdx++];
        int imgNum = atoi(imgNumChar);
        
        // Get the line in the csv that has the matching image number
        std::string line;
        while(std::getline(gtFile, line)) {
            size_t comma_idx = line.find_first_of(",");
            std::string sTemp = line.substr(0,comma_idx);
            int tempImgNum = atoi(sTemp.c_str());

            if(tempImgNum == imgNum) {
                // Found line, extract GT values (Homography 8 param)
                // printf("Found image number match: %s\n", line.c_str());
                comma_idx += 1;
                for(int i = 0; i < 8; i++) {
                    size_t comma_idx2 = line.find(",",comma_idx);
                    sTemp = line.substr(comma_idx, comma_idx2);
                    tempGTH[i] = std::stof(line.substr(comma_idx, comma_idx2));
                    comma_idx = ++comma_idx2;
                }
                break;
            }
        }
        for(int i = 0; i < 3; i++) tempGTH[i] *= scale_factor_x;
        for(int i = 3; i < 6; i++) tempGTH[i] *= scale_factor_y;
    }
    outfile << "GT Homography,";
    for (int i = 0; i < grid_dimension; i++) outfile << tempGTH[i] << ",";
    outfile << "%% Error,";
    for (int i = 0; i < grid_dimension; i++) outfile <<(minH[i] - tempGTH[i]) / tempGTH[i] << ",";
    outfile << "\n";
//    nv_ext::Vec<float, 8> H(initialH);
    nv_ext::Vec<float, 8> H(minH);
    nv_ext::Vec<float, 8> gtH(tempGTH);
    // Write an output image to disk
    writeTransformedImageToDisk_local<uint8_t, CHANNELS>(image_mov, yf, xf, H, img_out_filename);

    // Write aligned and fused output image to disk
    writeAlignedAndFusedImageToDisk_local<uint8_t, CHANNELS>(image_fix, image_mov, H, gtH, img_fused_filename);

    // Clean memory
    checkCudaErrors(cudaFree(image_fix.data()));
    checkCudaErrors(cudaFree(image_mov.data()));


    stbi_image_free(dataf);
    stbi_image_free(datam);
    delete[] covar_matrix;
    return EXIT_SUCCESS;
}