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

//#define CUDAFUNCTION
#define CUDAFUNCTION __host__ __device__

#include <cmath>
#include <iostream>

#include <cudaTensor.cuh>

#include "cudaGridSearch.cuh"
#include "cudaErrorFunctions.cuh"

#define grid_dimension 2        // the dimension of the grid, e.g., 1 => 1D grid, 2 => 2D grid, 3=> 3D grid, etc.
#define CHANNELS 1              // the number of channels in the image data
typedef float grid_precision;   // the type of values in the grid, e.g., float, double, int, etc.
typedef float func_precision;   // the type of values taken by the error function, e.g., float, double, int, etc.
typedef double pixel_precision; // the type of values in the image, e.g., float, double, int, etc.

typedef func_byreference_t<func_precision, grid_precision, grid_dimension,
        CudaImage<pixel_precision, CHANNELS>, CudaImage<pixel_precision, CHANNELS> > image_err_func_byref;

// create device function pointer for by-reference kernel function here
//__device__ image_err_func_byref dev_func_byref_ptr = averageAbsoluteDifference<func_precision, grid_precision, grid_dimension, pixel_precision>;
__device__ image_err_func_byref dev_func_byref_ptr = sumOfAbsoluteDifferences<func_precision, grid_precision,
        grid_dimension, CHANNELS, pixel_precision>;

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
    image_err_func_byref host_func_byref_ptr;

    int cuda_device = 0;
    cudaDeviceProp deviceProp;
    cuda_device = findCudaDevice(0, nullptr);
    checkCudaErrors(cudaGetDevice(&cuda_device));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

    CudaImage<pixel_precision> m1(6, 6);
    CudaImage<pixel_precision> m2(6, 6);

    checkCudaErrors(cudaMalloc(&m1.data(), m1.bytesSize()));
    checkCudaErrors(cudaMalloc(&m2.data(), m2.bytesSize()));

    m1.setValuesFromVector(std::vector<pixel_precision>(imageA_data, imageA_data + 6 * 6));
    m2.setValuesFromVector(std::vector<pixel_precision>(imageC_data, imageC_data + 6 * 6));

    m1.display("m1");
    m2.display("m2");

    std::vector<grid_precision> start_point = {(grid_precision) -m2.width() / 2, (grid_precision) -m2.height() / 2};
    std::vector<grid_precision> end_point = {(grid_precision) std::abs(m1.width() - (m2.width() / 2)),
                                             (grid_precision) std::abs(m1.height() - (m2.height() / 2))};
    std::vector<grid_precision> num_samples = {(grid_precision) 103, (grid_precision) 111};

    CudaGrid<grid_precision, grid_dimension> translation_xy_grid;
    checkCudaErrors(cudaMalloc(&translation_xy_grid.data(), translation_xy_grid.bytesSize()));

    translation_xy_grid.setStartPoint(start_point);
    translation_xy_grid.setEndPoint(end_point);
    translation_xy_grid.setNumSamples(num_samples);
    translation_xy_grid.display("translation_xy_grid");


    grid_precision axis_sample_counts[grid_dimension];
    translation_xy_grid.getAxisSampleCounts(axis_sample_counts);

    CudaTensor<func_precision, grid_dimension> func_values(axis_sample_counts);
    checkCudaErrors(cudaMalloc(&func_values._data, func_values.bytesSize()));
    func_values.fill(0);

    // first template argument is the error function return type
    // second template argument is the grid point value type
    CudaGridSearcher<func_precision, grid_precision, grid_dimension> translation_xy_gridsearcher(translation_xy_grid,
                                                                                                 func_values);

    CudaImage<pixel_precision> *d_m1, *d_m2;
    checkCudaErrors(cudaMalloc((void **) &d_m1, sizeof(CudaImage<pixel_precision>)));
    checkCudaErrors(cudaMalloc((void **) &d_m2, sizeof(CudaImage<pixel_precision>)));
    cudaMemcpy(d_m1, &m1, sizeof(CudaImage<pixel_precision>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, &m2, sizeof(CudaImage<pixel_precision>), cudaMemcpyHostToDevice);

    // Copy device function pointer for the function having by-reference parameters to host side
    cudaMemcpyFromSymbol(&host_func_byref_ptr, dev_func_byref_ptr,
                         sizeof(image_err_func_byref));

    // translation_xy_gridsearcher.search_by_reference(host_func_byref_ptr, d_m1, d_m2);
    translation_xy_gridsearcher.search_by_reference_stream(host_func_byref_ptr, 10000, 1, d_m1, d_m2);

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

    checkCudaErrors(cudaFree(d_m1));
    checkCudaErrors(cudaFree(d_m2));

    // Clean memory
    checkCudaErrors(cudaFree(m1.data()));
    checkCudaErrors(cudaFree(m2.data()));
    checkCudaErrors(cudaFree(translation_xy_grid.data()));
    checkCudaErrors(cudaFree(func_values.data()));

    return EXIT_SUCCESS;
}