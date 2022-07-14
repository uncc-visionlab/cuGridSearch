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
//#include <cstdlib>
#include <iostream>

#include "cudaTensor.cuh"

#include "cudaGridSearch.cuh"
#include "cudaErrorFunctions.cuh"
#include "cudaErrorFunctionsStreams.cuh"

#define grid_dimension 2        // the dimension of the grid, e.g., 1 => 1D grid, 2 => 2D grid, 3=> 3D grid, etc.
#define CHANNELS 1              // the number of channels in the image data
typedef float grid_precision;   // the type of values in the grid, e.g., float, double, int, etc.
typedef float func_precision;   // the type of values taken by the error function, e.g., float, double, int, etc.
typedef double pixel_precision; // the type of values in the image, e.g., float, double, int, etc.

typedef func_byvalue_t<func_precision, grid_precision, grid_dimension,
        CudaImage<pixel_precision, CHANNELS>, CudaImage<pixel_precision, CHANNELS> > image_err_func_byvalue;

// create device function pointer for by-value kernel function here
//__device__ image_err_func_byvalue dev_func_byvalue_ptr = averageAbsoluteDifference<func_precision, grid_precision,
//        grid_dimension, CHANNELS, pixel_precision>;
__device__ image_err_func_byvalue dev_func_byvalue_ptr = averageAbsoluteDifference_stream<func_precision, grid_precision,
        grid_dimension, CHANNELS, pixel_precision>;
//__device__ image_err_func_byvalue dev_func_byvalue_ptr = sumOfAbsoluteDifferences<func_precision, grid_precision,
//        grid_dimension, CHANNELS, pixel_precision>;

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

template
class CudaTensor<func_precision, grid_dimension>;

int main(int argc, char **argv) {
    image_err_func_byvalue host_func_byval_ptr;

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
    m2.setValuesFromVector(std::vector<pixel_precision>(imageA_data, imageA_data + 6 * 6));

    m1.display("m1");
    m2.display("m2");

    std::vector<grid_precision> start_point = {(grid_precision) -m2.width() / 2, (grid_precision) -m2.height() / 2};
    std::vector<grid_precision> end_point = {(grid_precision) std::abs(m1.width() - (m2.width() / 2)),
                                             (grid_precision) std::abs(m1.height() - (m2.height() / 2))};
    std::vector<grid_precision> num_samples = {(grid_precision) 1000, (grid_precision) 1000};

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

    // Copy device function pointer for the function having by-value parameters to host side
    cudaMemcpyFromSymbol(&host_func_byval_ptr, dev_func_byvalue_ptr,
                         sizeof(image_err_func_byvalue));

    //translation_xy_gridsearcher.search(host_func_byval_ptr, m1, m2);
//    translation_xy_gridsearcher.search_by_value(host_func_byval_ptr, m1, m2);
    translation_xy_gridsearcher.search_by_value_stream(host_func_byval_ptr, 5000, 10, m1, m2);

//    func_values.display("grid values",num_samples[0]);
//    func_values.display("grid values");

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
    checkCudaErrors(cudaFree(m1.data()));
    checkCudaErrors(cudaFree(m2.data()));
    checkCudaErrors(cudaFree(translation_xy_grid.data()));
    checkCudaErrors(cudaFree(func_values.data()));

    return EXIT_SUCCESS;
}
