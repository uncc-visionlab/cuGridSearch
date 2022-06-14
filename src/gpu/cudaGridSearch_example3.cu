//#define CUDAFUNCTION
#define CUDAFUNCTION __host__ __device__

#include <cmath>
//#include <cstdlib>
#include <iostream>

#include "cudaTensor.cuh"
#include "cudaGridSearch.cuh"
#include "cudaErrorFunctions.cuh"

#define grid_dimension 8        // the dimension of the grid, e.g., 1 => 1D grid, 2 => 2D grid, 3=> 3D grid, etc.
typedef float grid_precision;   // the type of values in the grid, e.g., float, double, int, etc.
typedef float func_precision;   // the type of values taken by the error function, e.g., float, double, int, etc.
typedef double pixel_precision; // the type of values in the image, e.g., float, double, int, etc.

typedef func_byvalue_t<func_precision, grid_precision, grid_dimension, CudaImage<pixel_precision>, CudaImage<pixel_precision> > image_err_func_byvalue;

// create device function pointer for by-value kernel function here
//__device__ image_err_func_byvalue dev_func_byvalue_ptr = averageAbsoluteDifference<func_precision, grid_precision, grid_dimension, pixel_precision>;
//__device__ image_err_func_byvalue dev_func_byvalue_ptr = sumOfAbsoluteDifferences<func_precision, grid_precision, grid_dimension, pixel_precision>;
__device__ image_err_func_byvalue dev_func_byvalue_ptr = averageAbsoluteDifferenceH<func_precision, grid_precision, grid_dimension, pixel_precision>;

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
    image_err_func_byvalue host_func_byval_ptr;

    int cuda_device = 0;
    cudaDeviceProp deviceProp;
    cuda_device = findCudaDevice(0, nullptr);
    checkCudaErrors(cudaGetDevice(&cuda_device));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

    CudaImage<pixel_precision> m1(6, 6);
    CudaImage<pixel_precision> m2(6, 6);

    ck(cudaMalloc(&m1.data(), m1.bytesSize()));
    ck(cudaMalloc(&m2.data(), m2.bytesSize()));

    m1.setValuesFromVector(std::vector<pixel_precision>(imageA_data, imageA_data + 6 * 6));
    m2.setValuesFromVector(std::vector<pixel_precision>(imageA_data, imageA_data + 6 * 6));

    m1.display("m1");
    m2.display("m2");

    std::vector<grid_precision> start_point = {
            0.5, 0.5, (grid_precision) -m2.width() / 2,
            0.5, 0.5, (grid_precision) -m2.height() / 2,
            0, 0
    };
    std::vector<grid_precision> end_point = {
            1, 1, (grid_precision) std::abs( m1.width() - (m2.width() / 2)),
            1, 1, (grid_precision) std::abs( m1.height() - (m2.height() / 2)),
            .5, .5
    };
    std::vector<grid_precision> resolution = {
            .25, .25, (grid_precision) 0.5f,
            .25, .25, (grid_precision) 0.5f,
            .25, .25
    };

    CudaGrid<grid_precision> perspective_transform_grid(grid_dimension);
    ck(cudaMalloc(&perspective_transform_grid.data(), perspective_transform_grid.bytesSize()));

    perspective_transform_grid.setStartPoint(start_point);
    perspective_transform_grid.setEndPoint(end_point);
    perspective_transform_grid.setResolution(resolution);
    perspective_transform_grid.display("perspective_xy_grid");


    grid_precision axis_sample_counts[grid_dimension];
    perspective_transform_grid.getAxisSampleCounts(axis_sample_counts);

    CudaTensor<func_precision, grid_dimension> func_values(axis_sample_counts);
    ck(cudaMalloc(&func_values._data, func_values.bytesSize()));
    func_values.fill(0);

    // first template argument is the error function return type
    // second template argument is the grid point value type
    CudaGridSearcher<func_precision, grid_precision, grid_dimension> perspective_transform_gridsearcher(perspective_transform_grid, func_values);

    // Copy device function pointer for the function having by-value parameters to host side
    cudaMemcpyFromSymbol(&host_func_byval_ptr, dev_func_byvalue_ptr,
                         sizeof(image_err_func_byvalue));

    //perspective_transform_gridsearcher.search(host_func_byval_ptr, m1, m2);
    perspective_transform_gridsearcher.search_by_value(host_func_byval_ptr, m1, m2);

    func_values.display();

    // Clean memory
    ck(cudaFree(m1.data()));
    ck(cudaFree(m2.data()));
    ck(cudaFree(perspective_transform_grid.data()));
    ck(cudaFree(func_values.data()));

    return EXIT_SUCCESS;
}