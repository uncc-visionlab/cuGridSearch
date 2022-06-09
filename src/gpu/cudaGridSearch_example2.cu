//#define CUDAFUNCTION
#define CUDAFUNCTION __host__ __device__

#include <cmath>
#include <iostream>

#include "cudaImage.cuh"
#include "cudaGridSearch.cuh"
#include "cudaErrorFunctions.cuh"

#define grid_dimension 2        // the dimension of the grid, e.g., 1 => 1D grid, 2 => 2D grid, 3=> 3D grid, etc.
typedef int32_t grid_precision;   // the type of values in the grid, e.g., float, double, int, etc.
typedef float func_precision;   // the type of values taken by the error function, e.g., float, double, int, etc.
typedef double pixel_precision; // the type of values in the image, e.g., float, double, int, etc.

typedef func_byreference_t<func_precision, grid_precision, grid_dimension, CudaImage<pixel_precision>, CudaImage<pixel_precision> > image_err_func_byref;

// create device function pointer for by-reference kernel function here
__device__ image_err_func_byref dev_func_byref_ptr = averageAbsoluteDifference<func_precision, grid_precision, grid_dimension, pixel_precision>;

// test grid search
// classes typically store images in column major format so the images
// stored are the transpose of that shown in initialization below
uint8_t imageA_data[6 * 6] = {0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0,
                              0, 0, 1, 1, 0, 0,
                              0, 0, 1, 1, 0, 0,
                              0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0};
uint8_t imageB_data[6 * 6] = {0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 1, 1,
                              0, 0, 0, 0, 1, 1};
uint8_t imageC_data[6 * 6] = {
        1, 1, 0, 0, 0, 0,
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

    ck(cudaMalloc(&m1._data, m1.bytesSize()));
    ck(cudaMalloc(&m2._data, m2.bytesSize()));

    // Test here

    //m1.setValuesFromVector({1, 1, 1, 2, 2, 2, 3, 3, 3});
    m1.fill(5);
    m2.fill(10);

    //m1.display("m1");
    //m2.display("m2");

    // Fails here
    //m1 *= m2;

    m1.display("m1 * m2");

    std::vector<Bounds<grid_precision>> bounds;
    bounds.push_back(Bounds<grid_precision>(-m2._width / 2, 1, std::abs(m1._width - 1 - (m2._width / 2))));
    bounds.push_back(Bounds<grid_precision>(-m2._height / 2, 1, std::abs(m1._height - 1 - (m2._height / 2))));

    //Grid<int32_t> translation_xy(2);
    Grid<grid_precision> translation_xy(grid_dimension, bounds);

    std::vector<float> search_resolution = {1.0f, 1.0f};
    translation_xy.setResolution(search_resolution);

    // first template argument is the error function return type
    // second template argument is the grid point value type
    CudaGridSearcher<func_precision, grid_precision> translation_xy_searcher(translation_xy);

    CudaImage<pixel_precision> *d_m1, *d_m2;
    ck(cudaMalloc((void **) &d_m1, sizeof(CudaImage<pixel_precision>)));
    ck(cudaMalloc((void **) &d_m2, sizeof(CudaImage<pixel_precision>)));
    cudaMemcpy(d_m1, &m1, sizeof(CudaImage<pixel_precision>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, &m2, sizeof(CudaImage<pixel_precision>),cudaMemcpyHostToDevice);
    // Copy device function pointer for the function having by-reference parameters to host side
    cudaMemcpyFromSymbol(&host_func_byref_ptr, dev_func_byref_ptr,
                         sizeof(image_err_func_byref));

    translation_xy_searcher.search_by_reference(host_func_byref_ptr, d_m1, d_m2);
    ck(cudaFree(d_m1));
    ck(cudaFree(d_m2));

    // Clean memory
    ck(cudaFree(m1._data));
    ck(cudaFree(m2._data));

    return EXIT_SUCCESS;
}