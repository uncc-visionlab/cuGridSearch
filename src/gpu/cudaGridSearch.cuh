/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   cudaMatrix.cuh
 * Author: arwillis
 *
 * Created on June 1, 2022, 6:10 PM
 */

// Related work:
// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.657.4308&rep=rep1&type=pdf

#ifndef CUDAGRIDSEARCH_CUH
#define CUDAGRIDSEARCH_CUH

#define CUDAFUNCTION __host__ __device__

#include <cassert>
#include <iostream>
#include <numeric>
#include <stdint.h>
#include <string>
#include <vector>
#include <nvVectorNd.h>

#include "helper_functions.h"
#include "helper_cuda.h"

#include "cudaTensor.cuh"

#define ck(x) x
typedef unsigned int uint32_t;
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// forward declare the CudaGrid class to allow __global__ CudaGrid kernel function declarations

template<typename precision, unsigned int D>
class CudaGrid;

// declare all CudaGrid __global__ kernel functions (eventually in cudaGridKernels.cuh) so they can be used in
// the CudaGrid class template definition

template<typename precision, unsigned int D>
__global__ void updateProcess(CudaGrid<precision, D> grid, int changed_axis);

template<typename precision, unsigned int D>
__global__ void gridPointProcess(CudaGrid<precision, D> grid, int index1d, precision *device_grid_point);

// CudaGrid class template definition

template<typename precision, unsigned int D>
struct CudaGrid : private CudaMatrix<precision> {
#define ROW_IDX_START 0
#define ROW_IDX_END 1
#define ROW_IDX_RESOLUTION 2
#define ROW_IDX_SAMPLE_COUNT 3

    CudaGrid() : CudaMatrix<precision>(D, 4) {
    }

    void getAxisSampleCounts(precision *axis_sample_counts) {
        cudaMemcpy(axis_sample_counts, this->axis_sample_count(), D * sizeof(precision),
                   cudaMemcpyDeviceToHost);
    }

    uint32_t numElements() {
        precision axis_sample_counts[D];
        int32_t total_size = 1;
        cudaMemcpy(&axis_sample_counts, this->axis_sample_count(), D * sizeof(precision),
                   cudaMemcpyDeviceToHost);
        for (int axis = 0; axis < D; axis++) {
            total_size = total_size * axis_sample_counts[axis];
        }
        return total_size;
    }

    void update(int changed_axis = -1) {
//        const uint threadsPerBlock = 128;
//        const uint numBlock = size() / threadsPerBlock + 1;
        updateProcess <<< 1, 1 >>>(*this, changed_axis);
        cudaDeviceSynchronize();
    }

    precision *&data() {
        return this->CudaMatrix<precision>::data();
    }

    uint32_t bytesSize() const {
        return this->CudaMatrix<precision>::bytesSize();
    }

    // return the vector of starting values in each dimension
    CUDAFUNCTION precision *start_point(int axis_index = 0) {
        return (this->_data + this->toIndex(0, ROW_IDX_START) + axis_index);
    }

    // return the vector of ending values in each dimension
    CUDAFUNCTION precision *end_point(int axis_index = 0) {
        return (this->_data + this->toIndex(0, ROW_IDX_END) + axis_index);
    }

    // return the vector of sample counts in each dimension
    CUDAFUNCTION precision *axis_sample_count(int axis_index = 0) {
        return (this->_data + this->toIndex(0, ROW_IDX_SAMPLE_COUNT) + axis_index);
    }

    // return the vector of resolutions in each dimension
    CUDAFUNCTION precision *resolution(int axis_index = 0) {
        return (this->_data + this->toIndex(0, ROW_IDX_RESOLUTION) + axis_index);
    }

    template<typename T>
    __device__ uint32_t gridPointToIndex(T *grid_point) {
        precision *resolutionArr = resolution();
        precision *start_pointArr = start_point();
        precision *axis_sample_countArr = axis_sample_count();
        int32_t axis_sample_indices[D];
        int32_t dimensional_increment = 1;

        // calculate point on grid closest to grid_point
#pragma unroll
        for (int axis = 0; axis < D; axis++) {
            axis_sample_indices[axis] = round(grid_point[axis] - start_pointArr[axis]) / resolutionArr[axis];
            assert(axis_sample_countArr[axis] >= 0);
        }

        int indexEncoding = 0;
#pragma unroll
        for (int axis = 0; axis < D; axis++) {
            indexEncoding += dimensional_increment * axis_sample_indices[axis];
            dimensional_increment *= axis_sample_countArr[axis];
        }

        return indexEncoding;
    }

    template<typename T>
    __device__ void indexToGridPoint(T index, precision *device_grid_point) {
        precision *resolutionArr = resolution();
        precision *start_pointArr = start_point();
        precision *axis_sample_countArr = axis_sample_count();
        int32_t axis_sample_indices[D];
        int32_t dimensional_increments[D];

        dimensional_increments[0] = 1;
#pragma unroll
        for (int axis = 1; axis < D; axis++) {
            dimensional_increments[axis] = axis_sample_countArr[axis - 1] * dimensional_increments[axis - 1];
        }

#pragma unroll
        for (int axis = D - 1; axis >= 0; axis--) {
            axis_sample_indices[axis] = ((int) index / dimensional_increments[axis]);
            assert(axis_sample_indices[axis] >= 0);
            index -= axis_sample_indices[axis] * dimensional_increments[axis];
        }

#pragma unroll
        for (int axis = 0; axis < D; axis++) {
            device_grid_point[axis] = start_pointArr[axis] + axis_sample_indices[axis] * resolutionArr[axis];
        }
    }

    __host__ void getGridPoint(precision (&grid_point)[D], int index) {
        precision *device_grid_point;
        ck(cudaMalloc(&device_grid_point, D * sizeof(precision)));
        gridPointProcess<<< 1, 1 >>>(*this, index, device_grid_point);
        cudaDeviceSynchronize();
        ck(cudaMemcpy(grid_point, device_grid_point, D * sizeof(precision), cudaMemcpyDeviceToHost));
        ck(cudaFree(device_grid_point));
    }

    template<typename T>
    void setStartPoint(std::vector<T> start_point) {
        this->setRowFromVector(ROW_IDX_START, start_point);
        update();
    }

    template<typename T>
    void setEndPoint(std::vector<T> end_point) {
        this->setRowFromVector(ROW_IDX_END, end_point);
        update();
    }

    template<typename T>
    void setResolution(std::vector<T> resolution) {
        this->setRowFromVector(ROW_IDX_RESOLUTION, resolution);
        update();
    }

    int getDimension() {
        return D;
    }

//    void fill(precision value);

    void display(const std::string &name = "") const {
        this->CudaMatrix<precision>::display(name);
    }

    void setValuesFromVector(const std::vector<precision> vals) const {
        this->CudaMatrix<precision>::setValuesFromVector(vals);
    }

};

// START put in cudaGridKernels.cuh?

template<typename precision, unsigned int D>
__global__ void updateProcess(CudaGrid<precision, D> grid, int changed_axis) {
    precision *start_pointArr = grid.start_point();
    precision *end_pointArr = grid.end_point();
    precision *resolutionArr = grid.resolution();
    precision *axis_sample_countArr = grid.axis_sample_count();
    if (changed_axis >= 0 && changed_axis < D) {
        axis_sample_countArr[changed_axis] =
                (int) (end_pointArr[changed_axis] - start_pointArr[changed_axis]) / resolutionArr[changed_axis];
    } else {
        for (int axis = 0; axis < D; axis++) {
            axis_sample_countArr[axis] = (int) 1 + (end_pointArr[axis] - start_pointArr[axis]) / resolutionArr[axis];
        }
    }
}

template<typename precision, unsigned int D>
__global__ void gridPointProcess(CudaGrid<precision, D> grid, int index1d, precision *device_grid_point) {
    grid.indexToGridPoint(index1d, device_grid_point);
}

// END put in cudaGridKernels.cuh?

//// Since C++ 11
template<typename func_precision, typename grid_precision, unsigned int D, typename ... Types>
using func_byvalue_t = func_precision (*)(nv_ext::Vec<grid_precision, D> &gridpt, Types ... arg_vals);

template<typename func_precision, typename grid_precision, unsigned int D, typename ... Types>
__global__ void evaluationKernel_by_value(CudaGrid<grid_precision, D> grid,
                                          func_precision *result,
                                          uint32_t max_thread_index,
                                          func_byvalue_t<func_precision, grid_precision, D, Types...> op,
                                          nv_ext::Vec<grid_precision, D> gridpt,
                                          Types ... arg_vals) {
    int threadIndex = (blockDim.x * blockIdx.x + threadIdx.x);
    if (threadIndex > max_thread_index) {
        return;
    }
//    grid_precision *grid_point = new grid_precision[D];
    grid_precision grid_point[D];// = new grid_precision[D];
    if (threadIndex > 300000) {
//        printf("index = %d\n", threadIndex);
    }
    grid.indexToGridPoint(threadIndex, grid_point);
    for (int d = 0; d < D; d++) {
        gridpt[d] = grid_point[d];
    }
//    printf("gridpt(%d,%d)\n", (int) grid_point[0], (int) grid_point[1]);
//    gridpt[0] = gridpt[1] = 0;
    *(result + threadIndex) = (*op)(gridpt, arg_vals...);
//    printf("func_byvalue_t %p setting gridvalue[%d] = %f\n", *op, threadIndex, *(result + threadIndex));
//    delete[] grid_point;
}

template<typename func_precision, typename grid_precision, unsigned int D, typename ... Types>
using func_byreference_t = func_precision (*)(nv_ext::Vec<grid_precision, D> &gridpt, Types *... arg_ptrs);

template<typename func_precision, typename grid_precision, unsigned int D, typename ... Types>
__global__ void evaluationKernel_by_reference(CudaGrid<grid_precision, D> grid,
                                              func_precision *result,
                                              uint32_t max_thread_index,
                                              func_byreference_t<func_precision, grid_precision, D, Types...> op,
                                              nv_ext::Vec<grid_precision, D> gridpt,
                                              Types *... arg_ptrs) {
    int threadIndex = (blockDim.x * blockIdx.x + threadIdx.x);
    if (threadIndex > max_thread_index) {
        return;
    }
    grid_precision *grid_point = new grid_precision[D];
//    printf("index = %d ", threadIndex);
    grid.indexToGridPoint(threadIndex, grid_point);
    for (int d = 0; d < D; d++) {
        gridpt[d] = grid_point[d];
    }
//    printf("gridpt(%d,%d)\n", (int) grid_point[0], (int) grid_point[1]);
//    gridpt[0] = gridpt[1] = 0;
    *(result + threadIndex) = (*op)(gridpt, arg_ptrs...);
//    printf("func_byreference_t %p setting gridvalue[%d] = %f\n", *op, threadIndex, *(result + threadIndex));
    delete[] grid_point;
}

template<typename func_precision, typename grid_precision, unsigned int D>
struct CudaGridSearcher {
#define BLOCK_DIM 512
    CudaGrid<grid_precision, D> *_grid;
    CudaTensor<func_precision, D> *_result;

    CudaGridSearcher(CudaGrid<grid_precision, D> &grid, CudaTensor<func_precision, D> &result) {
        _grid = &grid;
        _result = &result;
    }

    // default search procedure uses by-value function parameters

    template<typename ... Types>
    void search(func_byvalue_t<func_precision, grid_precision, D, Types ...> errorFunction, Types &... arg_vals) {
        search_by_value(errorFunction, arg_vals...);
    }

    // this will search a function with by-value arguments

    template<typename ... Types>
    void
    search_by_value(func_byvalue_t<func_precision, grid_precision, D, Types ...> errorFunction, Types &... arg_vals) {

        std::vector<grid_precision> point(_grid->getDimension(), 0);
        uint32_t total_samples = _grid->numElements();
        std::cout << "CudaGrid has " << total_samples << " search samples." << std::endl;
//        std::cout << "CudaTensor for results has " << _result->size() << " values." << std::endl;

        // compute 1D search grid, block and thread index pattern
        dim3 gridDim(1, 1, 1), blockDim(BLOCK_DIM, 1, 1);
        if (total_samples > BLOCK_DIM) {
            gridDim.x = (total_samples / (uint32_t) BLOCK_DIM) + 1;
        } else {
            blockDim.x = total_samples;
        }

        // create grid point and result image
        nv_ext::Vec<grid_precision, D> pt((grid_precision) 0.0f);

        evaluationKernel_by_value<<< gridDim, blockDim>>>(*_grid, (*_result).data(), total_samples, errorFunction,
                                                          pt, arg_vals...);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }

    // this will search a function with by-reference arguments

    template<typename ... Types>
    void search_by_reference(func_byreference_t<func_precision, grid_precision, D, Types ...> errorFunction,
                             Types *... arg_ptrs) {
        std::vector<grid_precision> point(_grid->getDimension(), 0);
        uint32_t total_samples = _grid->numElements();
        std::cout << "CudaGrid has " << total_samples << " search samples." << std::endl;

        // compute 1D search grid, block and thread index pattern
        dim3 gridDim(1, 1, 1), blockDim(BLOCK_DIM, 1, 1);
        if (total_samples > BLOCK_DIM) {
            gridDim.x = (total_samples / (uint32_t) BLOCK_DIM) + 1;
        } else {
            blockDim.x = total_samples;
        }

        // create grid point and result image
        nv_ext::Vec<grid_precision, D> pt((grid_precision) 0.0f);

        evaluationKernel_by_reference<<< gridDim, blockDim>>>(*_grid, (*_result).data(), total_samples, errorFunction,
                                                              pt, arg_ptrs...);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }

};

#endif /* CUDAGRIDSEARCH_CUH */

