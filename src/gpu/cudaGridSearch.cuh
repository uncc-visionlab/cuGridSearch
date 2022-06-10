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

#include "cudaImage.cuh"

#define ck(x) x
typedef unsigned int uint32_t;

// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.657.4308&rep=rep1&type=pdf

template<typename precision>
struct CudaGrid : private CudaMatrix<precision> {
#define ROW_IDX_START 0
#define ROW_IDX_END 1
#define ROW_IDX_RESOLUTION 2
#define ROW_IDX_SAMPLE_COUNT 3
    uint32_t _dimensions;

    CudaGrid() : _dimensions(0) {}

    CudaGrid(int dimensions) : CudaMatrix<precision>(dimensions, 4),
                               _dimensions(dimensions) {
    }

    void getAxisSampleCounts(precision *axis_sample_counts) {
        cudaMemcpy(axis_sample_counts, this->axis_sample_count(), _dimensions * sizeof(precision),
                   cudaMemcpyDeviceToHost);
    }

    int numElements() {
        precision axis_sample_counts[_dimensions];
        int32_t total_size = 1;
        cudaMemcpy(&axis_sample_counts, this->axis_sample_count(), _dimensions * sizeof(precision),
                   cudaMemcpyDeviceToHost);
        for (int axis = 0; axis < _dimensions; axis++) {
            total_size = total_size * axis_sample_counts[axis];
        }
        return total_size;
    }

    void update(int changed_axis = -1) {
//        const uint threadsPerBlock = 128;
//        const uint numBlock = size() / threadsPerBlock + 1;
        updateProcess <<< 1, 1 >>>(*this, changed_axis);
    }

    precision *&data() {
        return this->_data;
    }

    uint32_t bytesSize() const {
        return this->size() * sizeof(precision);
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
    CUDAFUNCTION uint32_t gridPointToIndex(T *grid_point) {
        precision *resolutionArr = this->resolution();
        precision *start_pointArr = this->start_point();
        precision *axis_sample_countArr = this->axis_sample_count();
        int32_t axis_sample_indices[_dimensions];
        int32_t dimensional_increment = 1;

        // calculate point on grid closest to grid_point
        for (int axis = 0; axis < _dimensions; axis++) {
            axis_sample_indices[axis] = round(grid_point[axis] - start_pointArr[axis]) / resolutionArr[axis];
            assert(axis_sample_count(axis) >= 0);
        }

        int indexEncoding = 0;
        for (int axis = 0; axis < _dimensions; axis++) {
            indexEncoding += dimensional_increment * axis_sample_countArr[axis];
            dimensional_increment *= *axis_sample_countArr[axis];
        }

        return indexEncoding;
    }

    template<typename T>
    __device__ void indexToGridPoint(T index, precision *grid_point) {
        precision *resolutionArr = resolution();
        precision *start_pointArr = start_point();
        precision *axis_sample_countArr = axis_sample_count();
        int32_t *axis_sample_indices = new int[_dimensions];
        int32_t *dimensional_increments = new int[_dimensions];

        dimensional_increments[0] = 1;
        for (int axis = 1; axis < _dimensions; axis++) {
            dimensional_increments[axis] = axis_sample_countArr[axis - 1] * dimensional_increments[axis - 1];
        }

        for (int axis = _dimensions - 1; axis >= 0; axis--) {
            axis_sample_indices[axis] = ((int) index / dimensional_increments[axis]);
            assert(axis_sample_indices[axis] >= 0);
            index -= axis_sample_indices[axis] * dimensional_increments[axis];
        }

        for (int axis = 0; axis < _dimensions; axis++) {
            grid_point[axis] = start_pointArr[axis] + axis_sample_indices[axis] * resolutionArr[axis];
        }

        delete[] axis_sample_indices;
        delete[] dimensional_increments;
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
        return _dimensions;
    }

//    void fill(precision value);

    void display(const std::string &name = "") const {
        this->CudaMatrix<precision>::display(name);
    }

    void setValuesFromVector(const std::vector<precision> vals) const {
        this->CudaMatrix<precision>::setValuesFromVector(vals);
    }

};

template<typename precision>
__global__ void updateProcess(CudaGrid<precision> matrix, int changed_axis) {
//    int x = blockDim.x * blockIdx.x + threadIdx.x;
//    if (x >= matrix.size()) {
//        return;
//    }
//    *(matrix._data + x) = value;
    precision *start_pointArr = matrix.start_point();
    precision *end_pointArr = matrix.end_point();
    precision *resolutionArr = matrix.resolution();
    precision *axis_sample_countArr = matrix.axis_sample_count();
    if (changed_axis >= 0 && changed_axis < matrix._dimensions) {
        axis_sample_countArr[changed_axis] =
                (int) (end_pointArr[changed_axis] - start_pointArr[changed_axis]) / resolutionArr[changed_axis];
    } else {
        for (int axis = 0; axis < matrix._dimensions; axis++) {
            axis_sample_countArr[axis] = (int) 1 + (end_pointArr[axis] - start_pointArr[axis]) / resolutionArr[axis];
        }
    }
}

//// Since C++ 11
template<typename func_precision, typename grid_precision, unsigned int D, typename ... Types>
using func_byvalue_t = func_precision (*)(nv_ext::Vec<grid_precision, D> &gridpt, Types ... arg_vals);


template<typename func_precision, typename grid_precision, unsigned int D, typename ... Types>
__global__ void evaluationKernel_by_value(CudaGrid<grid_precision> grid,
                                          func_precision *result,
                                          func_byvalue_t<func_precision, grid_precision, D, Types...> op,
                                          nv_ext::Vec<grid_precision, D> gridpt,
                                          Types ... arg_vals) {
    int offset = (blockDim.x * blockIdx.x + threadIdx.x);
    grid_precision *grid_point = new grid_precision[D];
//    printf("index = %d ", offset);
    grid.indexToGridPoint(offset, grid_point);
    gridpt[0] = grid_point[0];
    gridpt[1] = grid_point[1];
//    printf("gridpt(%d,%d)\n", (int) grid_point[0], (int) grid_point[1]);
//    gridpt[0] = gridpt[1] = 0;
    *(result + offset) = (*op)(gridpt, arg_vals...);
//    printf("func_byvalue_t %p setting gridvalue[%d] = %f\n", *op, offset, *(result + offset));
    delete[] grid_point;
}

template<typename func_precision, typename grid_precision, unsigned int D, typename ... Types>
using func_byreference_t = func_precision (*)(nv_ext::Vec<grid_precision, D> &gridpt, Types *... arg_ptrs);

template<typename func_precision, typename grid_precision, unsigned int D, typename ... Types>
__global__ void evaluationKernel_by_reference(CudaGrid<grid_precision> grid,
                                              func_precision *result,
                                              func_byreference_t<func_precision, grid_precision, D, Types...> op,
                                              nv_ext::Vec<grid_precision, D> gridpt,
                                              Types *... arg_ptrs) {
    int offset = (blockDim.x * blockIdx.x + threadIdx.x);
    grid_precision *grid_point = new grid_precision[D];
//    printf("index = %d ", offset);
    grid.indexToGridPoint(offset, grid_point);
    gridpt[0] = grid_point[0];
    gridpt[1] = grid_point[1];
//    printf("gridpt(%d,%d)\n", (int) grid_point[0], (int) grid_point[1]);
//    gridpt[0] = gridpt[1] = 0;
    *(result + offset) = (*op)(gridpt, arg_ptrs...);
//    printf("func_byreference_t %p setting gridvalue[%d] = %f\n", *op, offset, *(result + offset));
    delete[] grid_point;
}

template<typename func_precision, typename grid_precision>
struct CudaGridSearcher {
    CudaGrid<grid_precision> *_grid;

    CudaGridSearcher(CudaGrid<grid_precision> &grid) {
        _grid = &grid;
    }

    // default search procedure uses by-value function parameters

    template<typename ... Types>
    void search(func_byvalue_t<func_precision, grid_precision, 2, Types ...> errorFunction, Types &... arg_vals) {
        search_by_value(errorFunction, arg_vals...);
    }

    // this will search a function with by-value arguments

    template<typename ... Types>
    void
    search_by_value(func_byvalue_t<func_precision, grid_precision, 2, Types ...> errorFunction, Types &... arg_vals) {

        std::vector<grid_precision> point(_grid->getDimension(), 0);
        uint64_t total_samples = _grid->numElements();
        std::cout << "CudaGrid has " << total_samples << " search samples." << std::endl;

        // compute 1D search grid, block and thread index pattern
        dim3 gridDim(1, 1, 1), blockDim(1, 1, 1);
        if (total_samples > 512) {
            blockDim.x = 512;
            gridDim.x = total_samples / 512 + (total_samples % 512 == 0) ? 1 : 0;
        } else {
            blockDim.x = total_samples;
        }

        // create grid point and result image
        nv_ext::Vec<grid_precision, 2> pt((grid_precision) 0.0f);
        grid_precision axis_sample_counts[_grid->getDimension()];
        _grid->getAxisSampleCounts(axis_sample_counts);
        CudaMatrix<func_precision> grid_values(axis_sample_counts[0], axis_sample_counts[1]);
        ck(cudaMalloc(&grid_values._data, grid_values.bytesSize()));
        grid_values.fill(0);

        evaluationKernel_by_value<<< gridDim, blockDim>>>(*_grid, grid_values._data, errorFunction,
                                            pt, arg_vals...);
        cudaDeviceSynchronize();

        grid_values.display();
        ck(cudaFree(grid_values._data));

    }

    // this will search a function with by-reference arguments

    template<typename ... Types>
    void search_by_reference(func_byreference_t<func_precision, grid_precision, 2, Types ...> errorFunction,
                             Types *... arg_ptrs) {
        std::vector<grid_precision> point(_grid->getDimension(), 0);
        uint64_t total_samples = _grid->numElements();
        std::cout << "CudaGrid has " << total_samples << " search samples." << std::endl;

        // compute 1D search grid, block and thread index pattern
        dim3 gridDim(1, 1, 1), blockDim(1, 1, 1);
        if (total_samples > 512) {
            blockDim.x = 512;
            gridDim.x = total_samples / 512 + (total_samples % 512 == 0) ? 1 : 0;
        } else {
            blockDim.x = total_samples;
        }

        // create grid point and result image
        nv_ext::Vec<grid_precision, 2> pt((grid_precision) 0.0f);
        grid_precision axis_sample_counts[_grid->getDimension()];
        _grid->getAxisSampleCounts(axis_sample_counts);
        CudaMatrix<func_precision> grid_values(axis_sample_counts[0], axis_sample_counts[1]);
        ck(cudaMalloc(&grid_values._data, grid_values.bytesSize()));
        grid_values.fill(0);

        evaluationKernel_by_reference<<< gridDim, blockDim>>>(*_grid, grid_values._data, errorFunction,
                                                pt, arg_ptrs...);
        cudaDeviceSynchronize();

        grid_values.display();
        ck(cudaFree(grid_values._data));
    }

};

#endif /* CUDAGRIDSEARCH_CUH */

