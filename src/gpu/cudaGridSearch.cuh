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
struct Bounds {
    precision _start;
    precision _end;
    precision _increment;

    Bounds(precision start, precision increment, precision end) :
            _start(start), _increment(increment), _end(end) {
        if (_end < _start) {
            std::cout << "Created invalid bounds. Behavior is undefined." << std::endl;
        }

    }

    void setIncrement(precision increment) {
        _increment = increment;
    }
};

template<typename precision>
struct Grid {
    std::vector<Bounds<precision>> _bounds;
    std::vector<float> _resolution;
    uint32_t _dimensions;

    Grid() {}

    Grid(int dimensions) :
            _dimensions(dimensions),
            _bounds(dimensions, Bounds<precision>(0, 1, 0)) {
    }

    Grid(int dimensions, std::vector<Bounds<precision>> &bounds) :
            _dimensions(dimensions), _resolution(dimensions, 1.0) {
        _bounds = bounds;
    }

    template<typename precision2>
    Grid(Grid<precision2> &grid) {
        for (typename std::vector<Bounds<precision2>>::iterator giter = grid._bounds.begin();
             giter != grid._bounds.end(); ++giter) {
            _bounds.push_back(Bounds<precision>(giter->_start, giter->_increment, giter->_end));
        }
    }

    void setResolution(std::vector<float> &resolution) {
        _resolution = resolution;
    }

    int getDimension() {
        return _dimensions;
    }

    std::vector<uint32_t> getGridSize() {
        std::vector<uint32_t> gridsize(_dimensions, 0);
        for (int d = 0; d < _dimensions; ++d) {
            gridsize[d] = (uint32_t) (1 + (_bounds[d]._end - _bounds[d]._start) / _resolution[d]);
        }
        return gridsize;
    }

    void fill(precision value);

    void display(const std::string &name = "") const;

    void setValuesFromVector(const std::vector<precision> vals) const;

};

//// Since C++ 11
template<typename func_precision, typename grid_precision, unsigned int D, typename ... Types>
using func_byvalue_t = func_precision (*)(nv_ext::Vec<grid_precision, D> &gridpt, Types ... arg_vals);


template<typename func_precision, typename grid_precision, unsigned int D, typename ... Types>
__global__ void evaluationKernel_by_value(func_precision *result,
                                          func_byvalue_t<func_precision, grid_precision, D, Types...> op,
                                          nv_ext::Vec<grid_precision, D> gridpt,
                                          Types ... arg_vals) {
    gridpt[0] = blockDim.x * blockIdx.x + threadIdx.x;
    gridpt[1] = 0;
    int offset = gridpt[0];
    *(result + offset) = (*op)(gridpt, arg_vals...);
    printf("func_byvalue_t %p setting gridvalue[%d] = %f\n", *op, offset, *(result + offset));
}

template<typename func_precision, typename grid_precision, unsigned int D, typename ... Types>
using func_byreference_t = func_precision (*)(nv_ext::Vec<grid_precision, D> &gridpt, Types *... arg_ptrs);

template<typename func_precision, typename grid_precision, unsigned int D, typename ... Types>
__global__ void evaluationKernel_by_reference(func_precision *result,
                                              func_byreference_t<func_precision, grid_precision, D, Types...> op,
                                              nv_ext::Vec<grid_precision, D> gridpt,
                                              Types *... arg_ptrs) {
    gridpt[0] = blockDim.x * blockIdx.x + threadIdx.x;
    gridpt[1] = 0;
    int offset = gridpt[0];
    *(result + offset) = (*op)(gridpt, arg_ptrs...);
    printf("func_byreference_t %p setting gridvalue[%d] = %f\n", *op, offset, *(result + offset));
}

template<typename func_precision, typename grid_precision>
struct CudaGridSearcher {
    Grid<grid_precision> _grid;

    CudaGridSearcher(Grid<grid_precision> &grid) {
        _grid = grid;
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

        std::vector<grid_precision> point(_grid.getDimension(), 0);

        std::vector<uint32_t> gridsize = _grid.getGridSize();
        uint64_t total_samples = std::accumulate(gridsize.begin(), gridsize.end(), 1,
                                                 std::multiplies<grid_precision>());
        std::cout << "Grid has " << total_samples << " search samples." << std::endl;

        nv_ext::Vec<grid_precision, 2> pt((grid_precision) 0.0f);
        CudaImage<func_precision> grid_values(6, 6);
        ck(cudaMalloc(&grid_values._data, grid_values.bytesSize()));
        grid_values.fill(11);

        evaluationKernel_by_value<<<1, 1>>>(grid_values._data, errorFunction,
                                            pt, arg_vals...);
        cudaDeviceSynchronize();

        grid_values.display();
        ck(cudaFree(grid_values._data));

    }

    // this will search a function with by-reference arguments

    template<typename ... Types>
    void search_by_reference(func_byreference_t<func_precision, grid_precision, 2, Types ...> errorFunction,
                             Types *... arg_ptrs) {

        nv_ext::Vec<grid_precision, 2> pt((grid_precision) 0.0f);
        CudaImage<func_precision> grid_values(6, 6);
        ck(cudaMalloc(&grid_values._data, grid_values.bytesSize()));
        grid_values.fill(12);

        evaluationKernel_by_reference<<<1, 1>>>(grid_values._data, errorFunction,
                                                pt, arg_ptrs...);
        cudaDeviceSynchronize();

        grid_values.display();
        ck(cudaFree(grid_values._data));
    }

};

#endif /* CUDAGRIDSEARCH_CUH */

