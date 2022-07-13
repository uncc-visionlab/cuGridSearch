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

    CudaGrid() : CudaMatrix<precision>(4, D) {
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
        return (this->_data + this->toIndex(ROW_IDX_START, 0) + axis_index);
    }

    // return the vector of ending values in each dimension
    CUDAFUNCTION precision *end_point(int axis_index = 0) {
        return (this->_data + this->toIndex(ROW_IDX_END, 0) + axis_index);
    }

    // return the vector of sample counts in each dimension
    CUDAFUNCTION precision *axis_sample_count(int axis_index = 0) {
        return (this->_data + this->toIndex(ROW_IDX_SAMPLE_COUNT, 0) + axis_index);
    }

    // return the vector of resolutions in each dimension
    CUDAFUNCTION precision *resolution(int axis_index = 0) {
        return (this->_data + this->toIndex(ROW_IDX_RESOLUTION, 0) + axis_index);
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
//            printf("axis_index[%d] = %d\n", axis, axis_sample_indices[axis]);
        }

#pragma unroll
        for (int axis = 0; axis < D; axis++) {
            device_grid_point[axis] = start_pointArr[axis] + axis_sample_indices[axis] * resolutionArr[axis];
//            printf("gridpt[%d] = %f\n", axis, device_grid_point[axis]);
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
    void setNumSamples(std::vector<T> resolution) {
        //this->setRowFromVector(ROW_IDX_RESOLUTION, resolution);
        this->setRowFromVector(ROW_IDX_SAMPLE_COUNT, resolution);
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
//        if (abs(end_pointArr[changed_axis] - start_pointArr[changed_axis]) <
//            cuda::std::numeric_limits<precision>::epsilon() * axis_sample_countArr[changed_axis]) {
        if (abs(end_pointArr[changed_axis] - start_pointArr[changed_axis]) <
            1.0e-7 * axis_sample_countArr[changed_axis]) {
            printf("Requested step size is too small modifying axis sampling on axis index %d\n", changed_axis);
            axis_sample_countArr[changed_axis] = 1;
        }
        if (axis_sample_countArr[changed_axis] == 1) {
            resolutionArr[changed_axis] = 2 * (end_pointArr[changed_axis] - start_pointArr[changed_axis]);
        } else {
            resolutionArr[changed_axis] =
                    (int) (end_pointArr[changed_axis] - start_pointArr[changed_axis]) /
                    (axis_sample_countArr[changed_axis] - 1);
        }
    } else {
        for (int axis = 0; axis < D; axis++) {
            //assert(axis_sample_countArr[axis] >= 1);
//            if (abs(end_pointArr[axis] - start_pointArr[axis]) <
//                cuda::std::numeric_limits<precision>::epsilon() * axis_sample_countArr[changed_axis]) {
            if (abs(end_pointArr[axis] - start_pointArr[axis]) < 1.0e-7 * axis_sample_countArr[axis]) {
                printf("Requested step size is too small modifying axis sampling on axis index %d\n", axis);
                axis_sample_countArr[axis] = 1;
            }
            if (axis_sample_countArr[axis] == 1) {
                resolutionArr[axis] = 2 * (end_pointArr[changed_axis] - start_pointArr[changed_axis]);
            } else {
                resolutionArr[axis] = (end_pointArr[axis] - start_pointArr[axis]) / (axis_sample_countArr[axis] - 1);
            }
        }
    }
}

template<typename precision, unsigned int D>
__global__ void gridPointProcess(CudaGrid<precision, D> grid, int index1d, precision *device_grid_point) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        grid.indexToGridPoint(index1d, device_grid_point);
    }
}

// END put in cudaGridKernels.cuh?

//// Since C++ 11
template<typename func_precision, typename grid_precision, unsigned int D, typename ... Types>
using func_byvalue_t = func_precision (*)(nv_ext::Vec<grid_precision, D> &gridpt, Types ... arg_vals);

template<typename func_precision, typename grid_precision, unsigned int D, typename ... Types>
__global__ void evaluationKernel_by_value(CudaGrid<grid_precision, D> grid,
                                          func_precision *result,
                                          uint32_t result_size,
                                          func_byvalue_t<func_precision, grid_precision, D, Types...> op,
                                          nv_ext::Vec<grid_precision, D> gridpt,
                                          Types ... arg_vals) {
    int threadIndex = (blockDim.x * blockIdx.x + threadIdx.x);
    if (threadIndex > result_size) {
        return;
    }
//    grid_precision *grid_point = new grid_precision[D];
    grid_precision grid_point[D];// = new grid_precision[D];
//    if (threadIndex > 300000) {
//        printf("index = %d\n", threadIndex);
//    }
    grid.indexToGridPoint(threadIndex, grid_point);
#pragma unroll
    for (int d = 0; d < D; d++) {
        gridpt[d] = grid_point[d];
    }
//    printf("gridpt(%d,%d)\n", (int) grid_point[0], (int) grid_point[1]);
//    gridpt[0] = gridpt[1] = 0;
    *(result + threadIndex) = 0;
    *(result + threadIndex) += (*op)(gridpt, arg_vals...);
//    printf("func_byvalue_t %p setting gridvalue[%d] = %f\n", *op, threadIndex, *(result + threadIndex));
//    delete[] grid_point;
}

template<typename func_precision, typename grid_precision, unsigned int D, typename ... Types>
__global__ void evaluationKernel_by_value_stream(CudaGrid<grid_precision, D> grid,
                                                 func_precision *result,
                                                 uint32_t STREAM_BLOCK_DIM,
                                                 uint32_t stream_block_idx,
                                                 uint32_t stream_idx,
                                                 uint32_t result_size,
                                                 func_byvalue_t<func_precision, grid_precision, D, Types...> op,
                                                 nv_ext::Vec<grid_precision, D> gridpt,
                                                 Types ... arg_vals) {

    int streamIndex = stream_block_idx * STREAM_BLOCK_DIM + stream_idx;
    if (streamIndex > result_size) {
        return;
    }

    grid_precision grid_point[D];// = new grid_precision[D];

    grid.indexToGridPoint(streamIndex, grid_point);
#pragma unroll
    for (int d = 0; d < D; d++) {
        gridpt[d] = grid_point[d];
    }
    *(result + streamIndex) = 0;
    *(result + streamIndex) += (*op)(gridpt, arg_vals...);
}

template<typename func_precision, typename grid_precision, unsigned int D, typename ... Types>
using func_byreference_t = func_precision (*)(nv_ext::Vec<grid_precision, D> &gridpt, Types *... arg_ptrs);

template<typename func_precision, typename grid_precision, unsigned int D, typename ... Types>
__global__ void evaluationKernel_by_reference(CudaGrid<grid_precision, D> grid,
                                              func_precision *result,
                                              uint32_t result_size,
                                              func_byreference_t<func_precision, grid_precision, D, Types...> op,
                                              nv_ext::Vec<grid_precision, D> gridpt,
                                              Types *... arg_ptrs) {
    int threadIndex = (blockDim.x * blockIdx.x + threadIdx.x);
    if (threadIndex > result_size) {
        return;
    }
    grid_precision *grid_point = new grid_precision[D];
//    printf("index = %d ", threadIndex);
    grid.indexToGridPoint(threadIndex, grid_point);
#pragma unroll
    for (int d = 0; d < D; d++) {
        gridpt[d] = grid_point[d];
    }
//    printf("gridpt(%d,%d)\n", (int) grid_point[0], (int) grid_point[1]);
//    gridpt[0] = gridpt[1] = 0;
    *(result + threadIndex) = 0;
    *(result + threadIndex) += (*op)(gridpt, arg_ptrs...);
//    printf("func_byreference_t %p setting gridvalue[%d] = %f\n", *op, threadIndex, *(result + threadIndex));
    delete[] grid_point;
}

// TODO: 
template<typename func_precision, typename grid_precision, unsigned int D, typename ... Types>
__global__ void evaluationKernel_by_reference_stream(CudaGrid<grid_precision, D> grid,
                                                     func_precision *result,
                                                     uint32_t STREAM_BLOCK_DIM,
                                                     uint32_t stream_block_idx,
                                                     uint32_t stream_idx,
                                                     uint32_t result_size,
                                                     func_byreference_t<func_precision, grid_precision, D, Types...> op,
                                                     nv_ext::Vec<grid_precision, D> gridpt,
                                                     Types *... arg_ptrs) {

    int streamIndex = stream_block_idx * STREAM_BLOCK_DIM + stream_idx;
    if (streamIndex > result_size) {
        return;
    }
    grid_precision *grid_point = new grid_precision[D];
    grid.indexToGridPoint(streamIndex, grid_point);
#pragma unroll
    for (int d = 0; d < D; d++) {
        gridpt[d] = grid_point[d];
    }
    *(result + streamIndex) = 0;
    *(result + streamIndex) += (*op)(gridpt, arg_ptrs...);
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

#define MAX_THREADS_PER_BLOCK 1024

    // this will search a function with by-value arguments
    // TODO: Update to take into stream block size argument (Before arg_vals?)
    template<typename ... Types>
    void search_by_value_stream(func_byvalue_t<func_precision, grid_precision, D, Types ...> errorFunction,
                                int STREAM_BLOCK_DIM, int numCores, Types &... arg_vals) {

        std::vector<grid_precision> point(_grid->getDimension(), 0);
        uint32_t total_samples = _grid->numElements();
        std::cout << "CudaGrid has " << total_samples << " search samples." << std::endl;

        // allocate and initialize an array of stream handles
        // Found out that creating 10000 kernel streams takes up 1 GB of GPU memory
        // Need to use blocks to save on memory
        int numStreamBlocks = 1;
        int numStreams = STREAM_BLOCK_DIM;
        if (total_samples < STREAM_BLOCK_DIM) {
            numStreams = total_samples;
        } else {
            numStreamBlocks = (total_samples + STREAM_BLOCK_DIM) / STREAM_BLOCK_DIM;
        }
        int numBlocks = 1;
        int numThreads = numCores;
        if (numThreads > MAX_THREADS_PER_BLOCK) {
            numThreads = MAX_THREADS_PER_BLOCK;
            numBlocks = (numCores + MAX_THREADS_PER_BLOCK) / MAX_THREADS_PER_BLOCK;
        }
        cudaStream_t *streams = new cudaStream_t[numStreams];

        for (int i = 0; i < numStreams; i++) {
            checkCudaErrors(cudaStreamCreate(&(streams[i])));
        }

        // create CUDA event handles
        cudaEvent_t start_event, stop_event;
        checkCudaErrors(cudaEventCreate(&start_event));
        checkCudaErrors(cudaEventCreate(&stop_event));

        // the events are used for synchronization only and hence do not need to
        // record timings this also makes events not introduce global sync points when
        // recorded which is critical to get overlap

        cudaEvent_t *kernelEvent = new cudaEvent_t[numStreams];


        for (int i = 0; i < numStreams; i++) {
            checkCudaErrors(
                    cudaEventCreateWithFlags(&(kernelEvent[i]), cudaEventDisableTiming));
        }

        // create grid point and result image
        nv_ext::Vec<grid_precision, D> pt((grid_precision) 0.0f);

        cudaEventRecord(start_event, 0);

        // queue nkernels in separate streams and record when they are done
        for (int i = 0; i < numStreamBlocks; ++i) {
            // printf("Beginning Stream block %d\n",i);
            printf("Progress %.2f%% (%d/%d search blocks)\r", (float) (i + 1) * 100 / numStreamBlocks, i + 1,
                   numStreamBlocks);
            for (int j = 0; j < numStreams; ++j) {
                evaluationKernel_by_value_stream<<< numBlocks, numThreads, 0, streams[j]>>>(*_grid, (*_result).data(),
                                                                                  STREAM_BLOCK_DIM,
                                                                                  i, j, total_samples, errorFunction,
                                                                                  pt, arg_vals...);
                checkCudaErrors(cudaEventRecord(kernelEvent[j], streams[j]));

                // make the last stream wait for the kernel event to be recorded
                checkCudaErrors(cudaStreamWaitEvent(streams[numStreams - 1], kernelEvent[j], 0));
            }
            checkCudaErrors(cudaEventRecord(stop_event, 0));
            gpuErrchk(cudaPeekAtLastError());
            checkCudaErrors(cudaEventSynchronize(stop_event));
        }

        delete[] streams;
        delete[] kernelEvent;

        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    // this will search a function with by-reference arguments

    template<typename ... Types>
    void search_by_reference_stream(func_byreference_t<func_precision, grid_precision, D, Types ...> errorFunction,
                                    int STREAM_BLOCK_DIM, int numCores, Types *... arg_ptrs) {
        std::vector<grid_precision> point(_grid->getDimension(), 0);
        uint32_t total_samples = _grid->numElements();
        std::cout << "CudaGrid has " << total_samples << " search samples." << std::endl;

        // allocate and initialize an array of stream handles
        // Found out that creating 10000 kernel streams takes up 1 GB of GPU memory
        // Need to use blocks to save on memory
        int numStreamBlocks = 1;
        int numStreams = STREAM_BLOCK_DIM;
        if (total_samples < STREAM_BLOCK_DIM) {
            numStreams = total_samples;
        } else {
            numStreamBlocks = (total_samples + STREAM_BLOCK_DIM) / STREAM_BLOCK_DIM;
        }
        int numBlocks = 1;
        int numThreads = numCores;
        if (numThreads > MAX_THREADS_PER_BLOCK) {
            numThreads = MAX_THREADS_PER_BLOCK;
            numBlocks = (numCores + MAX_THREADS_PER_BLOCK) / MAX_THREADS_PER_BLOCK;
        }

        cudaStream_t *streams = new cudaStream_t[numStreams];

        for (int i = 0; i < numStreams; i++) {
            checkCudaErrors(cudaStreamCreate(&(streams[i])));
        }

        // create CUDA event handles
        cudaEvent_t start_event, stop_event;
        checkCudaErrors(cudaEventCreate(&start_event));
        checkCudaErrors(cudaEventCreate(&stop_event));

        // the events are used for synchronization only and hence do not need to
        // record timings this also makes events not introduce global sync points when
        // recorded which is critical to get overlap

        cudaEvent_t *kernelEvent = new cudaEvent_t[numStreams];


        for (int i = 0; i < numStreams; i++) {
            checkCudaErrors(cudaEventCreateWithFlags(&(kernelEvent[i]), cudaEventDisableTiming));
        }

        // create grid point and result image
        nv_ext::Vec<grid_precision, D> pt((grid_precision) 0.0f);

        cudaEventRecord(start_event, 0);

        // queue nkernels in separate streams and record when they are done
        for (int i = 0; i < numStreamBlocks; ++i) {
            //printf("Beginning Stream block %d\n",i);
            printf("Progress %.2f%% (%d/%d search blocks)\r", (float) (i + 1) * 100 / numStreamBlocks, i + 1,
                   numStreamBlocks);
            for (int j = 0; j < numStreams; ++j) {
                evaluationKernel_by_reference_stream<<< numBlocks, numThreads, 0, streams[j]>>>(*_grid, (*_result).data(),
                                                                                      STREAM_BLOCK_DIM, i, j,
                                                                                      total_samples,
                                                                                      errorFunction,
                                                                                      pt, arg_ptrs...);
                checkCudaErrors(cudaEventRecord(kernelEvent[j], streams[j]));

                // make the last stream wait for the kernel event to be recorded
                checkCudaErrors(
                        cudaStreamWaitEvent(streams[numStreams - 1], kernelEvent[j], 0));
            }
            checkCudaErrors(cudaEventRecord(stop_event, 0));
            gpuErrchk(cudaPeekAtLastError());
            checkCudaErrors(cudaEventSynchronize(stop_event));
        }

        delete[] streams;
        delete[] kernelEvent;

        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    // this will search a function with by-value arguments

    template<typename ... Types>
    void search_by_value(func_byvalue_t<func_precision, grid_precision, D, Types ...> errorFunction,
                         Types &... arg_vals) {

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

