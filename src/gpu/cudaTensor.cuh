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

#ifndef CUDATENSOR_CUH
#define CUDATENSOR_CUH

#define CUDAFUNCTION __host__ __device__

#include <cassert>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <cuda/std/limits>

// forward declare the CudaTensor class to allow __global__ CudaTensor kernel function declarations
template<typename precision, unsigned int D>
class CudaTensor;

// declare all CudaTensor __global__ and __device__ functions so they can be used in the CudaTensor
// class template definition
template<class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduceMin6(T *g_idata, int *g_idxs, T *g_odata, int *g_oIdxs, unsigned int n);

template<typename precision, unsigned int D>
__global__ void fillProcess(CudaTensor<precision, D> tensor, precision value);

__device__ void warp_reduce_min(volatile float smem[64]);

__device__ void warp_reduce_max(volatile float smem[64]);

template<typename precision, int els_per_block, int threads>
__global__ void find_min_max(precision *in, precision *out);

template<typename precision, typename T, unsigned int D>
__global__ void transformProcess(CudaTensor<precision, D> A,
                                 CudaTensor<precision, D> B,
                                 T transform);

template<typename precision, unsigned int D>
__global__ void
findExtremaProcess(CudaTensor<precision, D> tensor, CudaTensor<precision, D> extrema_value_buffer,
                   CudaTensor<uint32_t, 1> extrema_index_buffer);

template<typename precision, int els_per_block, int threads>
__global__ void find_min_max(precision *in, precision *out);

#define ck(x) x
typedef unsigned int uint32_t;

// CudaTensor class template definition

template<typename precision, unsigned int D>
struct __align__(16) CudaTensor {
    precision *_data;
    int32_t _dims[D];
    int32_t _dimensional_increments[D];
    int32_t _total_size;

    CUDAFUNCTION CudaTensor() : _data(nullptr), _total_size(0) {
#pragma unroll
        for (int d = 0; d < D; d++) {
            _dims[d] = 0;
            _dimensional_increments[d] = 0;
        }
    }

    template<typename T>
    CUDAFUNCTION CudaTensor(const T (&dims)[D]) : _data(nullptr) {
        _dimensional_increments[0] = 1;
        _dims[0] = dims[0];
        _total_size = dims[0];
#pragma unroll
        for (int d = 1; d < D; d++) {
            _dims[d] = dims[d];
            _total_size *= dims[d];
            _dimensional_increments[d] = _dims[d - 1] * _dimensional_increments[d - 1];
        }
        // if on CPU
        //_data = new precision[_total_size];
    }

    CUDAFUNCTION ~CudaTensor() {
        // if on CPU
        //delete[] _data;
    }

    precision *&data() {
        return _data;
    }

    CUDAFUNCTION uint32_t size(const int dim = -1) const {
        if (dim == -1) {
            return _total_size;
        } else if (dim < D) {
            return _dims[dim];
        }
        return 0;
    }

    CUDAFUNCTION int32_t width() const {
        return (int32_t) ((D == 2) ? _dims[0] : -1);
    }

    CUDAFUNCTION int32_t height() const {
        return (int32_t) ((D == 2) ? _dims[1] : -1);
    }

    CUDAFUNCTION precision &at(const uint32_t (&pt)[D]) const {
        return _data[toIndex1d(pt)];
    }

    uint32_t bytesSize() const {
        return size() * sizeof(precision);
    }

    template<typename T>
    CUDAFUNCTION int toIndex(T x, T y) const
    {
        return toIndex1d({x, y});
    }

    template<typename T>
    CUDAFUNCTION int toIndex1d(const T (&axis_index)[D]) const
    {
        // calculate 1-dimensional index from multi-dimensional index
        int indexEncoding = 0;
#pragma unroll
        for (int axis = 0; axis < D; axis++) {
            indexEncoding += _dimensional_increments[axis] * axis_index[axis];
        }
        return indexEncoding;
    }

    __device__ void toIndexNd(uint32_t index1d, uint32_t (&indexNd)[D]) const {
#pragma unroll
        for (int axis = D - 1; axis >= 0; axis--) {
            indexNd[axis] = ((uint32_t) index1d / _dimensional_increments[axis]);
            index1d -= indexNd[axis] * _dimensional_increments[axis];
        }
    }

    /**
     * Fill the matrix with the given value
     *
     * @tparam precision - The matrix precision
     *
     * @param value - The value to set all matrix's elements with
     */
    void fill(precision value) {
        const uint threadsPerBlock = 128;
        const uint numBlock = size() / threadsPerBlock + 1;

        // @fixme thrust fill method gives error after 1 iteration
        // thrust::device_ptr<precision> thrustPtr = thrust::device_pointer_cast(_data);
        // thrust::uninitialized_fill(thrustPtr, thrustPtr + size(), value);

        fillProcess <<< numBlock, threadsPerBlock >>>(*this, value);
    }

    /**
     * Find extreme values of the matrix and their indices.
     *
     * @tparam precision - The matrix precision
     *
     * @param value - The value to set all matrix's elements with
     */
    void find_extrema(precision &extrema_value, uint32_t &extrema_index1d) {
        const uint threadsPerBlock = 128;
        const uint numBlock = size() / threadsPerBlock + 1;
        CudaTensor<precision, D> extrema_value_buffer;
        CudaTensor<uint32_t, 1> extrema_index_buffer({this->_total_size});
        ck(cudaMalloc(&extrema_index_buffer.data(), extrema_index_buffer.bytesSize()));
        ck(cudaMalloc(&extrema_value_buffer.data(), extrema_value_buffer.bytesSize()));
        ck(cudaMemcpy(&extrema_value_buffer.data(), this->data(), this->bytesSize(), cudaMemcpyDeviceToDevice));
        std::vector<uint32_t> host_index_values(this->_total_size);
        std::iota (std::begin(host_index_values), std::end(host_index_values), 0);
        extrema_index_buffer.setValuesFromVector(host_index_values);
        int smemSize = (threadsPerBlock <= 32) ? 2 * threadsPerBlock * sizeof(precision) : threadsPerBlock * sizeof(precision);
        smemSize += threadsPerBlock*sizeof(int);
        findExtremaProcess <<< numBlock, threadsPerBlock, smemSize >>>(*this, extrema_value_buffer, extrema_index_buffer);

        ck(cudaMemcpy(&extrema_value, extrema_value_buffer.data(), 1 * sizeof(precision), cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(&extrema_index1d, extrema_index_buffer.data(), 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        ck(cudaFree(extrema_value_buffer.data()));
        ck(cudaFree(extrema_index_buffer.data()));
    }

    /**
     * Display the matrix
     *
     * @tparam precision - The matrix precision
     *
     * @param name - The matrix name
     */
    void display(const std::string &name = "") const {
        precision *hostValues;

        ck(cudaMallocHost(&hostValues, bytesSize()));
        ck(cudaMemcpy(hostValues, _data, bytesSize(), cudaMemcpyDeviceToHost));

        std::cout << "Matrix " << name << " ";
        for (int d = 0; d < D; d++) {
            std::cout << _dims[d] << ((d < D - 1) ? " x " : "");
        }
        std::cout << " elements of " << typeid(precision).name() << "\n\n";

        if (D != 2) {
            std::cout << "{ ";
            for (int i = 0; i < _total_size; ++i) {
                //for (int j = 0; j < _width - 1; ++j) {
                std::cout << *(hostValues + i) << ((i < _total_size - 1) ? ", " : " ");
                //}
                //std::cout << *(hostValues + (i + 1) * _width - 1) << " }\n";
            }
            std::cout << "} ";
        } else {
            for (int row = 0; row < _dims[1]; ++row) {
                std::cout << "{ ";
                for (int col = 0; col < _dims[0] - 1; ++col) {
                    std::cout << hostValues[toIndex(col, row)] << ((row < _total_size - 1) ? ", " : " ");
                }
                std::cout << hostValues[toIndex(_dims[0] - 1, row)] << " }\n";
            }
        }
        std::cout << std::endl;

        ck(cudaFreeHost(hostValues));
    }

    void setRowFromVector(int row_index, const std::vector<precision> vals) const {
        if (D == 2) {
            cudaMemcpy(((*this)._data + row_index * _dims[0]), vals.data(), vals.size() * sizeof(precision),
                       cudaMemcpyHostToDevice);
        } else {
            std::cout << "CudaTensor<" << typeid(precision).name() << D
                      << ">::setRowFromVector() not supported for tensors of dimension D = " << D << std::endl;
        }
    }

    void setValuesFromVector(const std::vector<precision> vals) const {
        cudaMemcpy((*this)._data, vals.data(), vals.size() * sizeof(precision), cudaMemcpyHostToDevice);
    }

    /**
     * Apply the function "fn" to all elements of the current matrix such as *this[i] = fn(*this[i], A[i])
     *
     * @tparam precision - The matrix precision
     *
     * @param A - The input matrix A
     * @param op - The binary function to apply
     *
     * @return This
     */
    template<typename T>
    CudaTensor transform(const CudaTensor &A, T fn) {
        const uint threadsPerBlock = 128;
        const uint numBlock = size() / threadsPerBlock + 1;

        assert(_total_size == A._total_size);

        transformProcess <<< numBlock, threadsPerBlock >>>(*this, A, fn);

        return *this;
    }

//    CudaMatrix &operator=(CudaMatrix m);

    CudaTensor operator+=(const CudaTensor &m) {
        return transform(m, [=] __device__(precision x, precision y) { return x + y; });
    }

    CudaTensor operator-=(const CudaTensor &m) {
        return transform(m, [=] __device__(precision x, precision y) { return x - y; });
    }

    CudaTensor operator*=(const CudaTensor &m) {
        return transform(m, [=] __device__(precision x, precision y) { return x * y; });
    }
};

template<typename precision>
struct CudaMatrix : public CudaTensor<precision, 2> {

    CUDAFUNCTION CudaMatrix(uint32_t width, uint32_t height) :
            CudaTensor<precision, 2>({width, height}) {
    }
};

template<typename precision>
struct CudaImage : public CudaMatrix<precision> {

    CUDAFUNCTION CudaImage(uint32_t _width, uint32_t _height) : CudaMatrix<precision>(_width, _height) {
    }

    CUDAFUNCTION ~CudaImage() {
    }

    CUDAFUNCTION bool inImage(float x, float y) const {
        if (x >= 0 && y >= 0 && x < this->width() && y < this->height())
            return true;
        return false;
    }

    CUDAFUNCTION float valueAt(float x, float y) const {
        //return valueAt_nearest_neighbor(x, y);
        return valueAt_bilinear(x, y);
    }

    CUDAFUNCTION float valueAt_nearest_neighbor(float x, float y) const {
        return (this->inImage(x, y)) ? this->_data[this->toIndex(x, y)] : cuda::std::numeric_limits<float>::infinity();
    }

    CUDAFUNCTION float valueAt_bilinear(float x, float y) const {
        if (this->inImage(x, y)) {
            float tlc = this->_data[this->toIndex(floor(x), floor(y))];
            float trc = this->_data[this->toIndex(ceil(x), floor(y))];
            float blc = this->_data[this->toIndex(floor(x), ceil(y))];
            float brc = this->_data[this->toIndex(ceil(x), ceil(y))];
            float alpha_x = x - floor(x);
            float alpha_y = y - floor(y);
            float value_top = (1.0f - alpha_x) * tlc + alpha_x * trc;
            float value_bottom = (1.0f - alpha_x) * blc + alpha_x * brc;
            float value = (1.0f - alpha_y) * value_top + alpha_y * value_bottom;
            return value;
        }
        return cuda::std::numeric_limits<float>::infinity();
    }
};

template<typename precision>
struct CudaVector : public CudaMatrix<precision> {

    CUDAFUNCTION CudaVector() : CudaMatrix<precision>(1, 0) {
    }

    CUDAFUNCTION CudaVector(uint32_t _dim) : CudaMatrix<precision>(1, _dim) {
    }

    CUDAFUNCTION ~CudaVector() {
    }

    template<typename T>
    CUDAFUNCTION int toIndex(T column_index) const {
        return toIndex(column_index, 0);
    }

    CUDAFUNCTION precision &at(int column_index) const {
        return this->CudaMatrix<precision>::at(column_index, 0);
    }
};

#include "cudaTensorKernels.cuh"

#endif /* CUDATENSOR_CUH */

