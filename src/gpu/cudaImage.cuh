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

#ifndef CUDAIMAGE_CUH
#define CUDAIMAGE_CUH

#define CUDAFUNCTION __host__ __device__

#include <cassert>
#include <stdint.h>
#include <string>
#include <vector>
#include <cuda/std/limits>

#define ck(x) x
typedef unsigned int uint32_t;

template<typename precision, unsigned int D>
class CudaTensor;

template<typename precision, unsigned int D>
__global__ void fillProcess(CudaTensor<precision, D> tensor, precision value) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (x >= tensor.size()) {
        return;
    }

    *(tensor._data + x) = value;
}

template<typename precision, typename T, unsigned int D>
__global__ void transformProcess(CudaTensor<precision, D> A,
                                 CudaTensor<precision, D> B,
                                 T transform) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (x >= A.size()) {
        return;
    }

    // transform(*(A._data + x), *(B._data + x)) seems to return nothing but do not crash ...

    *(A._data + x) = transform(*(A._data + x), *(B._data + x));
}

template<typename precision, unsigned int D>
struct CudaTensor {
    precision *_data;
    int32_t _dims[D];
    int32_t _dimensional_increments[D];
    int32_t _total_size;

    CUDAFUNCTION CudaTensor() : _data(nullptr), _total_size(0) {
        for (int d = 0; d < D; d++) {
            _dims[d] = 0;
            _dimensional_increments[d] = 0;
        }
    }

    template<typename T>
    CUDAFUNCTION CudaTensor(const T (&dims)[D]) : _data(nullptr) {
        // if on CPU
        //_data = new precision[_width * _height];
        _dimensional_increments[0] = 1;
        _dims[0] = dims[0];
        _total_size = dims[0];
        for (int d = 1; d < D; d++) {
            _dims[d] = dims[d];
            _total_size *= dims[d];
            _dimensional_increments[d] = _dims[d - 1] * _dimensional_increments[d - 1];
        }
    }

    CUDAFUNCTION ~CudaTensor() {
        // if on CPU
        //delete[] _data;
    }

    CUDAFUNCTION uint32_t size(const int dim = -1) const {
        if (dim == -1) {
            return _total_size;
        } else if (dim < D) {
            return _dims[dim];
        }
        return 0;
    }

    CUDAFUNCTION precision &at(const uint32_t (&pt)[D]) const {
        return _data[toIndex1d(pt)];
    }

    uint32_t bytesSize() const {
        return size() * sizeof(precision);
    }

    template<typename T>
    CUDAFUNCTION uint32_t toIndex1d(const T (&axis_index)[D]) const {
        // calculate 1-dimensional index from multi-dimensional index
        int indexEncoding = 0;
        for (int axis = 0; axis < D; axis++) {
            indexEncoding += _dimensional_increments[axis] * axis_index[axis];
        }
        return indexEncoding;
    }

    __device__ void toIndexNd(uint32_t index1d, uint32_t (&indexNd)[D]) const {
        for (int axis = D - 1; axis >= 0; axis--) {
            indexNd[axis] = ((uint32_t) index1d / _dimensional_increments[axis]);
            index1d -= indexNd[axis] * _dimensional_increments[axis];
        }
    }

    void fill(precision value) {
        const uint threadsPerBlock = 128;
        const uint numBlock = size() / threadsPerBlock + 1;

        // @fixme thrust fill method gives error after 1 iteration
        // thrust::device_ptr<precision> thrustPtr = thrust::device_pointer_cast(_data);
        // thrust::uninitialized_fill(thrustPtr, thrustPtr + size(), value);

        fillProcess <<< numBlock, threadsPerBlock >>>(*this, value);
    }

    void display(const std::string &name = "") const {
        precision *hostValues;

        ck(cudaMallocHost(&hostValues, bytesSize()));
        ck(cudaMemcpy(hostValues, _data, bytesSize(), cudaMemcpyDeviceToHost));

        std::cout << "Matrix " << name << " ";
        for (int d = 0; d < D; d++) {
            std::cout << _dims[d] << ((d < D - 1) ? " x " : "");
        }
        std::cout << " elements of " << typeid(precision).name() << "\n\n";

        std::cout << "{ ";
        for (int i = 0; i < _total_size; ++i) {

            //for (int j = 0; j < _width - 1; ++j) {
            std::cout << *(hostValues + i) << ((i < _total_size - 1) ? ", " : " ");
            //}

            //std::cout << *(hostValues + (i + 1) * _width - 1) << " }\n";
        }
        std::cout << "} ";

        std::cout << std::endl;

        ck(cudaFreeHost(hostValues));
    }

//    void setValuesFromVector(const std::vector<precision> vals) const;

    template<typename T>
    CudaTensor transform(const CudaTensor &A, T fn) {
        const uint threadsPerBlock = 128;
        const uint numBlock = size() / threadsPerBlock + 1;

        assert(_total_size == A._total_size);

        transformProcess <<< numBlock, threadsPerBlock >>>(*this, A, fn);

        return *this;
    }

//    CudaMatrix &operator=(CudaMatrix m);

//    CudaTensor operator+=(const CudaTensor &m) {
//        return transform(m, [=] __device__(precision x, precision y) { return x + y; });
//    }

//    CudaTensor operator-=(const CudaTensor &m) {
//        return transform(m, [=] __device__(precision x, precision y) { return x - y; });
//    }

//    CudaTensor operator*=(const CudaTensor &m) {
//        return transform(m, [=] __device__(precision x, precision y) { return x * y; });
//    }
};

//template
//class CudaTensor<float, 2>;

template<typename precision>
struct CudaMatrix {
    precision *_data;
    int32_t _width,
            _height;

    CUDAFUNCTION CudaMatrix() : _width(0), _height(0) {}

    CUDAFUNCTION CudaMatrix(uint32_t width, uint32_t height) :
            _width(width), _height(height) {
        // if on CPU
        //_data = new precision[_width * _height];
    }

    CUDAFUNCTION ~CudaMatrix() {
        // if on CPU
        //delete[] _data;
    }

    CUDAFUNCTION uint32_t size() const {
        return _width * _height;
    }

    CUDAFUNCTION int width() const {
        return _width;
    }

    CUDAFUNCTION int height() const {
        return _height;
    }

    CUDAFUNCTION precision *row(int column_index) {
        return (_data + column_index * _height);
    }

    CUDAFUNCTION precision &at(int x, int y) const {
        return _data[toIndex(x, y)];
    }

    template<typename T>
    CUDAFUNCTION int toIndex(T x, T y) const {
        // row-major ordering of _data
        return ((int) y) * _width + (int) x;
    }

    uint32_t bytesSize() const {
        return size() * sizeof(precision);
    }

    void fill(precision value);

    void display(const std::string &name = "") const;

    void setValuesFromVector(const std::vector<precision> vals) const;

    void setRowFromVector(int column_index, const std::vector<precision> vals) const;

    template<typename T>
    CudaMatrix transform(const CudaMatrix &A, T fn);

//    CudaMatrix &operator=(CudaMatrix m);

    CudaMatrix operator+=(const CudaMatrix &m) {
        return transform(m, [=] __device__(precision x, precision y) { return x + y; });
    }

    CudaMatrix operator-=(const CudaMatrix &m) {
        return transform(m, [=] __device__(precision x, precision y) { return x - y; });
    }

    CudaMatrix operator*=(const CudaMatrix &m) {
        return transform(m, [=] __device__(precision x, precision y) { return x * y; });
    }
};

template<typename precision>
struct CudaImage : public CudaMatrix<precision> {

    CUDAFUNCTION CudaImage(uint32_t _width, uint32_t _height) : CudaMatrix<precision>(_width, _height) {
    }

    CUDAFUNCTION ~CudaImage() {
    }

    CUDAFUNCTION bool inImage(float x, float y) const {
        if (x >= 0 && y >= 0 && x < this->_width && y < this->_height)
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
        // column-major ordering of _data
        return toIndex(column_index, 0);
    }

    CUDAFUNCTION precision &at(int column_index) const {
        return this->CudaMatrix<precision>::at(column_index, 0);
    }
};

#endif /* CUDAIMAGE_CUH */

