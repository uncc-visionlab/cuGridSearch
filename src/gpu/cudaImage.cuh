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

    CUDAFUNCTION precision *column(int column_index) {
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

    CUDAFUNCTION bool inImage(float x, float y) const {
        if (x >= 0 && y >= 0 && x < _width && y < _height)
            return true;
        return false;
    }

    CUDAFUNCTION float valueAt(float x, float y) const {
        return valueAt_nearest_neighbor(x, y);
    }

    CUDAFUNCTION float valueAt_nearest_neighbor(float x, float y) const {
        return (inImage(x, y)) ? _data[toIndex(x, y)] : cuda::std::numeric_limits<float>::infinity();
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

    //CudaMatrix &operator=(CudaMatrix m);

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

