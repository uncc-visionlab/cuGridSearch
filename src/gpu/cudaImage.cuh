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
struct CudaImage {
    precision *_data;
    int32_t _width,
            _height;

    CUDAFUNCTION CudaImage(uint32_t _width, uint32_t _height) :
            _width(_width), _height(_height) {
        // if on CPU
        //_data = new precision[_width * _height];
    }

    CUDAFUNCTION ~CudaImage() {
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

    CUDAFUNCTION precision &at(int x, int y) const {
        return _data[toIndex(x, y)];
    }

    template<typename T>
    CUDAFUNCTION int toIndex(T x, T y) const {
        // column-major ordering of _data
        return ((int) x) * _height + (int) y;
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

    template<typename T>
    CudaImage transform(const CudaImage &A, T fn);

    CudaImage &operator=(CudaImage m);

    CudaImage operator+=(const CudaImage &m) {
        return transform(m, [=] __device__(precision x, precision y) { return x + y; });
    }

    CudaImage operator-=(const CudaImage &m) {
        return transform(m, [=] __device__(precision x, precision y) { return x - y; });
    }

    CudaImage operator*=(const CudaImage &m) {
        return transform(m, [=] __device__(precision x, precision y) { return x * y; });
    }
};





class GridPointXY {
    int _x, _y;
    float _value;

public:

    GridPointXY(int x, int y, float value) : _value(value), _x(x), _y(y) {
    }

    int &x() {
        return _x;
    }

    int &y() {
        return _y;
    }

    float &value() {
        return _value;
    }
};

#endif /* CUDAIMAGE_CUH */

