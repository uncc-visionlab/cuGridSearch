/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <iostream>
#include "cudaImage.cuh"

/**
 * Device code to set a matrix value to the given one
 *
 * @tparam precision - The matrix precision
 *
 * @param matrix - The matrix to set the value to
 * @param value - The value to set
 */
template <typename precision>
__global__ void fillProcess(CudaImage<precision> matrix, precision value) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (x >= matrix.size()) {
        return;
    }

    *(matrix._data + x) = value;
}

/**
 * Device code to apply a function f for each element of matrix A and B with A = f(A, B)
 *
 * @tparam precision - The matrix precision
 *
 * @param A - The matrix A to store the result in
 * @param B - The matrix B to compute the result from
 * @param transform - The function to apply on each A'elements such as A(i) = transform(A(i), B(i))
 */
template <typename precision, typename T>
__global__ void transformProcess(CudaImage<precision> A,
        CudaImage<precision> B,
        T transform) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (x >= A.size()) {
        return;
    }

    // transform(*(A._data + x), *(B._data + x)) seems to return nothing but do not crash ...

    *(A._data + x) = transform(*(A._data + x), *(B._data + x));
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
template<typename precision> template<typename T>
CudaImage<precision> CudaImage<precision>::transform(const CudaImage &A, T fn) {
    const uint threadsPerBlock = 128;
    const uint numBlock = size() / threadsPerBlock + 1;

    assert(_width == A._width);
    assert(_height == A._height);

    transformProcess << < numBlock, threadsPerBlock >>>(*this, A, fn);

    return *this;
}

/**
 * Fill the matrix with the given value
 *
 * @tparam precision - The matrix precision
 *
 * @param value - The value to set all matrix's elements with
 */
template <typename precision>
void CudaImage<precision>::fill(precision value) {
    const uint threadsPerBlock = 128;
    const uint numBlock = size() / threadsPerBlock + 1;

    // @fixme thrust fill method gives error after 1 iteration
    // thrust::device_ptr<precision> thrustPtr = thrust::device_pointer_cast(_data);
    // thrust::uninitialized_fill(thrustPtr, thrustPtr + size(), value);

    fillProcess << < numBlock, threadsPerBlock >>>(*this, value);
}

template <typename precision>
void CudaImage<precision>::setValuesFromVector(const std::vector<precision> vals) const {

    cudaMemcpy((*this)._data, vals.data(), vals.size() * sizeof (precision), cudaMemcpyHostToDevice);

}

/**
 * Display the matrix
 *
 * @tparam precision - The matrix precision
 *
 * @param name - The matrix name
 */
template <typename precision>
void CudaImage<precision>::display(const std::string &name) const {
    precision *hostValues;

    ck(cudaMallocHost(&hostValues, bytesSize()));
    ck(cudaMemcpy(hostValues, _data, bytesSize(), cudaMemcpyDeviceToHost));

    std::cout << "Matrix " << name << " " << _width << " x " << _height << " pixels of " << typeid (precision).name()
            << "\n\n";

    for (int i = 0; i < _height; ++i) {
        std::cout << "{ ";

        for (int j = 0; j < _width - 1; ++j) {
            std::cout << *(hostValues + i * _width + j) << ", ";
        }

        std::cout << *(hostValues + (i + 1) * _width - 1) << " }\n";
    }

    std::cout << std::endl;

    ck(cudaFreeHost(hostValues));
}
template class CudaImage<double>;
template class CudaImage<float>;
template class CudaImage<uint32_t>;
template class CudaImage<uint8_t>;
