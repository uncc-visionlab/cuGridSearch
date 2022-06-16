//
// Created by arwillis on 6/14/22.
//

#ifndef CUDATENSORKERNELS_CUH
#define CUDATENSORKERNELS_CUH

#include <stdint.h>
#include <cuda/std/limits>

bool isPow2(unsigned int x) {
    return ((x & (x - 1)) == 0);
}

unsigned int nextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

#ifndef MIN_IDX
#define MIN_IDX(x, y, idx_x, idx_y) ((x < y) ? idx_x : idx_y)
#endif

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel
// 6, we observe the maximum specified number of blocks, because each thread in
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks,
                            int maxThreads, int &blocks, int &threads) {
    //get device capability, to avoid block/grid size exceed the upper bound
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    if (whichKernel < 3) {
        threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
        blocks = (n + threads - 1) / threads;
    } else {
        threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }
    if ((float) threads * blocks > (float) prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
        printf("n is too large, please choose a smaller number!\n");
    }
    if (blocks > prop.maxGridSize[0]) {
        printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
               blocks, prop.maxGridSize[0], threads * 2, threads);
        blocks /= 2;
        threads *= 2;
    }
    if (whichKernel == 6) {
        blocks = MIN(maxBlocks, blocks);
    }
}

// forward declaration of template class for __global__ device code
template<typename precision, unsigned int D>
class CudaTensor;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory {
    __device__ inline operator T *() {
        extern __shared__ int __smem[];
        return (T *) __smem;
    }

    __device__ inline explicit operator const T *() const {
        extern __shared__ int __smem[];
        return (T *) __smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double> {
    __device__ inline operator double *() {
        extern __shared__ double __smem_d[];
        return (double *) __smem_d;
    }

    __device__ inline operator const double *() const {
        extern __shared__ double __smem_d[];
        return (double *) __smem_d;
    }
};

/*
 This version adds multiple elements per thread sequentially.  This reduces the overall
 cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
 (Brent's Theorem optimization)

 Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
 In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
 If blockSize > 32, allocate blockSize*sizeof(T) bytes.

 https://stackoverflow.com/questions/38176136/finding-minimum-value-in-array-and-its-index-using-cuda-shfl-down-function

 */
template<class T, int blockSize>
__device__ void reduceMin6(T *g_idata, int *g_idxs, T *g_odata, int *g_oIdxs,
                           const unsigned int n) {
    T *sdata = SharedMemory<T>();
    // memory for indices is allocated after memory for data
    int *sdataIdx = (int *) (sdata + blockSize);

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    T myMin = cuda::std::numeric_limits<T>::infinity();
    int myMinIdx = -1;
    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n) {
        myMinIdx = MIN_IDX(g_idata[i], myMin, g_idxs[i], myMinIdx);
        myMin = MIN(g_idata[i], myMin);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (i + blockSize < n) {
            //myMin += g_idata[i + blockSize];
            myMinIdx = MIN_IDX(g_idata[i + blockSize], myMin, g_idxs[i + blockSize], myMinIdx);
            myMin = MIN(g_idata[i + blockSize], myMin);
        }
//        if (tid == 0 && blockIdx.x == 0) {
//            printf("(thread,block)=(%d,%d):: i1=%d, i2=%d, n=%d, myMin = %f, myMinIdx=%d\n", tid, blockIdx.x, i, i + blockSize, n, myMin, myMinIdx);
//        }
        i += gridSize;
    }
    // each thread puts its local sum into shared memory
    sdata[tid] = myMin;
    sdataIdx[tid] = myMinIdx;
    __syncthreads();
    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256)) {
        //sdata[tid] = mySum = mySum + sdata[tid + 256];
        sdataIdx[tid] = myMinIdx = MIN_IDX(sdata[tid + 256], myMin, sdataIdx[tid + 256], myMinIdx);
        sdata[tid] = myMin = MIN(sdata[tid + 256], myMin);
    }
    __syncthreads();
    if ((blockSize >= 256) && (tid < 128)) {
        //sdata[tid] = myMin = myMin + sdata[tid + 128];

        sdataIdx[tid] = myMinIdx = MIN_IDX(sdata[tid + 128], myMin, sdataIdx[tid + 128], myMinIdx);
        sdata[tid] = myMin = MIN(sdata[tid + 128], myMin);
    }
    __syncthreads();
    if ((blockSize >= 128) && (tid < 64)) {
        //sdata[tid] = myMin = myMin + sdata[tid + 64];

        sdataIdx[tid] = myMinIdx = MIN_IDX(sdata[tid + 64], myMin, sdataIdx[tid + 64], myMinIdx);
        sdata[tid] = myMin = MIN(sdata[tid + 64], myMin);
    }
    __syncthreads();
    if (tid < 32) {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >= 64) {
            //myMin += sdata[tid + 32];
            myMinIdx = MIN_IDX(sdata[tid + 32], myMin, sdataIdx[tid + 32], myMinIdx);
            myMin = MIN(sdata[tid + 32], myMin);
        }
        // Reduce final warp using shuffle
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            //myMin += __shfl_down(myMin, offset);
            int tempMyMinIdx = __shfl_down_sync(0xFFFFFFFF, myMinIdx, offset);
            float tempMyMin = __shfl_down_sync(0xFFFFFFFF, myMin, offset);

            myMinIdx = MIN_IDX(tempMyMin, myMin, tempMyMinIdx, myMinIdx);
            myMin = MIN(tempMyMin, myMin);
//            if (tid == 0 && blockIdx.x == 0) {
//                printf("(thread,block)=(%d,%d):: offset=%d, myMin = %f, myMinIdx=%d\n", tid, blockIdx.x, offset, (float) myMin, myMinIdx);
//            }
        }
    }
    __syncthreads();
    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = myMin;
        g_oIdxs[blockIdx.x] = myMinIdx;
    }
}

/**
 * Device code to set a matrix value to the given one
 *
 * @tparam precision - The matrix precision
 *
 * @param matrix - The matrix to set the value to
 * @param value - The value to set
 */
template<typename precision, unsigned int D>
__global__ void fillProcess(CudaTensor<precision, D> tensor, precision value) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x >= tensor.size()) {
        return;
    }
    *(tensor._data + x) = value;
}

/**
 * Device code to find a matrix extremal value
 *
 * @tparam precision - The matrix precision
 *
 * @param matrix - The matrix to set the value to
 * @param value - The value to set
 */
template<typename precision, unsigned int D>
__global__ void
findExtremaProcess(CudaTensor<precision, D> tensor,
                   CudaTensor<int32_t, 1> tensor_indices,
                   CudaTensor<precision, 1> device_block_extrema_values,
                   CudaTensor<int32_t, 1> device_block_extrema_indices) {
    // Do block reductions in parallel for each block
    switch (blockDim.x) {
        case 512:
            reduceMin6<precision, 512>(tensor.data(), tensor_indices.data(),
                                       device_block_extrema_values.data(), device_block_extrema_indices.data(), tensor.size());
            break;

        case 256:
            reduceMin6<precision, 256>(tensor.data(), tensor_indices.data(),
                                       device_block_extrema_values.data(), device_block_extrema_indices.data(), tensor.size());
            break;

        case 128:
            reduceMin6<precision, 128>(tensor.data(), tensor_indices.data(),
                                       device_block_extrema_values.data(), device_block_extrema_indices.data(), tensor.size());
            break;

        case 64:
            reduceMin6<precision, 64>(tensor.data(), tensor_indices.data(),
                                      device_block_extrema_values.data(), device_block_extrema_indices.data(), tensor.size());
            break;

        case 32:
            reduceMin6<precision, 32>(tensor.data(), tensor_indices.data(),
                                      device_block_extrema_values.data(), device_block_extrema_indices.data(), tensor.size());
            break;

        case 16:
            reduceMin6<precision, 16>(tensor.data(), tensor_indices.data(),
                                      device_block_extrema_values.data(), device_block_extrema_indices.data(), tensor.size());
            break;

        case 8:
            reduceMin6<precision, 8>(tensor.data(), tensor_indices.data(),
                                     device_block_extrema_values.data(), device_block_extrema_indices.data(), tensor.size());
            break;

        case 4:
            reduceMin6<precision, 4>(tensor.data(), tensor_indices.data(),
                                     device_block_extrema_values.data(), device_block_extrema_indices.data(), tensor.size());
            break;

        case 2:
            reduceMin6<precision, 2>(tensor.data(), tensor_indices.data(),
                                     device_block_extrema_values.data(), device_block_extrema_indices.data(), tensor.size());
            break;

        case 1:
            reduceMin6<precision, 1>(tensor.data(), tensor_indices.data(),
                                     device_block_extrema_values.data(), device_block_extrema_indices.data(), tensor.size());
            break;
    }
    __syncthreads();
    // compute global min
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        precision &extrema_value = device_block_extrema_values[0];
        int32_t &extrema_index1d = device_block_extrema_indices[0];
        for (int i = 1; i < gridDim.x; i++) {
//            printf("\n Reduce MIN GPU idx: %d  value: %f", device_block_extrema_indices[i], device_block_extrema_values[i]);
            if (device_block_extrema_values[i] < extrema_value) {
                extrema_value = device_block_extrema_values[i];
                extrema_index1d = device_block_extrema_indices[i];
            }
        }
        printf("\n Grid MIN value has idx: %d  value: %f", extrema_index1d, extrema_value);
        // decode the index to a grid point and put in the device_block_extrema_values at indices [1, ..., D]
    }
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

#endif //CUDATENSORKERNELS_CUH
