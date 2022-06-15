//
// Created by arwillis on 6/14/22.
//

#ifndef CUDATENSORKERNELS_CUH
#define CUDATENSORKERNELS_CUH

#include <stdint.h>
#include <cuda/std/limits>

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
 */
template<class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduceMin6(T *g_idata, int *g_idxs, T *g_odata, int *g_oIdxs, unsigned int n) {
    T *sdata = SharedMemory<T>();
    // memory for indices is allocated after memory for data
    int *sdataIdx = (int *) (sdata + blockSize);

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    T myMin = 99999;
    int myMinIdx = -1;
    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n) {
        myMinIdx = MIN_IDX(g_idata[i], myMin, g_idxs[i], myMinIdx);
        myMin = MIN(g_idata[i], myMin);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) {
            //myMin += g_idata[i + blockSize];
            myMinIdx = MIN_IDX(g_idata[i + blockSize], myMin, g_idxs[i + blockSize], myMinIdx);
            myMin = MIN(g_idata[i + blockSize], myMin);
        }
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
            int tempMyMinIdx = __shfl_down(myMinIdx, offset);
            float tempMyMin = __shfl_down(myMin, offset);

            myMinIdx = MIN_IDX(tempMyMin, myMin, tempMyMinIdx, myMinIdx);
            myMin = MIN(tempMyMin, myMin);
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

__device__ void warp_reduce_min(volatile float smem[64]) {
    smem[threadIdx.x] = smem[threadIdx.x + 32] < smem[threadIdx.x] ? smem[threadIdx.x + 32] : smem[threadIdx.x];
    smem[threadIdx.x] = smem[threadIdx.x + 16] < smem[threadIdx.x] ? smem[threadIdx.x + 16] : smem[threadIdx.x];
    smem[threadIdx.x] = smem[threadIdx.x + 8] < smem[threadIdx.x] ? smem[threadIdx.x + 8] : smem[threadIdx.x];
    smem[threadIdx.x] = smem[threadIdx.x + 4] < smem[threadIdx.x] ? smem[threadIdx.x + 4] : smem[threadIdx.x];
    smem[threadIdx.x] = smem[threadIdx.x + 2] < smem[threadIdx.x] ? smem[threadIdx.x + 2] : smem[threadIdx.x];
    smem[threadIdx.x] = smem[threadIdx.x + 1] < smem[threadIdx.x] ? smem[threadIdx.x + 1] : smem[threadIdx.x];
}

__device__ void warp_reduce_max(volatile float smem[64]) {
    smem[threadIdx.x] = smem[threadIdx.x + 32] > smem[threadIdx.x] ? smem[threadIdx.x + 32] : smem[threadIdx.x];
    smem[threadIdx.x] = smem[threadIdx.x + 16] > smem[threadIdx.x] ? smem[threadIdx.x + 16] : smem[threadIdx.x];
    smem[threadIdx.x] = smem[threadIdx.x + 8] > smem[threadIdx.x] ? smem[threadIdx.x + 8] : smem[threadIdx.x];
    smem[threadIdx.x] = smem[threadIdx.x + 4] > smem[threadIdx.x] ? smem[threadIdx.x + 4] : smem[threadIdx.x];
    smem[threadIdx.x] = smem[threadIdx.x + 2] > smem[threadIdx.x] ? smem[threadIdx.x + 2] : smem[threadIdx.x];
    smem[threadIdx.x] = smem[threadIdx.x + 1] > smem[threadIdx.x] ? smem[threadIdx.x + 1] : smem[threadIdx.x];
}

/**
 * Device code to find a matrix extremal value
 *
 * @tparam precision - The matrix precision
 *
 * @param matrix - The matrix to set the value to
 * @param value - The value to set
 */
template<typename precision, int els_per_block, int threads>
__global__ void find_min_max(precision *in, precision *out) {
    __shared__ float smem_min[64];
    __shared__ uint32_t smem_idx[64];
//    __shared__ float smem_max[64];

    int tid = threadIdx.x + blockIdx.x * els_per_block;

//    float max = -cuda::std::numeric_limits<precision>::infinity();
    float min = cuda::std::numeric_limits<precision>::infinity();
    float val;

    const int iterations = els_per_block / threads;

#pragma unroll
    for (int i = 0; i < iterations; i++) {
        val = in[tid + i * threads];
        min = val < min ? val : min;
//        max = val > max ? val : max;
    }

    if (threads == 32) {
        smem_min[threadIdx.x + 32] = 0.0f;
//        smem_max[threadIdx.x+32] = 0.0f;
    }

    smem_min[threadIdx.x] = min;
//    smem_max[threadIdx.x] = max;

    __syncthreads();

    if (threadIdx.x < 32) {
        warp_reduce_min(smem_min);
//        warp_reduce_max(smem_max);
    }
    if (threadIdx.x == 0) {
        out[blockIdx.x] = smem_min[threadIdx.x]; // out[0] == ans
//        out[blockIdx.x + gridDim.x] = smem_max[threadIdx.x];
    }
}

template<typename precision, unsigned int D>
__global__ void
findExtremaProcess(CudaTensor<precision, D> tensor, CudaTensor<precision, D> extrema_value_buffer,
                   CudaTensor<uint32_t, 1> extrema_index_buffer) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x >= tensor.size()) {
        return;
    }
    *(tensor._data + x) = 0;
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

#endif //CUDATENSORKERNELS_CUH
