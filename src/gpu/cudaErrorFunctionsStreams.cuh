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

//
// Created by arwillis on 7/13/22.
//

#ifndef CUDAERRORFUNCTIONSSTREAMS_CUH
#define CUDAERRORFUNCTIONSSTREAMS_CUH

#include <nvVectorNd.h>
#include "cudaTensor.cuh"

// An example of a device function with by-value arguments
#define THREADS_PER_BLOCK 1024

template<typename func_precision, typename grid_precision, uint32_t D, uint32_t CHANNELS, typename pixType>
__device__ func_precision
averageAbsoluteDifference_stream(nv_ext::Vec<grid_precision, D> &t, CudaImage<pixType, CHANNELS> img_moved,
                                 CudaImage<pixType, CHANNELS> img_fixed) {
    int y = blockDim.x * blockIdx.x + threadIdx.x;
    if (y >= img_fixed.height()) {
        return;
    }
    // statically defined shared memory size since the invocation will be cudaKernel<< A, B, 0, stream>>()
    __shared__ float sum_of_absolute_differences[THREADS_PER_BLOCK];  //local block memory cache
    __shared__ int num_errors[THREADS_PER_BLOCK];  //local block memory cache
    //int sum_of_absolute_differences[THREADS_PER_BLOCK];  //local block memory cache
    //int num_errors[THREADS_PER_BLOCK];  //local block memory cache
//    printf("y = %d\n", y);
//    printf("img_moved ptr %p\n", img_moved._data);
//    printf("img_fixed ptr %p\n", img_fixed._data);
    num_errors[y] = 0;
    sum_of_absolute_differences[y] = 0;
    for (int c = 0; c < CHANNELS; c++) {
        for (int x = 0; x < img_fixed.width(); x++) {
            //for (int y = 0; y < img_fixed.height(); y++) {
            if (img_moved.inImage(y + t[1], x + t[0])) {
                float image_moved_value = img_moved.valueAt(y + t[1], x + t[0], c);
                //if (image_moved_value != cuda::std::numeric_limits<float>::infinity()) {
                sum_of_absolute_differences[y] += abs(image_moved_value - img_fixed.valueAt(y, x, c));
                num_errors[y]++;
//                printf("moved(%0.2f,%0.2f) - fixed(%0.2f,%0.2f) = %f - %f\n", (float) x + t[0], (float) y + t[1],
//                       (float) x, (float) y, (float) image_moved_value,
//                       (float) img_fixed.valueAt(x, y));
            }
            //}
        }
    }

    //synchronize the local threads writing to the local memory cache
    __syncthreads();

    float sum_of_absolute_differences_block = 0;
    int num_errors_block = 0;
    // have the first thread perform the global block error sum
    if (y == 0) {
        for (int row = 0; row < img_fixed.height(); row++) {
            sum_of_absolute_differences_block += sum_of_absolute_differences[row];
            num_errors_block += num_errors[row];
        }
        if (num_errors_block == 0) {
            num_errors_block = 1;
        }
        return (func_precision) sum_of_absolute_differences_block / num_errors_block;
    } else {
        return (func_precision) 0;
    }
}

#endif //CUDAERRORFUNCTIONSSTREAMS_CUH
