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
// Created by arwillis on 6/8/22.
//

#ifndef CUDAERRORFUNCTIONS_CUH
#define CUDAERRORFUNCTIONS_CUH

#include <nvVectorNd.h>
#include "cudaTensor.cuh"

// An example of a device function with by-value arguments

template<typename func_precision, typename grid_precision, uint32_t D, uint32_t CHANNELS, typename pixType>
CUDAFUNCTION func_precision
averageAbsoluteDifference(nv_ext::Vec<grid_precision, D> &t, CudaImage<pixType, CHANNELS> img_moved,
                          CudaImage<pixType, CHANNELS> img_fixed) {
    int num_errors = 0;
//    printf("img_moved ptr %p\n", img_moved._data);
//    printf("img_fixed ptr %p\n", img_fixed._data);
    float sum_of_absolute_differences = 0;
    for (int c = 0; c < CHANNELS; c++) {
        for (int x = 0; x < img_fixed.width(); x++) {
            for (int y = 0; y < img_fixed.height(); y++) {
                if (img_moved.inImage(y + t[1], x + t[0])) {
                    float image_moved_value = img_moved.valueAt(y + t[1], x + t[0], c);
                    //if (image_moved_value != cuda::std::numeric_limits<float>::infinity()) {
                    sum_of_absolute_differences += abs(image_moved_value - img_fixed.valueAt(y, x, c));
                    num_errors++;
//                printf("moved(%0.2f,%0.2f) - fixed(%0.2f,%0.2f) = %f - %f\n", (float) x + t[0], (float) y + t[1],
//                       (float) x, (float) y, (float) image_moved_value,
//                       (float) img_fixed.valueAt(x, y));

                }
            }
        }
    }

//    printf("evalXY_by_value(%0.2f,%0.2f) exiting with value %0.2f.\n", (float) t[0], (float) t[1],
//           sum_of_absolute_differences / num_errors);
    return (func_precision) sum_of_absolute_differences / num_errors;
}

template<typename func_precision, typename grid_precision, unsigned int D, uint32_t CHANNELS, typename pixType>
CUDAFUNCTION func_precision
sumOfAbsoluteDifferences(nv_ext::Vec<grid_precision, D> &t, CudaImage<pixType> img_moved,
                         CudaImage<pixType> img_fixed) {
//    printf("img_moved ptr %p\n", img_moved._data);
//    printf("img_fixed ptr %p\n", img_fixed._data);
    float sum_of_absolute_differences = 0;
    for (int c = 0; c < CHANNELS; c++) {
        for (int x = 0; x < img_fixed.width(); x++) {
            for (int y = 0; y < img_fixed.height(); y++) {
                if (img_moved.inImage(y + t[1], x + t[0])) {
                    float image_moved_value = img_moved.valueAt(y + t[1], x + t[0], c);
                    //if (image_moved_value != cuda::std::numeric_limits<float>::infinity()) {
                    sum_of_absolute_differences += abs(image_moved_value - img_fixed.valueAt(y, x, c));
//                printf("moved(%0.2f,%0.2f) - fixed(%0.2f,%0.2f) = %f - %f\n", (float) x + t[0], (float) y + t[1],
//                       (float) x, (float) y, (float) image_moved_value,
//                       (float) img_fixed.valueAt(x, y));

                }
            }
        }
    }
//    printf("evalXY_by_value(%0.2f,%0.2f) exiting with value %0.2f.\n", (float) t[0], (float) t[1],
//           sum_of_absolute_differences / num_errors);
    return (func_precision) sum_of_absolute_differences;
}

// An example of a device function with by-reference arguments

template<typename func_precision, typename grid_precision, unsigned int D, uint32_t CHANNELS, typename pixType>
CUDAFUNCTION func_precision
averageAbsoluteDifference(nv_ext::Vec<grid_precision, D> &t, CudaImage<pixType, 1> *img_moved,
                          CudaImage<pixType, 1> *img_fixed) {
    int num_errors = 0;
//    printf("img_moved ptr %p\n", img_moved);
//    printf("img_fixed ptr %p\n", img_fixed);
//    printf("img_moved data ptr %p\n", img_moved->_data);
//    printf("img_fixed data ptr %p\n", img_fixed->_data);
    float sum_of_absolute_differences = 0;
    for (int c = 0; c < CHANNELS; c++) {
        for (int x = 0; x < img_fixed->width(); x++) {
            for (int y = 0; y < img_fixed->height(); y++) {
                float image_moved_value = img_moved->valueAt(y + t[1], x + t[0], c);
                if (image_moved_value != cuda::std::numeric_limits<float>::infinity()) {
                    sum_of_absolute_differences += std::abs(image_moved_value - img_fixed->valueAt(y, x, c));
                    num_errors++;
                }
            }
        }
    }
//    printf("evalXY_by_reference(%0.2f,%0.2f) exiting.\n", (float) t[0], (float) t[1]);
    return (func_precision) sum_of_absolute_differences / num_errors;
}

template<typename func_precision, typename grid_precision, unsigned int D, uint32_t CHANNELS, typename pixType>
CUDAFUNCTION func_precision
sumOfAbsoluteDifferences(nv_ext::Vec<grid_precision, D> &t, CudaImage<pixType> *img_moved,
                         CudaImage<pixType> *img_fixed) {
//    printf("img_moved ptr %p\n", img_moved);
//    printf("img_fixed ptr %p\n", img_fixed);
//    printf("img_moved data ptr %p\n", img_moved->_data);
//    printf("img_fixed data ptr %p\n", img_fixed->_data);
    float sum_of_absolute_differences = 0;
    for (int c = 0; c < CHANNELS; c++) {
        for (int x = 0; x < img_fixed->width(); x++) {
            for (int y = 0; y < img_fixed->height(); y++) {
                float image_moved_value = img_moved->valueAt(y + t[1], x + t[0], c);
                if (image_moved_value != cuda::std::numeric_limits<float>::infinity()) {
                    sum_of_absolute_differences += std::abs(image_moved_value - img_fixed->valueAt(y, x, c));
                }
            }
        }
    }
//    printf("evalXY_by_reference(%0.2f,%0.2f) exiting.\n", (float) t[0], (float) t[1]);
    return (func_precision) sum_of_absolute_differences;
}

template<typename func_precision, typename grid_precision, unsigned int D = 8, uint32_t CHANNELS, typename pixType>
CUDAFUNCTION func_precision
averageAbsoluteDifferenceH(nv_ext::Vec<grid_precision, D> &H, CudaImage<pixType> img_moved,
                           CudaImage<pixType> img_fixed) {
    int num_errors = 0;
//    printf("img_moved ptr %p\n", img_moved._data);
//    printf("img_fixed ptr %p\n", img_fixed._data);
    float sum_of_absolute_differences = 0;
    for (int c = 0; c < CHANNELS; c++) {
        for (int x = 0; x < img_fixed.width(); x++) {
            for (int y = 0; y < img_fixed.height(); y++) {
                float z_p = H[6] * x + H[7] * y + 1.0;
                float y_p = (H[3] * x + H[4] * y + H[5]) / z_p;
                float x_p = (H[0] * x + H[1] * y + H[2]) / z_p;
                if (img_moved.inImage(y_p, x_p)) {
                    float image_moved_value = img_moved.valueAt(y_p, x_p, c);
                    //if (image_moved_value != cuda::std::numeric_limits<float>::infinity()) {
                    sum_of_absolute_differences += std::abs(image_moved_value - img_fixed.valueAt(y, x, c));
                    num_errors++;
//                printf("moved(%0.2f,%0.2f) - fixed(%0.2f,%0.2f) = %f - %f\n", (float) x + t[0], (float) y + t[1],
//                       (float) x, (float) y, (float) image_moved_value,
//                       (float) img_fixed.valueAt(x, y));

                }
            }
        }
    }
//    printf("evalXY_by_value(%0.2f,%0.2f) exiting with value %0.2f.\n", (float) t[0], (float) t[1],
//           sum_of_absolute_differences / num_errors);
    return (func_precision) sum_of_absolute_differences / num_errors;
}

template<typename func_precision, typename grid_precision, uint32_t D, uint32_t CHANNELS, typename pixType>
CUDAFUNCTION func_precision calcSQD(nv_ext::Vec<grid_precision, D> &H,
    CudaImage<pixType, CHANNELS> img_moved, CudaImage<pixType, CHANNELS> img_fixed) {
    
    int colsf = img_fixed.width();
    int rowsf = img_fixed.height();
    int colsm = img_moved.width();
    int rowsm = img_moved.height();

    func_precision output = 0;
    
    for(int x = 0; x < colsm; x++) {
        for (int y = 0; y < rowsm; y++) {
            float new_x = float(H[3]*(x-colsm/2)*cos(H[2]) - (y-rowsm/2)*sin(H[2]) + H[0] + H[3]*colsm/2);
            float new_y = float((x-colsm/2)*sin(H[2]) + H[3]*(y-rowsm/2)*cos(H[2]) + H[1] + H[3]*rowsm/2);

            for(int c = 0; c < CHANNELS; c++) {
                float temp = 0;
    
                if ((new_x >= 0 && new_x < colsf) && (new_y >= 0 && new_y < rowsf)) {
                    float value = img_fixed.valueAt_bilinear(new_y, new_x, c);
    
                    temp = value/255.0f;
                }
    
                output += (temp - img_moved.valueAt(y, x, c)/255.0f) * (temp - img_moved.valueAt(y, x, c)/255.0f);
            }
        }
    }

    return output;
}

template<typename func_precision, typename grid_precision, uint32_t D, uint32_t CHANNELS, typename pixType>
CUDAFUNCTION func_precision calcNCC(nv_ext::Vec<grid_precision, D> &H,
    CudaImage<pixType, CHANNELS> img_moved, CudaImage<pixType, CHANNELS> img_fixed) {

    int colsf = img_fixed.width();
    int rowsf = img_fixed.height();
    int colsm = img_moved.width();
    int rowsm = img_moved.height();

    float i1 = 0;
    float i2 = 0;
    float ic = 0;
    
    for(int x = 0; x < colsm; x++) {
        for (int y = 0; y < rowsm; y++) {
            float new_x = float(H[3]*(x-colsm/2)*cos(H[2]) - (y-rowsm/2)*sin(H[2]) + H[0] + H[3]*colsm/2);
            float new_y = float((x-colsm/2)*sin(H[2]) + H[3]*(y-rowsm/2)*cos(H[2]) + H[1] + H[3]*rowsm/2);

            for(int c = 0; c < CHANNELS; c++) {
                float temp = 0;
    
                if ((new_x >= 0 && new_x < colsf) && (new_y >= 0 && new_y < rowsf)) {
                    float value = img_fixed.valueAt_bilinear(new_y, new_x, c);
    
                    temp = value/255.0f;
                }
    
                i1 += img_moved.valueAt(y, x, c)/255.0f * img_moved.valueAt(y, x, c)/255.0f;
                i2 += temp * temp;
                ic += (img_moved.valueAt(y, x, c)/255.0f * temp) * (img_moved.valueAt(y, x, c)/255.0f * temp);
            }
        }
    }

    return (func_precision) -1 * ic / (i1 * i2);
}

#endif //CUDAERRORFUNCTIONS_CUH
