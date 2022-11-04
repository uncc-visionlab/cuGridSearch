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
#include "cudaImageFunctions.cuh"

// An example of a device function with by-value arguments

template<typename func_precision, typename grid_precision, uint32_t D, uint32_t CHANNELS, typename pixType>
CUDAFUNCTION func_precision
averageAbsoluteDifference(nv_ext::Vec<grid_precision, D> &parameters, CudaImage<pixType, CHANNELS> img_moved,
                          CudaImage<pixType, CHANNELS> img_fixed) {
    int num_errors = 0;
//    printf("img_moved ptr %p\n", img_moved._data);
//    printf("img_fixed ptr %p\n", img_fixed._data);
    if (parameters.size() != 8) {
        printf("Error averageAbsoluteDifference() requires an 8-parameter homography and %d parameters given! Exiting.\n",
               parameters.size());
    }
    float &theta = parameters[0];
    float &scaleX = parameters[1];
    float &scaleY = parameters[2];
    float &shearXY = parameters[3];
    float &translateX = parameters[4];
    float &translateY = parameters[5];
    float &keystoneX = parameters[6];
    float &keystoneY = parameters[7];
    float h11 = scaleX * cos(theta);
    float h12 = scaleY * shearXY * cos(theta) - scaleY * sin(theta);
    float h13 = scaleX * translateX;
    float h21 = scaleX * sin(theta);
    float h22 = scaleY * shearXY * sin(theta) + scaleY * cos(theta);
    float h23 = scaleY * translateY;
    float h31 = keystoneX;
    float h32 = keystoneY;
    float sum_of_absolute_differences = 0;
    for (int c = 0; c < CHANNELS; c++) {
        for (int x = 0; x < img_fixed.width(); x++) {
            for (int y = 0; y < img_fixed.height(); y++) {
                float new_w = ((h21 * h32 - h22 * h31) * x + (h12 * h31 - h11 * h32) * y + h11 * h22 - h12 * h21);
                float new_x = ((h22 - h23 * h32) * x + (h13 * h32 - h12) * y + h12 * h23 - h13 * h22) / new_w;
                float new_y = ((h23 * h31 - h21) * x + (h11 - h13 * h31) * y + h13 * h21 - h11 * h23) / new_w;
                if (img_moved.inImage(new_y, new_x)) {

                    float image_moved_value = img_moved.valueAt(new_y, new_x, c);
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
    int gridIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (gridIndex % 100000 == 0) {
        printf("solved problem index = %d\n", gridIndex);
    }
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
averageAbsoluteDifference(nv_ext::Vec<grid_precision, D> &parameters, CudaImage<pixType, 1> *img_moved,
                          CudaImage<pixType, 1> *img_fixed) {
    int num_errors = 0;
//    printf("img_moved ptr %p\n", img_moved);
//    printf("img_fixed ptr %p\n", img_fixed);
//    printf("img_moved data ptr %p\n", img_moved->_data);
//    printf("img_fixed data ptr %p\n", img_fixed->_data);
    if (parameters.size() != 8) {
        printf("Error averageAbsoluteDifference() requires an 8-parameter homography and %d parameters given! Exiting.\n",
               parameters.size());
    }
    float &theta = parameters[0];
    float &scaleX = parameters[1];
    float &scaleY = parameters[2];
    float &shearXY = parameters[3];
    float &translateX = parameters[4];
    float &translateY = parameters[5];
    float &keystoneX = parameters[6];
    float &keystoneY = parameters[7];
    float h11 = scaleX * cos(theta);
    float h12 = scaleY * shearXY * cos(theta) - scaleY * sin(theta);
    float h13 = scaleX * translateX;
    float h21 = scaleX * sin(theta);
    float h22 = scaleY * shearXY * sin(theta) + scaleY * cos(theta);
    float h23 = scaleY * translateY;
    float h31 = keystoneX;
    float h32 = keystoneY;
//    float &h11 = H[0], &h12 = H[1], &h13 = H[2], &h21 = H[3], &h22 = H[4], &h23 = H[5], &h31 = H[6], &h32 = H[7];
    float sum_of_absolute_differences = 0;
    for (int c = 0; c < CHANNELS; c++) {
        for (int x = 0; x < img_fixed->width(); x++) {
            for (int y = 0; y < img_fixed->height(); y++) {
                float new_w = ((h21 * h32 - h22 * h31) * x + (h12 * h31 - h11 * h32) * y + h11 * h22 - h12 * h21);
                float new_x = ((h22 - h23 * h32) * x + (h13 * h32 - h12) * y + h12 * h23 - h13 * h22) / new_w;
                float new_y = ((h23 * h31 - h21) * x + (h11 - h13 * h31) * y + h13 * h21 - h11 * h23) / new_w;
                if (img_moved->inImage(new_y, new_x)) {
                    float image_moved_value = img_moved->valueAt(new_y, new_x, c);
                    sum_of_absolute_differences += std::abs(image_moved_value - img_fixed->valueAt(y, x, c));
                    num_errors++;
                }
            }
        }
    }
//    printf("evalXY_by_reference(%0.2f,%0.2f) exiting.\n", (float) t[0], (float) t[1]);
    int gridIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (gridIndex % 100000 == 0) {
        printf("solved problem index = %d\n", gridIndex);
    }
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
CUDAFUNCTION func_precision calcSQDAlt(nv_ext::Vec<grid_precision, D> &parameters,
                                    CudaImage<pixType, CHANNELS> img_moved, CudaImage<pixType, CHANNELS> img_fixed) {
    func_precision output = 0;

    int colsf = img_fixed.width();
    int rowsf = img_fixed.height();
    int colsm = img_moved.width();
    int rowsm = img_moved.height();

    // if (parameters.size() != 8) {
    //     printf("Error calcSQD() requires an 8-parameter homography and %d parameters given! Exiting.\n",
    //            parameters.size());
    //     return 0;
    // }

    float h11 = 0, h12 = 0, h13 = 0, h21 = 0, h22 = 0, h23 = 0, h31 = 0, h32 = 0, cx = (float)colsm/2, cy = (float)rowsm/2;
    parametersToHomography<float,D>(parameters, cx, cy,
        h11, h12, h13,
        h21, h22, h23,
        h31, h32);
    
    float i1 = 0;
    float i2 = 0;

    float mov_area = 0;
    float fix_area = 0;

    for(int row = 0; row < rowsf; row++) {
        float lambda_start = 0;
        float lambda_end = colsf;
        float v_x = 0;
        float v_y = 0;
        float p0_x = 0;
        float p0_y = 0;
        bool inImage = false;

        parametricAssignValues(h11, h12, h13, h21, h22, h23, h31, h32,
                            colsm, rowsm, colsf, rowsf, row,
                            lambda_start, lambda_end, v_x, v_y, p0_x, p0_y, inImage);

        // printf("row = %d\n", row);
        // printf("lambda_start = %f, lambda_end = %f, v_x = %f, v_y = %f, p_x = %f, p_y = %f\n", lambda_start, lambda_end, v_x, v_y, p0_x, p0_y);

        float mag_norm = sqrt(v_x * v_x + v_y * v_y);

        for (float lambda = lambda_start; lambda < lambda_end && lambda < colsf; lambda += 1) {
            float new_x = lambda * v_x + p0_x;
            float new_y = lambda * v_y + p0_y;

            //if (img_moved.inImage(new_y, new_x)) {
            if (inImage){
                for(int c = 0; c < CHANNELS; c++) {
                    float value_m = img_moved.valueAt_bilinear(new_y, new_x, c);
                    float value_f = img_fixed.valueAt_bilinear(row, lambda, c);

                    value_m /= 255.0f;
                    value_f /= 255.0f;
                    output += (value_m - value_f) * (value_m - value_f);
                    i1 += value_m * value_m;
                    i2 += value_f * value_f;
                }

                mov_area += mag_norm;
                fix_area += 1;
            }
        }
    }
    // if(mov_area != 0 || fix_area != 0)
    //     printf("mov_area_overlap = %f, fix_area_overlap = %f\noutput = %f, i1 = %f, i2 = %f, outNorm = %f\n", mov_area / (colsm * rowsm * parameters[2]), fix_area / (colsf * rowsf), output, i1, i2, output / sqrt(i1 * i2));
    if(mov_area / (colsm * rowsm * parameters[2]) > 0.25 && fix_area / (colsf * rowsf) > 0.35)
        return (func_precision) output / sqrt(i1 * i2);
    else
        return (func_precision) 123123.0f;
}

template<typename func_precision, typename grid_precision, uint32_t D, uint32_t CHANNELS, typename pixType>
CUDAFUNCTION func_precision calcNCCAlt(nv_ext::Vec<grid_precision, D> &parameters,
                                    CudaImage<pixType, CHANNELS> img_moved, CudaImage<pixType, CHANNELS> img_fixed) {
    func_precision output = 0;

    int colsf = img_fixed.width();
    int rowsf = img_fixed.height();
    int colsm = img_moved.width();
    int rowsm = img_moved.height();

    // if (parameters.size() != 8) {
    //     printf("Error calcSQD() requires an 8-parameter homography and %d parameters given! Exiting.\n",
    //            parameters.size());
    //     return 0;
    // }

    float h11 = 0, h12 = 0, h13 = 0, h21 = 0, h22 = 0, h23 = 0, h31 = 0, h32 = 0, cx = (float)colsm/2, cy = (float)rowsm/2;
    parametersToHomography<float,D>(parameters, cx, cy,
        h11, h12, h13,
        h21, h22, h23,
        h31, h32);
    
    float i1 = 0;
    float i2 = 0;

    float mov_area = 0;
    float fix_area = 0;

    for(int row = 0; row < rowsf; row++) {
        float lambda_start = 0;
        float lambda_end = colsf;
        float v_x = 0;
        float v_y = 0;
        float p0_x = 0;
        float p0_y = 0;
        bool inImage = false;

        parametricAssignValues(h11, h12, h13, h21, h22, h23, h31, h32,
                            colsm, rowsm, colsf, rowsf, row,
                            lambda_start, lambda_end, v_x, v_y, p0_x, p0_y, inImage);

        // printf("row = %d\n", row);
        // printf("lambda_start = %f, lambda_end = %f, v_x = %f, v_y = %f, p_x = %f, p_y = %f\n", lambda_start, lambda_end, v_x, v_y, p0_x, p0_y);

        float mag_norm = sqrt(v_x * v_x + v_y * v_y);

        for (float lambda = lambda_start; lambda < lambda_end && lambda < colsf; lambda += 1) {
            float new_x = lambda * v_x + p0_x;
            float new_y = lambda * v_y + p0_y;

            //if (img_moved.inImage(new_y, new_x)) {
            if (inImage){
                for(int c = 0; c < CHANNELS; c++) {
                    float value_m = img_moved.valueAt_bilinear(new_y, new_x, c);
                    float value_f = img_fixed.valueAt_bilinear(row, lambda, c);

                    value_m /= 255.0f;
                    value_f /= 255.0f;
                    output += (value_m) * (value_f);
                    i1 += value_m * value_m;
                    i2 += value_f * value_f;
                }

                mov_area += mag_norm;
                fix_area += 1;
            }
        }
    }
    // if(mov_area != 0 || fix_area != 0)
    //     printf("mov_area_overlap = %f, fix_area_overlap = %f\noutput = %f, i1 = %f, i2 = %f, outNorm = %f\n", mov_area / (colsm * rowsm * parameters[2]), fix_area / (colsf * rowsf), output, i1, i2, output / sqrt(i1 * i2));
    if(mov_area / (colsm * rowsm * parameters[2]) > 0.25 && fix_area / (colsf * rowsf) > 0.35)
        return (func_precision) (-1 * output / sqrt(i1 * i2));
    else
        return (func_precision) 123123.0f;
}

template<typename func_precision, typename grid_precision, uint32_t D, uint32_t CHANNELS, typename pixType>
CUDAFUNCTION func_precision calcSQD(nv_ext::Vec<grid_precision, D> &parameters,
                                    CudaImage<pixType, CHANNELS> img_moved, CudaImage<pixType, CHANNELS> img_fixed) {

    int colsf = img_fixed.width();
    int rowsf = img_fixed.height();
    int colsm = img_moved.width();
    int rowsm = img_moved.height();

    // if (parameters.size() != 8) {
    //     printf("Error calcSQD() requires an 8-parameter homography and %d parameters given! Exiting.\n",
    //            parameters.size());
    //     return 0;
    // }

    float h11 = 0, h12 = 0, h13 = 0, h21 = 0, h22 = 0, h23 = 0, h31 = 0, h32 = 0, cx = (float)colsm/2, cy = (float)rowsm/2;
    parametersToHomography<float,D>(parameters, cx, cy,
        h11, h12, h13,
        h21, h22, h23,
        h31, h32);

    func_precision output = 0;
    float i1 = 0;
    float i2 = 0;

    float num_errors = 0;
    // 1. Calc 4 Intersection points
    // 2. Test 4 points to find entrance and exit point
    //   a. Test vy for top and botttom
    //   b. Test vx for left and right
    // 3. Find the fixed image line segment for that row. (Mag[V] * Width_f+(-1?))
    // 4. for lambda = lambda_start; lambda < lambda_exit && lambda < fixed_w * Mag[V] - lambda_start; lambda += Mag[V]
    // 5. x' = lambda * vx + px, y' = lambda * vy + py
    // 6. Do error calc at interpolated point x' y'
    // -- (Somewhere, outside the loop) Num_error += (lambda - lambda_start) * Mag[V] (in pix) (prob: lambda will be greater than lambda_exit)
    // -- (in the loop) Num_error += Mag[V] 
    for (int x = 0; x < colsf; x++) {
        for (int y = 0; y < rowsf; y++) {
            float new_x = 0; float new_y = 0;
            calcNewCoordH(h11, h12, h13,
                          h21, h22, h23,
                          h31, h32,
                          x, y,
                          new_x, new_y);
            // float new_w = ((h21 * h32 - h22 * h31) * x + (h12 * h31 - h11 * h32) * y + h11 * h22 - h12 * h21);
            // float new_x = ((h22 - h23 * h32) * x + (h13 * h32 - h12) * y + h12 * h23 - h13 * h22) / new_w;
            // float new_y = ((h23 * h31 - h21) * x + (h11 - h13 * h31) * y + h13 * h21 - h11 * h23) / new_w;

            for (int c = 0; c < CHANNELS; c++) {
                float temp = 0;

                if (img_moved.inImage(new_y, new_x)) {
                    float value = img_moved.valueAt_bilinear(new_y, new_x, c);

                    temp = value / 255.0f;
                    output += (temp - img_fixed.valueAt(y, x, c) / 255.0f) * (temp - img_fixed.valueAt(y, x, c) / 255.0f);
                    i1 += temp * temp;
                    i2 += img_fixed.valueAt(y, x, c) / 255.0f * img_fixed.valueAt(y, x, c) / 255.0f;
                    num_errors += 1;
                }
            }
        }
    }

    if(num_errors > 0)
        return (func_precision) output / sqrt(i1 * i2);
    else
        return (func_precision) 123123.0f;
}

template<typename func_precision, typename grid_precision, uint32_t D, uint32_t CHANNELS, typename pixType>
CUDAFUNCTION func_precision calcNCC(nv_ext::Vec<grid_precision, D> &parameters,
                                    CudaImage<pixType, CHANNELS> img_moved, CudaImage<pixType, CHANNELS> img_fixed) {

    int colsf = img_fixed.width();
    int rowsf = img_fixed.height();
    int colsm = img_moved.width();
    int rowsm = img_moved.height();

    if (parameters.size() != 8) {
        printf("Error calcNCC() requires an 8-parameter homography and %d parameters given! Exiting.\n",
               parameters.size());
        return 0;
    }

    float h11 = 0, h12 = 0, h13 = 0, h21 = 0, h22 = 0, h23 = 0, h31 = 0, h32 = 0, cx = (float)colsm/2, cy = (float)rowsm/2;
    parametersToHomography<float,8>(parameters, cx, cy,
        h11, h12, h13,
        h21, h22, h23,
        h31, h32);

    float i1 = 0;
    float i2 = 0;
    float ic = 0;

    float num_errors = 0;
    for (int x = 0; x < colsf; x++) {
        for (int y = 0; y < rowsf; y++) {
            float new_x = 0; float new_y = 0;
            calcNewCoordH(h11, h12, h13,
                          h21, h22, h23,
                          h31, h32,
                          x, y,
                          new_x, new_y);
            // float new_w = ((h21 * h32 - h22 * h31) * x + (h12 * h31 - h11 * h32) * y + h11 * h22 - h12 * h21);
            // float new_x = ((h22 - h23 * h32) * x + (h13 * h32 - h12) * y + h12 * h23 - h13 * h22) / new_w;
            // float new_y = ((h23 * h31 - h21) * x + (h11 - h13 * h31) * y + h13 * h21 - h11 * h23) / new_w;

            for (int c = 0; c < CHANNELS; c++) {
                float temp = 0;

                if (img_moved.inImage(new_y, new_x)) {
                    float value = img_moved.valueAt_bilinear(new_y, new_x, c);

                    temp = value / 255.0f;

                    i1 += (float)img_fixed.valueAt(y, x, c) / 255.0f * (float)img_fixed.valueAt(y, x, c) / 255.0f;
                    i2 += temp * temp;
                    ic += ((float)img_fixed.valueAt(y, x, c) / 255.0f * temp);
                    num_errors += 1;
                }
            }
        }
    }

    if(num_errors > 0)
        return (func_precision) (-1 * ic / sqrt(i1 * i2));
    else
        return (func_precision) 0;
}

#endif //CUDAERRORFUNCTIONS_CUH
