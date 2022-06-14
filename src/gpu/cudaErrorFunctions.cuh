//
// Created by arwillis on 6/8/22.
//

#ifndef CUDAERRORFUNCTIONS_CUH
#define CUDAERRORFUNCTIONS_CUH

#include <nvVectorNd.h>
#include "cudaTensor.cuh"

// An example of a device function with by-value arguments

template<typename func_precision, typename grid_precision, unsigned int D, typename pixType>
CUDAFUNCTION func_precision
averageAbsoluteDifference(nv_ext::Vec<grid_precision, D> &t, CudaImage<pixType> img_moved, CudaImage<pixType> img_fixed) {
    int num_errors = 0;
//    printf("img_moved ptr %p\n", img_moved._data);
//    printf("img_fixed ptr %p\n", img_fixed._data);
    float sum_of_absolute_differences = 0;
    for (int x = 0; x < img_fixed.width(); x++) {
        for (int y = 0; y < img_fixed.height(); y++) {
            if (img_moved.inImage(x + t[0], y + t[1])) {
                float image_moved_value = img_moved.valueAt(x + t[0], y + t[1]);
                //if (image_moved_value != cuda::std::numeric_limits<float>::infinity()) {
                sum_of_absolute_differences += std::abs(image_moved_value - img_fixed.valueAt(x, y));
                num_errors++;
//                printf("moved(%0.2f,%0.2f) - fixed(%0.2f,%0.2f) = %f - %f\n", (float) x + t[0], (float) y + t[1],
//                       (float) x, (float) y, (float) image_moved_value,
//                       (float) img_fixed.valueAt(x, y));

            }
        }
    }

//    printf("evalXY_by_value(%0.2f,%0.2f) exiting with value %0.2f.\n", (float) t[0], (float) t[1],
//           sum_of_absolute_differences / num_errors);
    return (func_precision) sum_of_absolute_differences / num_errors;
}

template<typename func_precision, typename grid_precision, unsigned int D, typename pixType>
CUDAFUNCTION func_precision
sumOfAbsoluteDifferences(nv_ext::Vec<grid_precision, D> &t, CudaImage<pixType> img_moved, CudaImage<pixType> img_fixed) {
//    printf("img_moved ptr %p\n", img_moved._data);
//    printf("img_fixed ptr %p\n", img_fixed._data);
    float sum_of_absolute_differences = 0;
    for (int x = 0; x < img_fixed.width(); x++) {
        for (int y = 0; y < img_fixed.height(); y++) {
            if (img_moved.inImage(x + t[0], y + t[1])) {
                float image_moved_value = img_moved.valueAt(x + t[0], y + t[1]);
                //if (image_moved_value != cuda::std::numeric_limits<float>::infinity()) {
                sum_of_absolute_differences += std::abs(image_moved_value - img_fixed.valueAt(x, y));
//                printf("moved(%0.2f,%0.2f) - fixed(%0.2f,%0.2f) = %f - %f\n", (float) x + t[0], (float) y + t[1],
//                       (float) x, (float) y, (float) image_moved_value,
//                       (float) img_fixed.valueAt(x, y));

            }
        }
    }

//    printf("evalXY_by_value(%0.2f,%0.2f) exiting with value %0.2f.\n", (float) t[0], (float) t[1],
//           sum_of_absolute_differences / num_errors);
    return (func_precision) sum_of_absolute_differences;
}

// An example of a device function with by-reference arguments

template<typename func_precision, typename grid_precision, unsigned int D, typename pixType>
CUDAFUNCTION func_precision
averageAbsoluteDifference(nv_ext::Vec<grid_precision, D> &t, CudaImage<pixType> *img_moved, CudaImage<pixType> *img_fixed) {
    int num_errors = 0;
//    printf("img_moved ptr %p\n", img_moved);
//    printf("img_fixed ptr %p\n", img_fixed);
//    printf("img_moved data ptr %p\n", img_moved->_data);
//    printf("img_fixed data ptr %p\n", img_fixed->_data);
    float sum_of_absolute_differences = 0;
    for (int x = 0; x < img_fixed->width(); x++) {
        for (int y = 0; y < img_fixed->height(); y++) {
            float image_moved_value = img_moved->valueAt(x + t[0], y + t[1]);
            if (image_moved_value != cuda::std::numeric_limits<float>::infinity()) {
                sum_of_absolute_differences += std::abs(image_moved_value - img_fixed->valueAt(x, y));
                num_errors++;
            }
        }
    }
//    printf("evalXY_by_reference(%0.2f,%0.2f) exiting.\n", (float) t[0], (float) t[1]);
    return (func_precision) sum_of_absolute_differences / num_errors;
}

template<typename func_precision, typename grid_precision, unsigned int D, typename pixType>
CUDAFUNCTION func_precision
sumOfAbsoluteDifferences(nv_ext::Vec<grid_precision, D> &t, CudaImage<pixType> *img_moved, CudaImage<pixType> *img_fixed) {
//    printf("img_moved ptr %p\n", img_moved);
//    printf("img_fixed ptr %p\n", img_fixed);
//    printf("img_moved data ptr %p\n", img_moved->_data);
//    printf("img_fixed data ptr %p\n", img_fixed->_data);
    float sum_of_absolute_differences = 0;
    for (int x = 0; x < img_fixed->width(); x++) {
        for (int y = 0; y < img_fixed->height(); y++) {
            float image_moved_value = img_moved->valueAt(x + t[0], y + t[1]);
            if (image_moved_value != cuda::std::numeric_limits<float>::infinity()) {
                sum_of_absolute_differences += std::abs(image_moved_value - img_fixed->valueAt(x, y));
            }
        }
    }
//    printf("evalXY_by_reference(%0.2f,%0.2f) exiting.\n", (float) t[0], (float) t[1]);
    return (func_precision) sum_of_absolute_differences;
}

template<typename func_precision, typename grid_precision, unsigned int D=8, typename pixType>
CUDAFUNCTION func_precision
averageAbsoluteDifferenceH(nv_ext::Vec<grid_precision, D> &H, CudaImage<pixType> img_moved, CudaImage<pixType> img_fixed) {
    int num_errors = 0;
//    printf("img_moved ptr %p\n", img_moved._data);
//    printf("img_fixed ptr %p\n", img_fixed._data);
    float sum_of_absolute_differences = 0;
    for (int x = 0; x < img_fixed.width(); x++) {
        for (int y = 0; y < img_fixed.height(); y++) {
            float z_p = H[6]*x + H[7]*y + 1.0;
            float y_p = (H[3]*x + H[4]*y + H[5]) / z_p;
            float x_p = (H[0]*x + H[1]*y + H[2]) / z_p;
            if (img_moved.inImage(x_p, y_p)) {
                float image_moved_value = img_moved.valueAt(x_p, y_p);
                //if (image_moved_value != cuda::std::numeric_limits<float>::infinity()) {
                sum_of_absolute_differences += std::abs(image_moved_value - img_fixed.valueAt(x, y));
                num_errors++;
//                printf("moved(%0.2f,%0.2f) - fixed(%0.2f,%0.2f) = %f - %f\n", (float) x + t[0], (float) y + t[1],
//                       (float) x, (float) y, (float) image_moved_value,
//                       (float) img_fixed.valueAt(x, y));

            }
        }
    }

//    printf("evalXY_by_value(%0.2f,%0.2f) exiting with value %0.2f.\n", (float) t[0], (float) t[1],
//           sum_of_absolute_differences / num_errors);
    return (func_precision) sum_of_absolute_differences / num_errors;
}


#endif //CUDAERRORFUNCTIONS_CUH
