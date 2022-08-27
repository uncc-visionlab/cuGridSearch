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
// Created by arwillis on 7/27/22.
//

#ifndef CUGRIDSEARCH_CUDAIMAGEFUNCTIONS_CUH
#define CUGRIDSEARCH_CUDAIMAGEFUNCTIONS_CUH

#include <cstdlib>
#include <cstdint>
#include <string.h>
#include <vector>

#include <cuda.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

#include "cudaTensor.cuh"

#define CUDAFUNCTION __host__ __device__

template<typename pixType, uint8_t D, uint8_t CHANNELS>
__global__ void transformImage(nv_ext::Vec<float, D> H, int rowsm, int colsm,
                               CudaImage<pixType, CHANNELS> imageIn,
                               CudaImage<pixType, CHANNELS> imageOut) {
    // int colsm = imageOut.width();
    // int rowsm = imageOut.height();

    // Transform the image
    float &h11 = H[0], &h12 = H[1], &h13 = H[2], &h21 = H[3], &h22 = H[4], &h23 = H[5], &h31 = H[6], &h32 = H[7];
    for (int x = 0; x < colsm; x++) {
        for (int y = 0; y < rowsm; y++) {
            //float denom = (h11*h22 - h12*h21 - h11*h23*h32 + h12*h23*h31 + h13*h21*h32 - h13*h22*h31);
            float new_w = ((h21 * h32 - h22 * h31) * x + (h12 * h31 - h11 * h32) * y + h11 * h22 - h12 * h21);
            float new_x = ((h22 - h23 * h32) * x + (h13 * h32 - h12) * y + h12 * h23 - h13 * h22) / new_w;
            float new_y = ((h23 * h31 - h21) * x + (h11 - h13 * h31) * y + h13 * h21 - h11 * h23) / new_w;
            if (imageIn.inImage(new_y, new_x)) {
                for (int c = 0; c < CHANNELS; c++) {
                    imageOut.template at<float>(y, x, c) = imageIn.valueAt_bilinear(new_y, new_x, c);
                }
            }
        }
    }
}

template<typename pixType, uint8_t D, uint8_t CHANNELS>
__global__ void fuseAlignedImages(nv_ext::Vec<float, D> estimatedH,
                                  nv_ext::Vec<float, D> trueH,
                                  CudaImage<pixType, CHANNELS> imageReference,
                                  CudaImage<pixType, CHANNELS> imageMoving,
                                  CudaImage<pixType, 3> imageFused,
                                  int fusedChannel = 0) {
    int columns = imageFused.width();
    int rows = imageFused.height();
    float identityH_vals[] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    nv_ext::Vec<float, D> identityH(identityH_vals);
    nv_ext::Vec<float, D> H;
    // Transform the image
    for (int c = 0; c < 3; c++) {
        switch (c) {
            case 0:
                H = estimatedH;
                break;
            case 1: //GREEN
                H = identityH;
                break;
            case 2:
                H = trueH;
                break;
        }
        float &h11 = H[0], &h12 = H[1], &h13 = H[2], &h21 = H[3], &h22 = H[4], &h23 = H[5], &h31 = H[6], &h32 = H[7];
        for (int x = 0; x < columns; x++) {
            for (int y = 0; y < rows; y++) {
                //float denom = (h11*h22 - h12*h21 - h11*h23*h32 + h12*h23*h31 + h13*h21*h32 - h13*h22*h31);
                float new_w = ((h21 * h32 - h22 * h31) * x + (h12 * h31 - h11 * h32) * y + h11 * h22 - h12 * h21);
                float new_x = ((h22 - h23 * h32) * x + (h13 * h32 - h12) * y + h12 * h23 - h13 * h22) / new_w;
                float new_y = ((h23 * h31 - h21) * x + (h11 - h13 * h31) * y + h13 * h21 - h11 * h23) / new_w;
                switch (c) {
                    case 1:
                        if (imageReference.inImage(new_y, new_x)) {
                            imageFused.template at<float>(y, x, c) = imageReference.valueAt_bilinear(new_y, new_x,
                                                                                                     fusedChannel);
                        }
                        break;
                    case 0:
                    case 2:
                        if (imageMoving.inImage(new_y, new_x)) {
                            imageFused.template at<float>(y, x, c) = imageMoving.valueAt_bilinear(new_y, new_x,
                                                                                                  fusedChannel);
                        }
                        break;
                }
            }
        }
    }
}

/*
 * Case Insensitive Implementation of endsWith()
 * It checks if the string 'mainStr' ends with given string 'toMatch'
 */
bool endsWithCaseInsensitive(std::string mainStr, std::string toMatch) {
    auto it = toMatch.begin();
    return mainStr.size() >= toMatch.size() &&
           std::all_of(std::next(mainStr.begin(), mainStr.size() - toMatch.size()), mainStr.end(),
                       [&it](const char &c) {
                           return ::tolower(c) == ::tolower(*(it++));
                       });
}

template<typename pixType, uint8_t CHANNELS>
void
writeTransformedImageToDisk(CudaImage<pixType, CHANNELS> image, int rowsf, int colsf, nv_ext::Vec<float, 8> H, std::string img_out_filename) {
    CudaImage<uint8_t, CHANNELS> image_out(rowsf, colsf);
    checkCudaErrors(cudaMalloc(&image_out.data(), image_out.bytesSize()));
    checkCudaErrors(cudaMemset(image_out.data(), 0, image_out.bytesSize()));
    transformImage<uint8_t, 8, CHANNELS><<<1, 1>>>(H, rowsf, colsf, image, image_out);

    uint8_t *hostValues;
    checkCudaErrors(cudaMallocHost(&hostValues, image_out.bytesSize()));
    checkCudaErrors(cudaMemcpy(hostValues, image_out.data(), image_out.bytesSize(), cudaMemcpyDeviceToHost));
    if (endsWithCaseInsensitive(img_out_filename, ".png")) {
        stbi_write_png(img_out_filename.c_str(), image_out.width(), image_out.height(), CHANNELS, hostValues,
                       image.width() * sizeof(pixType) * CHANNELS);
        // You have to use 3 comp for complete jpg file. If not, the image will be grayscale or nothing.
    } else if (endsWithCaseInsensitive(img_out_filename, ".jpg")) {
        stbi_write_jpg(img_out_filename.c_str(), image_out.width(), image_out.height(), CHANNELS, hostValues, 95);
    } else {
        std::cout << "Filename suffix has image format not recognized." << std::endl;
    }
    cudaFreeHost(hostValues);
    checkCudaErrors(cudaFree(image_out.data()));
}

template<typename pixType, uint8_t CHANNELS>
void writeAlignedAndFusedImageToDisk(CudaImage<pixType, CHANNELS> image_fix,
                                     CudaImage<pixType, CHANNELS> image_mov,
                                     nv_ext::Vec<float, 8> estimatedH,
                                     nv_ext::Vec<float, 8> trueH,
                                     std::string img_fused_filename) {
    CudaImage<uint8_t, 3> image_fused(image_fix.height(), image_fix.width());
    checkCudaErrors(cudaMalloc(&image_fused.data(), image_fused.bytesSize()));
    checkCudaErrors(cudaMemset(image_fused.data(), 0, image_fused.bytesSize()));
    fuseAlignedImages<uint8_t, 8, CHANNELS><<<1, 1>>>(estimatedH, trueH, image_fix, image_mov, image_fused, 0);

    uint8_t *hostValues_fused;
    checkCudaErrors(cudaMallocHost(&hostValues_fused, image_fused.bytesSize()));
    checkCudaErrors(cudaMemcpy(hostValues_fused, image_fused.data(), image_fused.bytesSize(), cudaMemcpyDeviceToHost));
    if (endsWithCaseInsensitive(img_fused_filename, ".png")) {
        stbi_write_png(img_fused_filename.c_str(), image_fused.width(), image_fused.height(), 3, hostValues_fused,
                       image_fix.width() * sizeof(uint8_t) * 3);
        // You have to use 3 comp for complete jpg file. If not, the image will be grayscale or nothing.
    } else if (endsWithCaseInsensitive(img_fused_filename, ".jpg")) {
        stbi_write_jpg(img_fused_filename.c_str(), image_fused.width(), image_fused.height(), 3, hostValues_fused, 95);
    } else {
        std::cout << "Filename suffix has image format not recognized." << std::endl;
    }
    cudaFreeHost(hostValues_fused);
    checkCudaErrors(cudaFree(image_fused.data()));
}

template<typename Tp, uint8_t D>
CUDAFUNCTION void parametersToHomography(nv_ext::Vec<Tp, D> &parameters,
                            float &cx, float& cy,
                            float &h11, float &h12, float &h13,
                            float &h21, float &h22, float &h23,
                            float &h31, float &h32) {
    float &theta = parameters[0];
    float &scaleX = parameters[1];
    float &scaleY = parameters[2];
    float &shearXY = parameters[3];
    float &translateX = parameters[4];
    float &translateY = parameters[5];
    float &keystoneX = parameters[6];
    float &keystoneY = parameters[7];

    // Changed to one that calculates based off of center of moving image and follows the decomposition of C * K * T * R * Sh * Sc * C
    // This version closely resembles those of previous obtained homographies.

    h11 = scaleX*(cos(theta)*(cx*keystoneX + 1) + cx*keystoneY*sin(theta));
    h12 = scaleY*(shearXY*(cos(theta)*(cx*keystoneX + 1) + cx*keystoneY*sin(theta)) - sin(theta)*(cx*keystoneX + 1) + cx*keystoneY*cos(theta));
    h13 = cx + translateX*(cx*keystoneX + 1) - cx*scaleX*(cos(theta)*(cx*keystoneX + 1) + cx*keystoneY*sin(theta)) - cy*scaleY*(shearXY*(cos(theta)*(cx*keystoneX + 1) + cx*keystoneY*sin(theta)) - sin(theta)*(cx*keystoneX + 1) + cx*keystoneY*cos(theta)) + cx*keystoneY*translateY;
    h21 = scaleX*(sin(theta)*(cy*keystoneY + 1) + cy*keystoneX*cos(theta));
    h22 = scaleY*(shearXY*(sin(theta)*(cy*keystoneY + 1) + cy*keystoneX*cos(theta)) + cos(theta)*(cy*keystoneY + 1) - cy*keystoneX*sin(theta));
    h23 = cy + translateY*(cy*keystoneY + 1) - cx*scaleX*(sin(theta)*(cy*keystoneY + 1) + cy*keystoneX*cos(theta)) - cy*scaleY*(shearXY*(sin(theta)*(cy*keystoneY + 1) + cy*keystoneX*cos(theta)) + cos(theta)*(cy*keystoneY + 1) - cy*keystoneX*sin(theta)) + cy*keystoneX*translateX;
    h31 = scaleX*(keystoneX*cos(theta) + keystoneY*sin(theta));
    h32 = scaleY*(shearXY*(keystoneX*cos(theta) + keystoneY*sin(theta)) + keystoneY*cos(theta) - keystoneX*sin(theta));
    float h33 = keystoneX*translateX + keystoneY*translateY - cy*scaleY*(shearXY*(keystoneX*cos(theta) + keystoneY*sin(theta)) + keystoneY*cos(theta) - keystoneX*sin(theta)) - cx*scaleX*(keystoneX*cos(theta) + keystoneY*sin(theta)) + 1;
    
    h11 /= h33;
    h12 /= h33;
    h13 /= h33;
    h21 /= h33;
    h22 /= h33;
    h23 /= h33;
    h31 /= h33;
    h32 /= h33;


    // h11 = scaleX * cos(theta);
    // h12 = scaleY * shearXY * cos(theta) - scaleY * sin(theta);
    // h13 = scaleX * translateX;
    // h21 = scaleX * sin(theta);
    // h22 = scaleY * shearXY * sin(theta) + scaleY * cos(theta);
    // h23 = scaleY * translateY;
    // h31 = keystoneX;
    // h32 = keystoneY;
}

CUDAFUNCTION inline void calcNewCoordH(float &h11, float &h12, float &h13,
                   float &h21, float &h22, float &h23,
                   float &h31, float &h32,
                   int &x, int &y,
                   float &new_x, float &new_y) {
    float new_w = ((h21 * h32 - h22 * h31) * x + (h12 * h31 - h11 * h32) * y + h11 * h22 - h12 * h21);
    new_x = ((h22 - h23 * h32) * x + (h13 * h32 - h12) * y + h12 * h23 - h13 * h22) / new_w;
    new_y = ((h23 * h31 - h21) * x + (h11 - h13 * h31) * y + h13 * h21 - h11 * h23) / new_w;
}

template<typename Tp, uint8_t D>
void homographyToParameters(float &h11, float &h12, float &h13,
                            float &h21, float &h22, float &h23,
                            float &h31, float &h32, nv_ext::Vec<Tp, D> &parameters) {
    float theta = atan2(h21, h11);
    float scaleX = sqrt(h11 * h11 + h21 * h21);
    float shearXY_scaleY = h12 * cos(theta) + h22 * sin(theta);
    float scaleY;
    if (abs(sin(theta)) > 1.0e-6) {
        scaleY = (shearXY_scaleY * cos(theta) - h12) / sin(theta);
    } else {
        scaleY = (h22 - shearXY_scaleY * sin(theta)) / cos(theta);
    }
    float shearXY = shearXY_scaleY / scaleY;
    float translateX = h13 / scaleX;
    float translateY = h23 / scaleY;
    float &keystoneX = h31;
    float &keystoneY = h32;
    parameters[0] = theta;
    parameters[1] = scaleX;
    parameters[2] = scaleY;
    parameters[3] = shearXY;
    parameters[4] = translateX;
    parameters[5] = translateY;
    parameters[6] = keystoneX;
    parameters[7] = keystoneY;
}

#endif //CUGRIDSEARCH_CUDAIMAGEFUNCTIONS_CUH
