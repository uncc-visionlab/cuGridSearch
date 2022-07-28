//
// Created by arwillis on 7/27/22.
//

#ifndef CUGRIDSEARCH_CUDAIMAGEFUNCTIONS_CUH
#define CUGRIDSEARCH_CUDAIMAGEFUNCTIONS_CUH

#include <cstdint>
#include <string.h>
#include <vector>

#include <cuda.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

#include "cudaTensor.cuh"

template<typename pixType, uint8_t D, uint8_t CHANNELS>
__global__ void transformImage(nv_ext::Vec<float, D> H,
                               CudaImage<pixType, CHANNELS> imageIn,
                               CudaImage<pixType, CHANNELS> imageOut) {
    int colsm = imageOut.width();
    int rowsm = imageOut.height();
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
                    imageOut.template at<float>(y, x) = imageIn.valueAt_bilinear(new_y, new_x);
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

template <typename pixType, uint8_t CHANNELS>
void writeTransformedImageToDisk(CudaImage<pixType,CHANNELS> image, nv_ext::Vec<float, 8> H, std::string img_out_filename) {
    CudaImage<uint8_t, CHANNELS> image_out(image.height(), image.width());
    checkCudaErrors(cudaMalloc(&image_out.data(), image_out.bytesSize()));
    checkCudaErrors(cudaMemset(image_out.data(), 0, image_out.bytesSize()));
    transformImage<uint8_t, 8, CHANNELS><<<1, 1>>>(H, image, image_out);

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

template <typename pixType, uint8_t CHANNELS>
void writeAlignedAndFusedImageToDisk(CudaImage<pixType,CHANNELS> image_fix,
                                     CudaImage<pixType,CHANNELS> image_mov,
                                     nv_ext::Vec<float, 8> estimatedH,
                                     nv_ext::Vec<float, 8> trueH,
                                     std::string img_fused_filename) {
    CudaImage<uint8_t, 3> image_fused(image_mov.height(), image_mov.width());
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

#endif //CUGRIDSEARCH_CUDAIMAGEFUNCTIONS_CUH
