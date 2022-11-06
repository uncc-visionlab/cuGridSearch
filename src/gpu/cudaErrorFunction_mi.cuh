#ifndef CUDAERRORFUNCTIONS_MI_CUH
#define CUDAERRORFUNCTIONS_MI_CUH

#include <nvVectorNd.h>
#include "cudaTensor.cuh"
#include "cudaImageFunctions.cuh"

template<typename func_precision, typename grid_precision, uint8_t D = 8, uint8_t CHANNELS, typename pixType>
CUDAFUNCTION func_precision calcMI(nv_ext::Vec<grid_precision, D> &parameters,
                                    CudaImage<pixType, CHANNELS> img_moved, CudaImage<pixType, CHANNELS> img_fixed) {
    // int binIdx = blockDim.x * blockIdx.x + threadIdx.x;
    // px = moving image (pre-calculated)
    // py = fixed image (zeroed)
    // pxy = 2d hist (zeroed)
    // TODO: Try CudaImage type for px,py,pxy
    int colsf = img_fixed.width();
    int rowsf = img_fixed.height();
    int colsm = img_moved.width();
    int rowsm = img_moved.height();
    int binN = 64;
    // // Output holder
    func_precision output = 0;

    // printf("Starting %d\n", blockIdx.x * blockDim.x + threadIdx.x);

    float h11 = 0, h12 = 0, h13 = 0, h21 = 0, h22 = 0, h23 = 0, h31 = 0, h32 = 0, cx = (float)colsm/2, cy = (float)rowsm/2;
    parametersToHomography<float,8>(parameters, cx, cy,
        h11, h12, h13,
        h21, h22, h23,
        h31, h32);


    float test_pxy[64][64] = {{0}};
    float test_py[64] = {0};
    float test_px[64] = {0};

    float i1 = 0;
    float i2 = 0;

    // Transform the image
    int count = 0;
    for (int x = 0; x < colsf; x++) {
        for (int y = 0; y < rowsf; y++) {
            float new_x = 0, new_y = 0;
            calcNewCoordH(h11, h12, h13,
                h21, h22, h23,
                h31, h32,
                x, y,
                new_x, new_y);
            for (int c = 0; c < CHANNELS; c++) {
                if (img_moved.inImage(new_y, new_x)) {
                    // tempImage[new_y * colsf + new_x] = img_moved[y * colsm + x];
                    // Change this to take into account of pxy and py
                    unsigned long tempx = floor((binN) * (img_fixed.valueAt(y, x, c) / 256.0f));
                    unsigned long tempy = floor((binN) * (img_moved.valueAt(new_y, new_x, c) / 256.0f));

                    test_pxy[tempx][tempy] += (1.0f);
                    test_px[tempx] += (1.0f);
                    test_py[tempy] += (1.0f);
                    count += 1;

                    i1 += img_moved.valueAt(new_y, new_x, c) / 255.0f * img_moved.valueAt(new_y, new_x, c) / 255.0f;
                    i2 += img_fixed.valueAt(y, x, c) / 255.0f * img_fixed.valueAt(y, x, c) / 255.0f;
                }
            }
        }
    }

    for(int i = 0; i < binN * binN; i++) {
        if(i < binN) {
            test_px[i] /= count;
            test_py[i] /= count;
        }
        test_pxy[i / binN][i % binN] /= count;
    }

    // Calculate MI
    
    /*
        # Convert bins counts to probability values
        pxy = hgram / float(np.sum(hgram)) binxbin
        px = np.sum(pxy, axis=1, keepdims=True) # marginal for x over y binx1
        py = np.sum(pxy, axis=0, keepdims=True) # marginal for y over x 1xbin
        #px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals 
        px_py = px * py binxbin
        # Now we can do the calculation using the pxy, px_py 2D arrays
        nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
        vector1 = pxy[nzs]
        vector2 = px_py[nzs]
        vector2 = np.log2(pxy[nzs] / px_py[nzs])
        mutual_information = np.sum(vector1 * vector2)
        return mutual_information
    */
    // Needed variables

    // Get the mutual information
    for (int x = 0; x < binN; x++) {
        for (int y = 0; y < binN; y++) {
            float tempP = test_pxy[x][y]; // numPix > 0, pxy[i] > 0
            float tempPx = test_px[x];
            float tempPy = test_py[y];
            if (tempP > 0 && tempPx > 0 && tempPy > 0) {
                float tempTemp = tempP * (log2(tempP) - (log2(tempPx) + log2(tempPy))); // px_py[i] > 0, tempP > 0
                //if (tempTemp != INFINITY && tempTemp != -INFINITY && tempTemp != NAN && tempTemp != -NAN)
                    output += tempTemp;
                    //temp += tempTemp;
                //DEBUG
                //printf("%f %f %f %f\n",tempP,temp, px_py[i], log2(tempP / px_py[i]));
            }
        }
    }
    
    // Return output
    return -1*output;
}

template<typename func_precision, typename grid_precision, uint8_t D = 8, uint8_t CHANNELS, typename pixType>
CUDAFUNCTION func_precision calcMIAlt(nv_ext::Vec<grid_precision, D> &parameters,
                                    CudaImage<pixType, CHANNELS> img_moved, CudaImage<pixType, CHANNELS> img_fixed) {
    // int binIdx = blockDim.x * blockIdx.x + threadIdx.x;
    // px = moving image (pre-calculated)
    // py = fixed image (zeroed)
    // pxy = 2d hist (zeroed)
    // TODO: Try CudaImage type for px,py,pxy
    int colsf = img_fixed.width();
    int rowsf = img_fixed.height();
    int colsm = img_moved.width();
    int rowsm = img_moved.height();
    const int binN = 64;
    // // Output holder
    func_precision output = 0;

    // printf("Starting %d\n", blockIdx.x * blockDim.x + threadIdx.x);

    float h11 = 0, h12 = 0, h13 = 0, h21 = 0, h22 = 0, h23 = 0, h31 = 0, h32 = 0, cx = (float)colsm/2, cy = (float)rowsm/2;
    parametersToHomography<float,8>(parameters, cx, cy,
        h11, h12, h13,
        h21, h22, h23,
        h31, h32);


    float test_pxy[64][64] = {{0}};
    float test_py[64] = {0};
    float test_px[64] = {0};

    float i1 = 0;
    float i2 = 0;

    // Transform the image
    int count = 0;

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
        float mag_norm = sqrt(v_x * v_x + v_y * v_y);
    // for (int x = 0; x < colsf; x++) {
        // for (int y = 0; y < rowsf; y++) {
        for (float lambda = lambda_start; lambda < lambda_end && lambda < colsf; lambda += 1) {
            float new_x = lambda * v_x + p0_x;
            float new_y = lambda * v_y + p0_y;
            // float new_x = 0, new_y = 0;
            // calcNewCoordH(h11, h12, h13,
            //     h21, h22, h23,
            //     h31, h32,
            //     x, y,
            //     new_x, new_y);
            if (inImage){
                for (int c = 0; c < CHANNELS; c++) {
                // if (img_moved.inImage(new_y, new_x)) {
                    // tempImage[new_y * colsf + new_x] = img_moved[y * colsm + x];
                    // Change this to take into account of pxy and py
                    unsigned long tempx = floor((binN) * (img_fixed.valueAt(row, lambda, c) / 256.0f));
                    unsigned long tempy = floor((binN) * (img_moved.valueAt(new_y, new_x, c) / 256.0f));

                    test_pxy[tempx][tempy] += (1.0f);
                    test_px[tempx] += (1.0f);
                    test_py[tempy] += (1.0f);
                    count += 1;

                    i1 += img_moved.valueAt(new_y, new_x, c) / 255.0f * img_moved.valueAt(new_y, new_x, c) / 255.0f;
                    i2 += img_fixed.valueAt(row, lambda, c) / 255.0f * img_fixed.valueAt(row, lambda, c) / 255.0f;
                }
                mov_area += mag_norm;
                fix_area += 1;
            }
        }
    }

    for(int i = 0; i < binN * binN; i++) {
        if(i < binN) {
            test_px[i] /= count;
            test_py[i] /= count;
        }
        test_pxy[i / binN][i % binN] /= count;
    }

    // Calculate MI
    
    /*
        # Convert bins counts to probability values
        pxy = hgram / float(np.sum(hgram)) binxbin
        px = np.sum(pxy, axis=1, keepdims=True) # marginal for x over y binx1
        py = np.sum(pxy, axis=0, keepdims=True) # marginal for y over x 1xbin
        #px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals 
        px_py = px * py binxbin
        # Now we can do the calculation using the pxy, px_py 2D arrays
        nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
        vector1 = pxy[nzs]
        vector2 = px_py[nzs]
        vector2 = np.log2(pxy[nzs] / px_py[nzs])
        mutual_information = np.sum(vector1 * vector2)
        return mutual_information
    */
    // Needed variables

    // Get the mutual information
    for (int x = 0; x < binN; x++) {
        for (int y = 0; y < binN; y++) {
            float tempP = test_pxy[x][y]; // numPix > 0, pxy[i] > 0
            float tempPx = test_px[x];
            float tempPy = test_py[y];
            if (tempP > 0 && tempPx > 0 && tempPy > 0) {
                float tempTemp = tempP * (log2(tempP) - (log2(tempPx) + log2(tempPy))); // px_py[i] > 0, tempP > 0
                //if (tempTemp != INFINITY && tempTemp != -INFINITY && tempTemp != NAN && tempTemp != -NAN)
                    output += tempTemp;
                    //temp += tempTemp;
                //DEBUG
                //printf("%f %f %f %f\n",tempP,temp, px_py[i], log2(tempP / px_py[i]));
            }
        }
    }
    
    // Return output
    
    if(mov_area / (colsm * rowsm * parameters[2]) > 0.90 && fix_area / (colsf * rowsf) > 0.50)
        return -1*output;
    else
        return (func_precision) 123123.0f;
}

#endif