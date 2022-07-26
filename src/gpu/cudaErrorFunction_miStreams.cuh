#ifndef CUDAERRORFUNCTIONS_MI_STREAMS_CUH
#define CUDAERRORFUNCTIONS_MI_STREAMS_CUH

#include <nvVectorNd.h>
#include "cudaTensor.cuh"

#define THREADS_PER_BLOCK 1024

template<typename func_precision, typename grid_precision, uint8_t D = 8, uint8_t CHANNELS, typename pixType>
__device__ func_precision calcMIstream(nv_ext::Vec<grid_precision, D> &H,
                                    CudaImage<pixType, CHANNELS> img_moved, CudaImage<pixType, CHANNELS> img_fixed,
                                    CudaImage<float, 1> px, CudaImage<float, 1> py, CudaImage<float, 1> pxy) {
    int binIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int y  = binIdx;
    // px = moving image (pre-calculated)
    // py = fixed image (zeroed)
    // pxy = 2d hist (zeroed)
    int colsf = img_fixed.width();
    int rowsf = img_fixed.height();
    int colsm = img_moved.width();
    int rowsm = img_moved.height();
    int binN = px.height();

    // // Output holder
    func_precision output = 0;
    __shared__ float test_pxy[64*64];
    __shared__ float test_py[64];

    if(binIdx < 64) {
        test_py[binIdx] = 0;
        for(int i = 0; i < 64; i++) {
            test_pxy[binIdx * 64 + i] = 0;
        }
    }

    __syncthreads();

    // In case H < 4 //Similarity transform
    float cpH[4] = {0.0, 0.0, 0.0, 1.0};
    for (int i = 0; i < D; i++) {
        cpH[i] = H[i];
    }

    // Transform the image
    float ct = cos(H[2]);
    float st = sin(H[2]);
    for (int x = 0; x < colsm; x++) {
        //for (int y = 0; y < rowsm; y++) {
            float new_x = (cpH[3] * (x - colsm / 2) * ct - (y - rowsm / 2) * st + cpH[0] + cpH[3] * colsm / 2);
            float new_y = ((x - colsm / 2) * st + cpH[3] * (y - rowsm / 2) * ct + cpH[1] + cpH[3] * rowsm / 2);
            //if ((new_x >= 0 && new_x < colsf) && (new_y >= 0 && new_y < rowsf))
                // tempImage[new_y * colsf + new_x] = img_moved[y * colsm + x];
                // Change this to take into account of pxy and py
                unsigned long temp = (img_moved.valueAt(y, x, 0) / (256 / binN)) * binN + (img_fixed.valueAt(new_y, new_x, 0) / (256 / binN));
                if (temp < binN * binN)
                    atomicAdd(&test_pxy[temp], (1.0f / (colsm * rowsm)));
                unsigned long temp2 = img_fixed.valueAt(new_y, new_x, 0)/ (256 / binN);
                atomicAdd(&test_py[temp2], (1.0f / (colsm * rowsm)));
        //}
    }

    // Makes sure that all of the threads are together before calculating MI
    __syncthreads();

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
    __shared__ float errorTemp[64];
    if (binIdx < binN && binIdx < 64) {
        errorTemp[binIdx] = 0;

        // Get the mutual information
        for (int i = 0; i < binN; i++) {
            float tempP = test_pxy[binIdx * binN + i]; // numPix > 0, pxy[i] > 0
            if (tempP > 0 && px.at(binIdx, 0, 0) > 0 && test_py[i] > 0) {
                float tempTemp = tempP * (log2(tempP) - (log2(px.at(binIdx, 0, 0)) + log2(test_py[i]))); // px_py[i] > 0, tempP > 0
                if (tempTemp != INFINITY && tempTemp != -INFINITY && tempTemp != NAN && tempTemp != -NAN)
                    errorTemp[binIdx] += tempTemp;
                    //temp += tempTemp;
                //DEBUG
                //printf("%f %f %f %f\n",tempP,temp, px_py[i], log2(tempP / px_py[i]));
            }
        }
    }

    __syncthreads();

    if(binIdx == 0) {
        for (int i = 0; i < binN; i++)
            output += errorTemp[i];
    }

    __syncthreads();

    // Return output
    if(binIdx == 0)
        return -1*output;
    else
        return 123456123;
}

#endif