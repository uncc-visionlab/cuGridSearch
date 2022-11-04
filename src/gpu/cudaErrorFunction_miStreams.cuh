#ifndef CUDAERRORFUNCTIONS_MI_STREAMS_CUH
#define CUDAERRORFUNCTIONS_MI_STREAMS_CUH

#include <nvVectorNd.h>
#include "cudaTensor.cuh"

#define THREADS_PER_BLOCK 1024

template<typename func_precision, typename grid_precision, uint32_t D, uint32_t CHANNELS, typename pixType>
__device__ func_precision calcMIstreamAlt(nv_ext::Vec<grid_precision, D> &H,
                        CudaImage<pixType, CHANNELS> img_moved, CudaImage<pixType, CHANNELS> img_fixed) {
    int binIdx = threadIdx.x;
    int y = threadIdx.x;
    // px = moving image (pre-calculated)
    // py = fixed image (zeroed)
    // pxy = 2d hist (zeroed)
    int colsf = img_fixed.width();
    int rowsf = img_fixed.height();
    int colsm = img_moved.width();
    int rowsm = img_moved.height();
    int binN = 64;

    // // Output holder
    func_precision output = 0;
   __shared__ float test_pxy[64][64];
   __shared__ float test_py[64];
   __shared__ float test_px[64];

   if (binIdx < 64) {
        test_px[binIdx] = 0;
        test_py[binIdx] = 0;
        for (int i = 0; i < 64; i++) {
            test_pxy[binIdx][i] = 0;
        }
   }

    __syncthreads();

    float h11 = 0, h12 = 0, h13 = 0, h21 = 0, h22 = 0, h23 = 0, h31 = 0, h32 = 0, cx = (float)colsm/2, cy = (float)rowsm/2;
    parametersToHomography<grid_precision,D>(H, cx, cy,
        h11, h12, h13,
        h21, h22, h23,
        h31, h32);

    
    __shared__ float i1[THREADS_PER_BLOCK];
    __shared__ float i2[THREADS_PER_BLOCK];
    __shared__ int counter[THREADS_PER_BLOCK];
    __shared__ float mov_area[THREADS_PER_BLOCK];
    __shared__ float fix_area[THREADS_PER_BLOCK];
    __shared__ int total_counter;
    counter[y] = 0;
    i1[y] = 0;
    i2[y] = 0;
    if (threadIdx.x == 0)
        total_counter = 0;

    /*
        Thought process time:
        The setup is already done beforehand.
        What needs to be done --
            Call the assignparametricvalue function (need to move it to cudaImageFunctions).
            Currently the old implementation is setup to use the rows so each thread needs to work on each lambda search space.
            Once done converting from x space to lambda space, perform the normal calculation at the end.
        That's it?
        Apparently.
        We'll see at the end.
    */

    float lambda_start = 0;
    float lambda_end = colsf;
    float v_x = 0;
    float v_y = 0;
    float p0_x = 0;
    float p0_y = 0;
    bool inImage = false;

    parametricAssignValues(h11, h12, h13, h21, h22, h23, h31, h32,
                        colsm, rowsm, colsf, rowsf, y,
                        lambda_start, lambda_end, v_x, v_y, p0_x, p0_y, inImage);

    float mag_norm = sqrt(v_x * v_x + v_y * v_y);

    //for (int x = 0; x < colsf; x++) {
    for (float lambda = lambda_start; lambda < lambda_end && lambda < colsf; lambda += 1) {
        //for (int y = 0; y < rowsm; y++) {
        // float new_x = 0, new_y = 0;
        // calcNewCoordH(h11, h12, h13,
        //     h21, h22, h23,
        //     h31, h32,
        //     x, y,
        //     new_x, new_y);
        float new_x = lambda * v_x + p0_x;
        float new_y = lambda * v_y + p0_y;
        if (inImage){
            for (int c = 0; c < CHANNELS; c++) {
            // if (img_moved.inImage(new_y, new_x)) {
                counter[y] += 1;
                //if ((new_x >= 0 && new_x < colsf) && (new_y >= 0 && new_y < rowsf))
                // tempImage[new_y * colsf + new_x] = img_moved[y * colsm + x];
                // Change this to take into account of pxy and py
    //                int temp = (img_moved.valueAt(y, x, 0) / (256 / binN)) * binN + (img_fixed.valueAt(new_y, new_x, 0) / (256 / binN));
                int binY = (int) floor(binN * (img_moved.valueAt(new_y, new_x, c) / 256.0f));
                int binX = (int) floor(binN * (img_fixed.valueAt(y, lambda, c) / 256.0f));
                atomicAdd(&test_pxy[binX][binY], 1.0f);
                atomicAdd(&test_py[binY], 1.0f);
                atomicAdd(&test_px[binX], 1.0f);
                i1[y] += img_fixed.valueAt(y,lambda,c) / 255.0f * img_fixed.valueAt(y,lambda,c) / 255.0f;
                i2[y] += img_moved.valueAt(new_y,new_x,c) / 255.0f * img_moved.valueAt(new_y,new_x,c) / 255.0f;
                mov_area[y] += mag_norm;
                fix_area[y] += 1;
                
//
//                pxy.at<pixType>(binY, binX) += 1.0f;
//                if (temp < binN * binN)
//                    atomicAdd(&test_pxy[temp], (1.0f / (colsm * rowsm)));
//                unsigned long temp2 = img_fixed.valueAt(new_y, new_x, 0)/ (256 / binN);
//                atomicAdd(&test_py[temp2], (1.0f / (colsm * rowsm)));
            }
        }
    }

    // Makes sure that all of the threads are together before calculating MI
    __syncthreads();

    float i1t = 0, i2t = 0;
    float mov_areat = 0, fix_areat = 0;
    if (threadIdx.x == 0) {
        for(int i = 0; i < rowsf; i++) {
            total_counter += counter[i];
            i1t += i1[i];
            i2t += i2[i];
            mov_areat += mov_area[i];
            fix_areat += fix_area[i];
        }
    }

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
    __shared__ float rowMutualInformation[64];
    float Hx = 0;
    float Hy = 0;
    if (binIdx < binN && binIdx < 64) {
        rowMutualInformation[binIdx] = 0;

        // Get the mutual information
        for (int x = 0; x < binN; x++) {
//            float tempPxy = test_pxy[binIdx * binN + x]; // numPix > 0, pxy[i] > 0
            float tempPxy = test_pxy[binIdx][x] / total_counter;
            float tempPy = test_py[x] / total_counter;
            float tempPx = test_px[binIdx] / total_counter;
            if (tempPxy > 0 && tempPx > 0 && tempPy > 0) {  // px_py[i] > 0, tempPxy > 0
                float tempTemp = (tempPxy) * (log2(tempPxy) - (log2(tempPx) + log2(tempPy)));
//                if (tempTemp != INFINITY && tempTemp != -INFINITY && tempTemp != NAN && tempTemp != -NAN)
                rowMutualInformation[binIdx] += tempTemp;
                //temp += tempTemp;
                //DEBUG
                //printf("%f %f %f %f\n",tempPxy,temp, px_py[i], log2(tempPxy / px_py[i]));
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < binN; i++) {
            output += rowMutualInformation[i];
            float tempPx = test_px[i] / total_counter;
            float tempPy = test_py[i] / total_counter;
            if (tempPx > 0)
                Hx -= tempPx * log2(tempPx);
            if (tempPy > 0)
                Hy -= tempPy * log2(tempPy);
        }
    }

   __syncthreads();

    // Return output
    if (threadIdx.x == 0) {
        if(mov_areat / (colsm * rowsm * H[2]) > 0.25 && fix_areat / (colsf * rowsf) > 0.35) {
            output = -1 * 2 * output / (Hx + Hy);
        } else {
            output = 123123.0f;
        }
        // printf("output = %f\n", output);
        // printf("mov_areat = %f / %f (%d), fix_areat = %f/ (%f,%f * %f,%f) (%d)\n",mov_areat, (colsm * rowsm * H[2]), (mov_areat / (colsm * rowsm * H[2])) > 0.25, fix_areat, colsf, img_fixed.height(), rowsf, img_fixed.width(), (fix_areat / (colsf * rowsf)) > 0.35);
        return output;
    }
    else
        return 0;
}

template<typename func_precision, typename grid_precision, uint32_t D, uint32_t CHANNELS, typename pixType>
__device__ func_precision calcMIstream(nv_ext::Vec<grid_precision, D> &H,
                        CudaImage<pixType, CHANNELS> img_moved, CudaImage<pixType, CHANNELS> img_fixed) {
    int binIdx = threadIdx.x;
    int y = blockDim.x * blockIdx.x + threadIdx.x;
    // px = moving image (pre-calculated)
    // py = fixed image (zeroed)
    // pxy = 2d hist (zeroed)
    int colsf = img_fixed.width();
    int rowsf = img_fixed.height();
    int colsm = img_moved.width();
    int rowsm = img_moved.height();
    int binN = 64;

    // // Output holder
    func_precision output = 0;
   __shared__ float test_pxy[64][64];
   __shared__ float test_py[64];
   __shared__ float test_px[64];

   if (binIdx < 64) {
        test_px[binIdx] = 0;
        test_py[binIdx] = 0;
        for (int i = 0; i < 64; i++) {
            test_pxy[binIdx][i] = 0;
        }
   }

    __syncthreads();

    float h11 = 0, h12 = 0, h13 = 0, h21 = 0, h22 = 0, h23 = 0, h31 = 0, h32 = 0, cx = (float)colsm/2, cy = (float)rowsm/2;
    parametersToHomography<grid_precision,D>(H, cx, cy,
        h11, h12, h13,
        h21, h22, h23,
        h31, h32);

    
    __shared__ float i1[THREADS_PER_BLOCK];
    __shared__ float i2[THREADS_PER_BLOCK];
    __shared__ int counter[THREADS_PER_BLOCK];
    __shared__ int total_counter;
    counter[y] = 0;
    i1[y] = 0;
    i2[y] = 0;
    if (threadIdx.x == 0)
        total_counter = 0;

    for (int x = 0; x < colsf; x++) {
        //for (int y = 0; y < rowsm; y++) {
        float new_x = 0, new_y = 0;
        calcNewCoordH(h11, h12, h13,
            h21, h22, h23,
            h31, h32,
            x, y,
            new_x, new_y);
        for (int c = 0; c < CHANNELS; c++) {
            if (img_moved.inImage(new_y, new_x)) {
                counter[y] += 1;
                //if ((new_x >= 0 && new_x < colsf) && (new_y >= 0 && new_y < rowsf))
                // tempImage[new_y * colsf + new_x] = img_moved[y * colsm + x];
                // Change this to take into account of pxy and py
    //                int temp = (img_moved.valueAt(y, x, 0) / (256 / binN)) * binN + (img_fixed.valueAt(new_y, new_x, 0) / (256 / binN));
                int binY = (int) floor(binN * (img_moved.valueAt(new_y, new_x, c) / 256.0f));
                int binX = (int) floor(binN * (img_fixed.valueAt(y, x, c) / 256.0f));
                atomicAdd(&test_pxy[binX][binY], 1.0f);
                atomicAdd(&test_py[binY], 1.0f);
                atomicAdd(&test_px[binX], 1.0f);
                i1[y] += img_fixed.valueAt(y,x,c) / 255.0f * img_fixed.valueAt(y,x,c) / 255.0f;
                i2[y] += img_moved.valueAt(y,x,c) / 255.0f * img_moved.valueAt(y,x,c) / 255.0f;
                
//
//                pxy.at<pixType>(binY, binX) += 1.0f;
//                if (temp < binN * binN)
//                    atomicAdd(&test_pxy[temp], (1.0f / (colsm * rowsm)));
//                unsigned long temp2 = img_fixed.valueAt(new_y, new_x, 0)/ (256 / binN);
//                atomicAdd(&test_py[temp2], (1.0f / (colsm * rowsm)));
            }
        }
    }

    // Makes sure that all of the threads are together before calculating MI
    __syncthreads();

    float i1t = 0, i2t = 0;
    if (threadIdx.x == 0) {
        for(int i = 0; i < rowsf; i++) {
            total_counter += counter[i];
            i1t += i1[i];
            i2t += i2[i];
        }
    }

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
    __shared__ float rowMutualInformation[64];
    float Hx = 0;
    float Hy = 0;
    if (binIdx < binN && binIdx < 64) {
        rowMutualInformation[binIdx] = 0;

        // Get the mutual information
        for (int x = 0; x < binN; x++) {
//            float tempPxy = test_pxy[binIdx * binN + x]; // numPix > 0, pxy[i] > 0
            float tempPxy = test_pxy[binIdx][x] / total_counter;
            float tempPy = test_py[x] / total_counter;
            float tempPx = test_px[binIdx] / total_counter;
            if (tempPxy > 0 && tempPx > 0 && tempPy > 0) {  // px_py[i] > 0, tempPxy > 0
                float tempTemp = (tempPxy) * (log2(tempPxy) - (log2(tempPx) + log2(tempPy)));
//                if (tempTemp != INFINITY && tempTemp != -INFINITY && tempTemp != NAN && tempTemp != -NAN)
                rowMutualInformation[binIdx] += tempTemp;
                //temp += tempTemp;
                //DEBUG
                //printf("%f %f %f %f\n",tempPxy,temp, px_py[i], log2(tempPxy / px_py[i]));
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < binN; i++) {
            output += rowMutualInformation[i];
            float tempPx = test_px[i] / total_counter;
            float tempPy = test_py[i] / total_counter;
            if (tempPx > 0)
                Hx -= tempPx * log2(tempPx);
            if (tempPy > 0)
                Hy -= tempPy * log2(tempPy);
        }
    }

//    __syncthreads();

    // Return output
    if (threadIdx.x == 0)
        return -1 * 2 * output / (Hx + Hy);
    else
        return 0;
}

#endif