#ifndef CUDAERRORFUNCTIONS_MI_CUH
#define CUDAERRORFUNCTIONS_MI_CUH

#include <nvVectorNd.h>
#include "cudaTensor.cuh"

/*
Function to multiply to matrices
INPUTS:
    Nx1 float* A
    1xN float* B
    NxN float* C - Output
    int N
*/
CUDAFUNCTION void matMul(float *A, float *B, float *C, int size) {
    // Assuming A is sizex1 and B is 1xsize
    // Multiply the values and insert them into C
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i * size + j] = A[i] * B[j];
        }
    }
}

/*
Function to calculate mutualinformation from a NxN bin 2d histogram
INPUTS:
    NxN float* pxy (2dhist)
    int N
OUTPUTS:
    float MI
*/
CUDAFUNCTION float mutualInformation(unsigned int *pxy, int cols, int rows, int binN) {
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
    float temp = 0.0f;
    float *px = new float[binN];
    float *py = new float[binN];
    float *px_py = new float[binN * binN];
    float numPix = rows * cols;

    // Collect probabilities in x and y
    for (int i = 0; i < binN * binN; i++) {
        float tempP = pxy[i] / numPix;
        //DEBUG
        //if(pxy[i] > 0) printf("%d %f %d %d\n",pxy[i], tempP, i/binN, i%binN);
        px[i % binN] += tempP;
        py[i / binN] += tempP;
    }

    // Get the combined probabilities
    matMul(px, py, px_py, binN);

    // Get the mutual information
    for (int i = 0; i < binN * binN; i++) {
        float tempP = pxy[i] / numPix; // numPix > 0, pxy[i] > 0
        if (pxy[i] > 0) {
            float tempTemp = tempP * log2(tempP / px_py[i]); // px_py[i] > 0, tempP > 0
            if (tempTemp != INFINITY)
                temp += tempTemp;
            //DEBUG
            //printf("%f %f %f %f\n",tempP,temp, px_py[i], log2(tempP / px_py[i]));
        }
    }

    // Free memory
    delete[] px;
    delete[] py;
    delete[] px_py;

    // Return output
    return temp;
}

/*
Function to calculate the 2d histogram of 2 OpenCV GPU Mats
Assumes the images are the same size
INPUTS:
    NxN float* hist2d - Output
    rxc uchar* image1
    rxc uchar* image2
    int cols
    int rows
    int binN Number of bins
*/
CUDAFUNCTION void
histogram2d(unsigned int *hist2d, unsigned char *image1, unsigned char *image2, int cols, int rows, int binN) {
    // Image data is in a Image[y * cols + x] format
    for (int x = 0; x < cols; x++) {
        for (int y = 0; y < rows; y++) {
            unsigned long temp = (image1[y * cols + x] / (256 / binN)) * binN + (image2[y * cols + x] / (256 / binN));
            if (temp < binN * binN)
                hist2d[temp] += 1;
        }
    }
}

/*
Function that calls the other MI collection functions
INPUTS:
    rxc uchar* image1
    rxc uchar* image2
    int cols
    int rows
    int N Number of bins
OUTPUT:
    float MI
*/
CUDAFUNCTION float calcMI(unsigned char *image1, unsigned char *image2, int cols, int rows, int binN) {
    // Create NxN bin 2D histogram
    unsigned int *hist2d = new unsigned int[binN * binN];

    // Populate 2D histogram
    histogram2d(hist2d, image1, image2, cols, rows, binN);

    // Calculate MI
    float output = 0;
    output = mutualInformation(hist2d, cols, rows, binN);

    // Free Memory
    delete[] hist2d;

    return output;
}

// TODO: Go simpler than starting with H 
template<typename func_precision, typename grid_precision, unsigned int D = 8, typename pixType>
CUDAFUNCTION func_precision
grid_mi(nv_ext::Vec<grid_precision, D> &H,
        unsigned char *img_moved, unsigned char *img_fixed, int colsm, int rowsm, int colsf, int rowsf) {

    // Create blank temp image for after perspective transform
    unsigned char *tempImage = new unsigned char[colsf * rowsf];
    if (tempImage == NULL) {
        delete[] tempImage;
        printf("Out of heap memory!\n");
        return 1;
    }
    for (int i = 0; i < colsf * rowsf; i++) {
        tempImage[i] = 0;
    }

    // // Output holder
    func_precision output = 0;

    // In case H < 4 //Similarity transform
    float cpH[4] = {0.0, 0.0, 0.0, 1.0};
    for (int i = 0; i < D; i++) {
        cpH[i] = H[i];
    }

    // Transform the image
    float ct = cos(H[2]);
    float st = sin(H[2]);
    for (int x = 0; x < colsm; x++) {
        for (int y = 0; y < rowsm; y++) {
            int new_x = (int) (cpH[3] * (x - colsm / 2) * ct - (y - rowsm / 2) * st + cpH[0] + cpH[3] * colsm / 2);
            int new_y = (int) ((x - colsm / 2) * st + cpH[3] * (y - rowsm / 2) * ct + cpH[1] + cpH[3] * rowsm / 2);
            if ((new_x >= 0 && new_x < colsf) && (new_y >= 0 && new_y < rowsf))
                tempImage[new_y * colsf + new_x] = img_moved[y * colsm + x];
        }
    }

    // Calculate MI
    output = calcMI(img_fixed, tempImage, colsf, rowsf, 256);
    //output = calcNCC(img_fixed, tempImage, cols, rows);
    //output = -1*calcSQD(img_fixed, tempImage, cols, rows);

    // Free memory
    delete[] tempImage;
    // Return output
    return -1 * output;
}

#endif