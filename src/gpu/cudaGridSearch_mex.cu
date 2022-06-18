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

/* system header */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

/* MEX header */
#include <mex.h> 
#include "matrix.h"

/* nVIDIA CUDA header */
#include <cuda.h> 

/* fixing error : identifier "IUnknown" is undefined" */
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

//#define CUDAFUNCTION
#define CUDAFUNCTION __host__ __device__

/* includes CUDA kernel */
#include "cudaGridSearchKernels.cuh"

/* MEX entry function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
 {
    double *A, *B, *C;
    mwSignedIndex Am, An, Bm, Bn;

    /* argument check */
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:cudaAdd:inputmismatch",
                "Input arguments must be 2!");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:cudaAdd:outputmismatch",
                "Output arguments must be 1!");
    }

    A = mxGetPr(prhs[0]);
    B = mxGetPr(prhs[1]);
    mexPrintf_800("%f\n",A[0]);
    mexPrintf_800("%f\n",B[1]);
    /* matrix size */
    Am = (mwSignedIndex) mxGetM(prhs[0]);
    An = (mwSignedIndex) mxGetN(prhs[0]);
    Bm = (mwSignedIndex) mxGetM(prhs[1]);
    Bn = (mwSignedIndex) mxGetN(prhs[1]);
    if (Am != Bm || An != Bn) {
        mexErrMsgIdAndTxt("MATLAB:cudaAdd:sizemismatch",
                "Input matrices must have the same size!");
    }

    /* allocate output */
    plhs[0] = mxCreateDoubleMatrix(Am, An, mxREAL);
    C = mxGetPr(plhs[0]);

    /* set GPU grid & block configuration */
    cudaDeviceProp deviceProp;
    memset(&deviceProp, 0, sizeof (deviceProp));
    if (cudaSuccess != cudaGetDeviceProperties(&deviceProp, 0)) {
        mexPrintf_800("\n%s", cudaGetErrorString(cudaGetLastError()));
        return;
    }

    dim3 dimGridImg(8, 1, 1);
    dim3 dimBlockImg(1, 64, 1);

    /* allocate device memory for matrices */
    double *d_A = NULL;
    cudaMalloc((void**) &d_A, Am * An * sizeof (double));
    cudaMemcpy(d_A, A, Am * An * sizeof (double), cudaMemcpyHostToDevice);
    double *d_B = NULL;
    cudaMalloc((void**) &d_B, Bm * Bn * sizeof (double));
    cudaMemcpy(d_B, B, Bm * Bn * sizeof (double), cudaMemcpyHostToDevice);
    double *d_C = NULL;
    cudaMalloc((void**) &d_C, Am * An * sizeof (double));

    /* call GPU kernel for addition */
    gpuAddKernel << < dimGridImg, dimBlockImg >>>(d_A, d_B, d_C, Am, An);
    cudaDeviceSynchronize();

    /* copy result from device */
    cudaMemcpy(C, d_C, Am * An * sizeof (double), cudaMemcpyDeviceToHost);

    /* free GPU memory */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}
