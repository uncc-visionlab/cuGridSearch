/*
 * Copyright (c) 2011-2013 NVIDIA Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 *
 *   Redistributions of source code must retain the above copyright notice, 
 *   this list of conditions and the following disclaimer.
 *
 *   Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 *   Neither the name of NVIDIA Corporation nor the names of its contributors
 *   may be used to endorse or promote products derived from this software 
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#if !defined(INVERSE_H_)
#define INVERSE_H_
#include "cuComplex.h"

#ifdef __cplusplus
extern "C" {
#endif

/* smatinv_batch() inverts one or many square, non-singular matrices of single-
   precision elements. Partial pivoting is employed in the inversion process 
   for increased numerical stability.

   A     pointer to an array of the single-precision matrices to be inverted, 
         where each matrix is stored in column-major order
   Ainv  pointer to an array of the single-precision matrices which receive
         the inverses of the corresponding matrices pointed to by A, where 
         each matrix is stored in column-major order
   n     number of rows and columns of the matrices in the arrays pointed to 
         by A and Ainv. n must be greater than, or equal to 2. On sm_13 GPUs,
         n must be less than, or equal to, 62. On sm_2x and sm_3x GPUs, n must
         be less than, or equal to, 109.
   batch the number of matrices to be inverted. It must be greater than zero.

   Returns:

    0    operation completed successfully
   -1    n is out of bounds, batch is out of bounds
   -2    a CUDA error occured
*/
int smatinv_batch(float *A, float *Ainv, int n, int batch);

/* dmatinv_batch() inverts one or many square, non-singular matrices of double-
   precision elements. Partial pivoting is employed in the inversion process 
   for increased numerical stability.

   A     pointer to an array of the double-precision matrices to be inverted, 
         where each matrix is stored in column-major order
   Ainv  pointer to an array of the double-precision matrices which receive
         the inverses of the corresponding matrices pointed to by A, where 
         each matrix is stored in column-major order
   n     number of rows and columns of the matrices in the arrays pointed to 
         by A and Ainv. n must be greater than, or equal to 2. On sm_13 GPUs,
         n must be less than, or equal to, 44. On sm_2x and sm_3x GPUs, n must
         be less than, or equal to, 77.
   batch the number of matrices to be inverted. It must be greater than zero.

   Returns:

    0    operation completed successfully
   -1    n is out of bounds, batch is out of bounds
   -2    a CUDA error occured
*/
int dmatinv_batch(double *A, double *Ainv, int n, int batch);

/* cmatinv_batch() inverts one or many square, non-singular matrices of float-
   complex elements. Partial pivoting is employed in the inversion process for
   increased numerical stability.

   A     pointer to an array of the float-complex matrices to be inverted, 
         where each matrix is stored in column-major order
   Ainv  pointer to an array of the float-complex matrices which receive
         the inverses of the corresponding matrices pointed to by A, where 
         each matrix is stored in column-major order
   n     number of rows and columns of the matrices in the arrays pointed to 
         by A and Ainv. n must be greater than, or equal to, 2. On sm_13 GPUs,
         n must be less than, or equal to, 44. On sm_2x and sm_3x GPUs, n must
         be less than, or equal to, 77.
   batch the number of matrices to be inverted. It must be greater than zero.

   Returns:

    0    operation completed successfully
   -1    n is out of bounds, batch is out of bounds
   -2    a CUDA error occured
*/
int cmatinv_batch(cuComplex *A, cuComplex *Ainv, int n, int batch);

/* zmatinv_batch() inverts one or many square, non-singular matrices of double-
   complex elements. Partial pivoting is employed in the inversion process for
   increased numerical stability.

   A     pointer to an array of the double-complex matrices to be inverted, 
         where each matrix is stored in column-major order
   Ainv  pointer to an array of the double-complex matrices which receive
         the inverses of the corresponding matrices pointed to by A, where 
         each matrix is stored in column-major order
   n     number of rows and columns of the matrices in the arrays pointed to 
         by A and Ainv. n must be greater than, or equal to, 2. On sm_13 GPUs,
         n must be less than, or equal to, 31. On sm_2x and sm_3x GPUs, n must
         be less than, or equal to, 55.
   batch the number of matrices to be inverted. It must be greater than zero.

   Returns:

    0    operation completed successfully
   -1    n is out of bounds, batch is out of bounds
   -2    a CUDA error occured
*/
int zmatinv_batch(cuDoubleComplex *A, cuDoubleComplex *Ainv, int n, int batch);

#ifdef __cplusplus
}
#endif

#endif /* INVERSE_H_ */
