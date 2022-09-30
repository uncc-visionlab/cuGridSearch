#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "cuComplex.h"
#include "inverse.h"
#include "solve.h"

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

#define A(row,col)     A[(col)*N+(row)]
#define C(row,col)     C[(col)*N+(row)]
#define T(row,col)     T[(col)*N+(row)]
#define TC(row,col)    TC[(col)*N+(row)]
#define Ainv(row,col)  Ainv[(col)*N+(row)]
#define Cinv(row,col)  Cinv[(col)*N+(row)]
#define Tinv(row,col)  Tinv[(col)*N+(row)]
#define TCinv(row,col) TCinv[(col)*N+(row)]

#define N 3
#define BATCH 3

int main (void)
{
    double A[BATCH*N*N];
    double Ainv[BATCH*N*N];
    cuDoubleComplex C[BATCH*N*N];
    cuDoubleComplex Cinv[BATCH*N*N];
    double b[BATCH*N];
    double x[BATCH*N];
    cuDoubleComplex bc[BATCH*N];
    cuDoubleComplex xc[BATCH*N];
    double *A_d;
    double *Ainv_d;
    cuDoubleComplex *C_d;
    cuDoubleComplex *Cinv_d;
    double *b_d;
    double *x_d;
    cuDoubleComplex *bc_d;
    cuDoubleComplex *xc_d;
    double *T;
    double *Tinv;
    cuDoubleComplex *TC;
    cuDoubleComplex *TCinv;
    double *tb;
    double *tx;
    cuDoubleComplex *tbc;
    cuDoubleComplex *txc;
    int i;
    
    CUDA_SAFE_CALL (cudaMalloc ((void**)&A_d, BATCH*N*N*sizeof(A_d[0])));
    CUDA_SAFE_CALL (cudaMalloc ((void**)&Ainv_d, BATCH*N*N*sizeof(Ainv_d[0])));
    CUDA_SAFE_CALL (cudaMalloc ((void**)&C_d, BATCH*N*N*sizeof(C_d[0])));
    CUDA_SAFE_CALL (cudaMalloc ((void**)&Cinv_d, BATCH*N*N*sizeof(Cinv_d[0])));
    CUDA_SAFE_CALL (cudaMalloc ((void**)&b_d, BATCH*N*sizeof(b_d[0])));
    CUDA_SAFE_CALL (cudaMalloc ((void**)&x_d, BATCH*N*sizeof(x_d[0])));
    CUDA_SAFE_CALL (cudaMalloc ((void**)&bc_d, BATCH*N*sizeof(bc_d[0])));
    CUDA_SAFE_CALL (cudaMalloc ((void**)&xc_d, BATCH*N*sizeof(xc_d[0])));

    printf ("\nNon-batched matrix inversion\n\n");

    A(0,0) = 3.0;  A(0,1) = 1.0;  A(0,2) = 1.0;
    A(1,0) = 1.0;  A(1,1) = 3.0;  A(1,2) = 1.0;
    A(2,0) = 1.0;  A(2,1) = 1.0;  A(2,2) = 3.0;

    CUDA_SAFE_CALL (cudaMemcpy (A_d, A, N*N*sizeof(A_d[0]),
                                cudaMemcpyHostToDevice));
    dmatinv(A_d, Ainv_d, N);
    CUDA_SAFE_CALL (cudaMemcpy (Ainv, Ainv_d, N*N*sizeof(Ainv_d[0]),
                                cudaMemcpyDeviceToHost));
    printf ("       % f  % f  % f             % f  % f  % f\n",
            A(0,0), A(0,1), A(0,2), Ainv(0,0), Ainv(0,1), Ainv(0,2));
    printf ("A =    % f  % f  % f   Ainv =    % f  % f  % f\n",
            A(1,0), A(1,1), A(1,2), Ainv(1,0), Ainv(1,1), Ainv(1,2));
    printf ("       % f  % f  % f             % f  % f  % f\n\n",
            A(2,0), A(2,1), A(2,2), Ainv(2,0), Ainv(2,1), Ainv(2,2));

    C(0,0) = make_cuDoubleComplex( 1.0,  0.0);
    C(0,1) = make_cuDoubleComplex( 1.0,  2.0);
    C(0,2) = make_cuDoubleComplex( 2.0, 10.0);
    C(1,0) = make_cuDoubleComplex( 1.0,  1.0);
    C(1,1) = make_cuDoubleComplex( 0.0,  3.0);
    C(1,2) = make_cuDoubleComplex(-5.0, 14.0);
    C(2,0) = make_cuDoubleComplex( 1.0,  1.0);
    C(2,1) = make_cuDoubleComplex( 0.0,  5.0);
    C(2,2) = make_cuDoubleComplex(-8.0, 20.0);

    CUDA_SAFE_CALL (cudaMemcpy (C_d, C, N*N*sizeof(C_d[0]),
                                cudaMemcpyHostToDevice));
    zmatinv(C_d, Cinv_d, N);
    CUDA_SAFE_CALL (cudaMemcpy (Cinv, Cinv_d, N*N*sizeof(Cinv_d[0]),
                                cudaMemcpyDeviceToHost));

    printf ("        % 3.0f%+4.0fi   % 3.0f%+4.0fi   % 3.0f%+4.0fi              % 3.0f%+4.0fi   % 3.0f%+4.0fi   % 3.0f%+4.0fi\n",
            C(0,0).x, C(0,0).y, C(0,1).x, C(0,1).y, C(0,2).x, C(0,2).y, 
            Cinv(0,0).x, Cinv(0,0).y, Cinv(0,1).x, Cinv(0,1).y, Cinv(0,2).x, Cinv(0,2).y);
    printf ("A =     % 3.0f%+4.0fi   % 3.0f%+4.0fi   % 3.0f%+4.0fi   Ainv =     % 3.0f%+4.0fi   % 3.0f%+4.0fi   % 3.0f%+4.0fi\n",
            C(1,0).x, C(1,0).y, C(1,1).x, C(1,1).y, C(1,2).x, C(1,2).y, 
            Cinv(1,0).x, Cinv(1,0).y, Cinv(1,1).x, Cinv(1,1).y, Cinv(1,2).x, Cinv(1,2).y);
    printf ("        % 3.0f%+4.0fi   % 3.0f%+4.0fi   % 3.0f%+4.0fi              % 3.0f%+4.0fi   % 3.0f%+4.0fi   % 3.0f%+4.0fi\n",
            C(2,0).x, C(2,0).y, C(2,1).x, C(2,1).y, C(2,2).x, C(2,2).y, 
            Cinv(2,0).x, Cinv(2,0).y, Cinv(2,1).x, Cinv(2,1).y, Cinv(2,2).x, Cinv(2,2).y);

    printf ("\n\nBatched matrix inversion\n\n");

    T = A + 0*N*N;
    T(0,0) =   1.0;  T(0,1) = 1./2.;  T(0,2) = 1./3.;
    T(1,0) = 1./2.;  T(1,1) =   1.0;  T(1,2) = 2./3.;
    T(2,0) = 1./3.;  T(2,1) = 2./3.;  T(2,2) =   1.0;
    T = A + 1*N*N;
    T(0,0) =   3.0;  T(0,1) =   1.0;  T(0,2) =   1.0;
    T(1,0) =   1.0;  T(1,1) =   3.0;  T(1,2) =   1.0;
    T(2,0) =   1.0;  T(2,1) =   1.0;  T(2,2) =   3.0;
    T = A + 2*N*N;
    T(0,0) =  -3.0;  T(0,1) =   1.0;  T(0,2) =   0.0;
    T(1,0) =   1.0;  T(1,1) =  -2.0;  T(1,2) =   1.0;
    T(2,0) =   0.0;  T(2,1) =   1.0;  T(2,2) =  -1.0;

    CUDA_SAFE_CALL (cudaMemcpy (A_d, A, BATCH*N*N*sizeof(A_d[0]),
                                cudaMemcpyHostToDevice));
    dmatinv_batch (A_d, Ainv_d, N, BATCH);
    CUDA_SAFE_CALL (cudaMemcpy (Ainv, Ainv_d, BATCH*N*N*sizeof(Ainv_d[0]),
                                cudaMemcpyDeviceToHost));
    for (i = 0; i < BATCH; i++) {
        T    = A    + i*N*N;
        Tinv = Ainv + i*N*N;
        printf ("       % f  % f  % f             % f  % f  % f\n",
                T(0,0), T(0,1), T(0,2), Tinv(0,0), Tinv(0,1), Tinv(0,2));
        printf ("A(%d) = % f  % f  % f   Ainv(%d) = % f  % f  % f\n",
                i, T(1,0), T(1,1), T(1,2), i, Tinv(1,0), Tinv(1,1), Tinv(1,2));
        printf ("       % f  % f  % f             % f  % f  % f\n",
                T(2,0), T(2,1), T(2,2), Tinv(2,0), Tinv(2,1), Tinv(2,2));
        printf ("\n");
    }        

    TC = C + 0*N*N;
    TC(0,0) = make_cuDoubleComplex(-1.0,  0.0);
    TC(0,1) = make_cuDoubleComplex( 1.0, -2.0);
    TC(0,2) = make_cuDoubleComplex( 0.0,  0.0);
    TC(1,0) = make_cuDoubleComplex( 1.0,  2.0);
    TC(1,1) = make_cuDoubleComplex( 0.0,  0.0);
    TC(1,2) = make_cuDoubleComplex( 0.0, -1.0);
    TC(2,0) = make_cuDoubleComplex( 0.0,  0.0);
    TC(2,1) = make_cuDoubleComplex( 0.0,  1.0);
    TC(2,2) = make_cuDoubleComplex( 1.0,  0.0);
    TC = C + 1*N*N;
    TC(0,0) = make_cuDoubleComplex( 1.0,  0.0);
    TC(0,1) = make_cuDoubleComplex( 1.0,  1.0);
    TC(0,2) = make_cuDoubleComplex( 0.0,  2.0);
    TC(1,0) = make_cuDoubleComplex( 1.0, -1.0);
    TC(1,1) = make_cuDoubleComplex( 5.0,  0.0);
    TC(1,2) = make_cuDoubleComplex(-3.0,  0.0);
    TC(2,0) = make_cuDoubleComplex( 0.0, -2.0);
    TC(2,1) = make_cuDoubleComplex(-3.0,  0.0);
    TC(2,2) = make_cuDoubleComplex( 0.0,  0.0);
    TC = C + 2*N*N;
    TC(0,0) = make_cuDoubleComplex( 1.0,  0.0);
    TC(0,1) = make_cuDoubleComplex( 1.0,  2.0);
    TC(0,2) = make_cuDoubleComplex( 2.0, 10.0);
    TC(1,0) = make_cuDoubleComplex( 1.0,  1.0);
    TC(1,1) = make_cuDoubleComplex( 0.0,  3.0);
    TC(1,2) = make_cuDoubleComplex(-5.0, 14.0);
    TC(2,0) = make_cuDoubleComplex( 1.0,  1.0);
    TC(2,1) = make_cuDoubleComplex( 0.0,  5.0);
    TC(2,2) = make_cuDoubleComplex(-8.0, 20.0);
    
    CUDA_SAFE_CALL (cudaMemcpy (C_d, C, BATCH*N*N*sizeof(C_d[0]),
                                cudaMemcpyHostToDevice));
    zmatinv_batch (C_d, Cinv_d, N, BATCH);
    CUDA_SAFE_CALL (cudaMemcpy (Cinv, Cinv_d, BATCH*N*N*sizeof(Cinv_d[0]),
                                cudaMemcpyDeviceToHost));

    for (i = 0; i < BATCH; i++) {
        TC    = C    + i*N*N;
        TCinv = Cinv + i*N*N;
        printf ("        % 3.0f%+4.0fi   % 3.0f%+4.0fi   % 3.0f%+4.0fi            % 7.3f%+4.3fi  % 6.3f%+4.3fi  % 6.3f%+6.3fi\n",
                TC(0,0).x, TC(0,0).y, TC(0,1).x, 
                TC(0,1).y, TC(0,2).x, TC(0,2).y, 
                TCinv(0,0).x, TCinv(0,0).y, TCinv(0,1).x, 
                TCinv(0,1).y, TCinv(0,2).x, TCinv(0,2).y);
        printf ("A(%d) =  % 3.0f%+4.0fi   % 3.0f%+4.0fi   % 3.0f%+4.0fi   Ainv(%d) =% 7.3f%+4.3fi  % 6.3f%+4.3fi  % 6.3f%+6.3fi\n",
                i, TC(1,0).x, TC(1,0).y, TC(1,1).x,
                TC(1,1).y, TC(1,2).x, TC(1,2).y, 
                i, TCinv(1,0).x, TCinv(1,0).y, TCinv(1,1).x, 
                TCinv(1,1).y, TCinv(1,2).x, TCinv(1,2).y);
        printf ("        % 3.0f%+4.0fi   % 3.0f%+4.0fi   % 3.0f%+4.0fi            % 7.3f%+4.3fi  % 6.3f%+4.3fi  % 6.3f%+6.3fi\n",
                TC(2,0).x, TC(2,0).y, TC(2,1).x,
                TC(2,1).y, TC(2,2).x, TC(2,2).y, 
                TCinv(2,0).x, TCinv(2,0).y, TCinv(2,1).x,
                TCinv(2,1).y, TCinv(2,2).x, TCinv(2,2).y);
        printf ("\n");
    }        

    printf ("\nBatched solver (single right-hand side)\n\n");

    T = A + 0*N*N;
    tb = b + 0*N;
    T(0,0) = 0.0;  T(0,1) = 4.0;  T(0,2) = 1.0;  tb[0] =  9.0;
    T(1,0) = 1.0;  T(1,1) = 1.0;  T(1,2) = 3.0;  tb[1] =  6.0;
    T(2,0) = 2.0;  T(2,1) =-2.0;  T(2,2) = 1.0;  tb[2] = -1.0;
    T = A + 1*N*N;
    tb = b + 1*N;
    T(0,0) = 1.0;  T(0,1) = 0.5;  T(0,2) = 0.2;  tb[0] =  4.0;
    T(1,0) = 0.5;  T(1,1) = 1.0;  T(1,2) = 0.5;  tb[1] = -1.0;
    T(2,0) = 0.2;  T(2,1) = 0.5;  T(2,2) = 1.0;  tb[2] =  3.0;
    T = A + 2*N*N;
    tb = b + 2*N;
    T(0,0) = 2.0;  T(0,1) = 0.0;  T(0,2) = 0.0;  tb[0] =  6.0;
    T(1,0) = 1.0;  T(1,1) = 5.0;  T(1,2) = 0.0;  tb[1] =  2.0;
    T(2,0) = 7.0;  T(2,1) = 9.0;  T(2,2) = 8.0;  tb[2] =  5.0;
    
    CUDA_SAFE_CALL (cudaMemcpy (A_d, A, BATCH*N*N*sizeof(A_d[0]),
                                cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL (cudaMemcpy (b_d, b, BATCH*N*sizeof(b_d[0]),
                                cudaMemcpyHostToDevice));
    dsolve_batch (A_d, b_d, x_d, N, BATCH);
    CUDA_SAFE_CALL (cudaMemcpy (x, x_d, BATCH*N*sizeof(x_d[0]),
                                cudaMemcpyDeviceToHost));
    
    for (i = 0; i < BATCH; i++) {
        T  = A + i*N*N;
        tb = b + i*N;
        tx = x + i*N;
        printf ("       % f  % f  % f             % f             % f\n",
                T(0,0), T(0,1), T(0,2), tb[0], tx[0]);
        printf ("A(%d) = % f  % f  % f      b(%d) = % f      x(%d) = % f\n",
                i, T(1,0), T(1,1), T(1,2), i, tb[1], i, tx[1]);
        printf ("       % f  % f  % f             % f             % f\n",
                T(2,0), T(2,1), T(2,2), tb[2], tx[2]);
        printf ("\n");
    }        

    TC = C + 0*N*N;
    tbc = bc + 0*N;
    TC(0,0) = make_cuDoubleComplex(-2.0,  3.0);
    TC(0,1) = make_cuDoubleComplex( 1.0,  0.0);
    TC(0,2) = make_cuDoubleComplex( 0.0,  2.0);
    TC(1,0) = make_cuDoubleComplex( 0.0, -5.0);
    TC(1,1) = make_cuDoubleComplex( 3.0,  0.0);
    TC(1,2) = make_cuDoubleComplex( 1.0, -1.0);
    TC(2,0) = make_cuDoubleComplex( 3.0, -2.0);
    TC(2,1) = make_cuDoubleComplex(-2.0,  0.0);
    TC(2,2) = make_cuDoubleComplex( 2.0,  0.0);
    tbc[0]  = make_cuDoubleComplex(-2.0,  5.0);
    tbc[1]  = make_cuDoubleComplex(11.0,  0.0);
    tbc[2]  = make_cuDoubleComplex( 6.0,  1.0);
    TC = C + 1*N*N;
    tbc = bc + 1*N;
    TC(0,0) = make_cuDoubleComplex( 1.0, -5.0);
    TC(0,1) = make_cuDoubleComplex(-3.0,  0.0);
    TC(0,2) = make_cuDoubleComplex( 1.0, -1.0);
    TC(1,0) = make_cuDoubleComplex( 2.0,  0.0);
    TC(1,1) = make_cuDoubleComplex( 0.0,  1.0);
    TC(1,2) = make_cuDoubleComplex( 5.0,  0.0);
    TC(2,0) = make_cuDoubleComplex( 1.0,  0.0);
    TC(2,1) = make_cuDoubleComplex(-3.0,  2.0);
    TC(2,2) = make_cuDoubleComplex(-1.0,  0.0);
    tbc[0]  = make_cuDoubleComplex(15.0,-14.0);
    tbc[1]  = make_cuDoubleComplex( 6.0, -2.0);
    tbc[2]  = make_cuDoubleComplex( 6.0, +1.0);
    TC = C + 2*N*N;
    tbc = bc + 2*N;
    TC(0,0) = make_cuDoubleComplex( 5.0, -1.0);
    TC(0,1) = make_cuDoubleComplex( 0.0,  3.0);
    TC(0,2) = make_cuDoubleComplex( 1.0,  0.0);
    TC(1,0) = make_cuDoubleComplex( 0.0,  1.0);
    TC(1,1) = make_cuDoubleComplex( 2.0,  1.0);
    TC(1,2) = make_cuDoubleComplex(-4.0,  0.0);
    TC(2,0) = make_cuDoubleComplex( 2.0, -3.0);
    TC(2,1) = make_cuDoubleComplex( 1.0,  0.0);
    TC(2,2) = make_cuDoubleComplex( 3.0,  0.0);
    tbc[0]  = make_cuDoubleComplex( 8.0,  5.0);
    tbc[1]  = make_cuDoubleComplex(-8.0,  3.0);
    tbc[2]  = make_cuDoubleComplex(13.0, -3.0);


    CUDA_SAFE_CALL (cudaMemcpy (C_d, C, BATCH*N*N*sizeof(C_d[0]),
                                cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL (cudaMemcpy (bc_d, bc, BATCH*N*sizeof(bc_d[0]),
                                cudaMemcpyHostToDevice));
    zsolve_batch (C_d, bc_d, xc_d, N, BATCH);
    CUDA_SAFE_CALL (cudaMemcpy (xc, xc_d, BATCH*N*sizeof(xc_d[0]),
                                cudaMemcpyDeviceToHost));

    for (i = 0; i < BATCH; i++) {
        TC = C + i*N*N;
        tbc = bc + i*N;
        txc = xc + i*N;
        printf ("        % 3.0f%+4.0fi   % 3.0f%+4.0fi   % 3.0f%+4.0fi              % 3.0f%+4.0fi              % 3.0f%+4.0fi\n",
                TC(0,0).x, TC(0,0).y, TC(0,1).x, 
                TC(0,1).y, TC(0,2).x, TC(0,2).y, 
                tbc[0].x, tbc[0].y, txc[0].x, txc[0].y);
        printf ("A(%d) =  % 3.0f%+4.0fi   % 3.0f%+4.0fi   % 3.0f%+4.0fi      b(%d) =  % 3.0f%+4.0fi      x(%d) =  % 3.0f%+4.0fi\n",
                i, TC(1,0).x, TC(1,0).y, TC(1,1).x, 
                TC(1,1).y, TC(1,2).x, TC(1,2).y, 
                i, tbc[1].x, tbc[1].y, i, txc[1].x, txc[1].y);
        printf ("        % 3.0f%+4.0fi   % 3.0f%+4.0fi   % 3.0f%+4.0fi              % 3.0f%+4.0fi              % 3.0f%+4.0fi\n",
                TC(2,0).x, TC(2,0).y, TC(2,1).x, 
                TC(2,1).y, TC(2,2).x, TC(2,2).y, 
                tbc[2].x, tbc[2].y, txc[2].x, txc[2].y);
        printf("\n");
    }
            
    CUDA_SAFE_CALL(cudaFree(A_d));
    CUDA_SAFE_CALL(cudaFree(Ainv_d));
    CUDA_SAFE_CALL(cudaFree(C_d));
    CUDA_SAFE_CALL(cudaFree(Cinv_d));
    CUDA_SAFE_CALL(cudaFree(b_d));
    CUDA_SAFE_CALL(cudaFree(x_d));

    return EXIT_SUCCESS;
}
