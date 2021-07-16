#ifndef MATRIX_HELPER_H
#define MATRIX_HELPER_H

#ifdef USE_MKL_CBLAS
#include "mkl_cblas.h"
#else
#include "cblas.h"
#endif

namespace mxhelp
{
    // Copy upper triangle of matrix A to its lower triangle
    // A is NxN
    void copyUpperToLower(double *A, int N);

    void vector_add(double *target, const double *source, int size);
    void vector_sub(double *target, const double *source, int size);

    // Trace of A.B
    // Both are assumed to general and square NxN
    double trace_dgemm(const double *A, const double *B, int N);

    // Trace of A.B
    // Assumes A and B square matrices NxN, and at least one to be symmetric.
    // No stride or whatsoever. Continous allocation
    // Uses CBLAS dot product.
    double trace_dsymm(const double *A, const double *B, int N);

    // Trace of A.B
    // Assumes A square matrix, B is diagonal. 
    // Only pass diagonal terms of B
    // If A is NxN, then B is N
    // Uses CBLAS dot product.
    double trace_ddiagmv(const double *A, const double *B, int N);

    // vT . S . v
    // Assumes S is square symmetric matrix NxN
    double my_cblas_dsymvdot(const double *v, const double *S, int N);

    void printfMatrix(const double *A, int N1, int N2);
    void fprintfMatrix(const char *fname, const double *A, int N1, int N2);
    void fscanfMatrix(const char *fname, double *& A, int &N1, int &N2);

    // LAPACKE functions
    // In-place invert by first LU factorization
    void LAPACKE_InvertMatrixLU(double *A, int N);

    class Resolution
    {
        double* _getDiagonal(int d);
        double* sandwich_buffer;
    public:
        int ndim, ndiags, size;
        int *offsets;
        double *matrix;

        Resolution(int nm, int ndia);
        ~Resolution();
        void freeBuffer();
        void constructGaussian(double *v, double R_kms, double a_kms);
        void fprintfMatrix(const char *fname);
        void orderTranspose();

        // B initialized to zero
        // SIDE = 'L' or 'l',   B = op( R ) . A,
        // SIDE = 'R' or 'r',   B = A . op( R ).
        // TRANSR = 'N' or 'n',  op( R ) = R.
        // TRANSR = 'T' or 't',  op( R ) = R^T.
        void multiply(int N, char SIDER, char TRANSR, const double* A, double *B);

        // R . inplace . R^T
        void sandwich(int N, double *inplace);

        double getMinMemUsage();
        double getBufMemUsage();
    };
}

#endif
