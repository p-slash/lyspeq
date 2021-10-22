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

    // The resolution matrix is assumed to have consecutive row elements.
    // Do not skip pixels when using resolution matrix
    class Resolution
    {
        // int *indices, *iptrs;
        int nrows, ncols, nvals;//, nptrs;
        int nelem_per_row, oversampling;
        double fine_dlambda, *sandwich_buffer;

        double* _getRow(int i);
    public:
        double *values, *temp_highres_mat;

        // n1 : Number of rows.
        // nelem_prow : Number of elements per row. Should be odd.
        // osamp : Oversampling coefficient.
        // dlambda : Linear wavelength spacing of the original grid (i.e. rows)
        Resolution(int n1, int nelem_prow, int osamp, double dlambda);
        ~Resolution();

        int getNCols() const { return ncols; };

        // R . A fine_dlambda = B
        // A should be ncols x ncols symmetric matrix. 
        // B should be nrows x ncols, will be initialized to zero
        void multiplyLeft(const double* A, double *B);
        // A . R^T fine_dlambda = B
        // A should be nrows x ncols matrix. 
        // B should be nrows x nrows, will be initialized to zero
        void multiplyRight(const double* A, double *B);

        // Manually create and set temp_highres_mat
        void allocateTempHighRes();
        double *allocWaveGrid(double w1);
        // B = R . Temp . R^T fine_dlambda^2
        void sandwichHighRes(double *B);

        void freeBuffers();
        double getMinMemUsage();
        double getBufMemUsage();

        void fprintfMatrix(const char *fname);
    };
}

#endif
