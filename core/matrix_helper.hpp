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

    class DiaMatrix
    {
        double* _getDiagonal(int d);
        double* sandwich_buffer;
    public:
        int ndim, ndiags, size;
        int *offsets;
        double *matrix;

        DiaMatrix(int nm, int ndia);
        ~DiaMatrix();

        void getRow(int i, double *row);
        // Swap diagonals
        void transpose();

        void freeBuffer();
        void constructGaussian(double *v, double R_kms, double a_kms);
        void fprintfMatrix(const char *fname);
        void orderTranspose();

        // B initialized to zero
        // SIDE = 'L' or 'l',   B = op( R ) . A,
        // SIDE = 'R' or 'r',   B = A . op( R ).
        // TRANSR = 'N' or 'n',  op( R ) = R.
        // TRANSR = 'T' or 't',  op( R ) = R^T.
        void multiply(char SIDER, char TRANSR, const double* A, double *B);

        // R . inplace . R^T
        void sandwich(double *inplace);

        double getMinMemUsage();
        double getBufMemUsage();
    };

    class OversampledMatrix
    {
        double *sandwich_buffer;

        double* _getRow(int i);
    public:
        int nrows, ncols, nvals;
        int nelem_per_row, oversampling;
        double fine_dlambda, *values, *temp_highres_mat;

        // n1 : Number of rows.
        // nelem_prow : Number of elements per row. Should be odd.
        // osamp : Oversampling coefficient.
        // dlambda : Linear wavelength spacing of the original grid (i.e. rows)
        OversampledMatrix(int n1, int nelem_prow, int osamp, double dlambda);
        ~OversampledMatrix();

        int getNCols() const { return ncols; };

        // R . A = B
        // A should be ncols x ncols symmetric matrix. 
        // B should be nrows x ncols, will be initialized to zero
        void multiplyLeft(const double* A, double *B);
        // A . R^T = B
        // A should be nrows x ncols matrix. 
        // B should be nrows x nrows, will be initialized to zero
        void multiplyRight(const double* A, double *B);

        // Manually create and set temp_highres_mat
        void allocateTempHighRes();
        double* allocWaveGrid(double w1);
        // B = R . Temp . R^T
        void sandwichHighRes(double *B);

        void freeBuffers();
        double getMinMemUsage();
        double getBufMemUsage();

        void fprintfMatrix(const char *fname);
    };

    // The resolution matrix is assumed to have consecutive row elements.
    // Do not skip pixels when using resolution matrix.
    // Assumes each row is properly normalized and scaled by the pixel size.
    class Resolution
    {
        bool is_dia_matrix;
        int ncols;
        DiaMatrix *dia_matrix;
        OversampledMatrix *osamp_matrix;
    public:
        double *values, *temp_highres_mat;

        Resolution(int nm, int ndia);
        // n1 : Number of rows.
        // nelem_prow : Number of elements per row. Should be odd.
        // osamp : Oversampling coefficient.
        // dlambda : Linear wavelength spacing of the original grid (i.e. rows)
        Resolution(int n1, int nelem_prow, int osamp, double dlambda);
        ~Resolution();

        int getNCols() const { return ncols; };
        bool isDiaMatrix() const { return is_dia_matrix; };
        void cutBoundary(int i1, int i2);

        void orderTranspose();
        void oversample(int osamp, double dlambda);

        // Manually create and set temp_highres_mat
        void allocateTempHighRes();
        double* allocWaveGrid(double w1);
        // B = R . Temp . R^T
        void sandwich(double *B);

        void freeBuffers();
        double getMinMemUsage();
        double getBufMemUsage();

        void fprintfMatrix(const char *fname);
    };
}

#endif
