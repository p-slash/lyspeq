#ifndef MATRIX_HELPER_H
#define MATRIX_HELPER_H

#include <memory>
#include <vector>

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

    // v always starts at 0, ends at N-1-abs(d)
    // A is NxN
    void getDiagonal(const double *A, int N, int d, double *v);
    inline
    void vector_add(double *target, const double *source, int size)
    {cblas_daxpy(size, 1, source, 1, target, 1);}
    inline
    void vector_sub(double *target, const double *source, int size)
    {cblas_daxpy(size, -1, source, 1, target, 1);}

    // Trace of A.B
    // Both are assumed to general and square NxN
    double trace_dgemm(const double *A, const double *B, int N);

    // Trace of A.B
    // Assumes A and B square matrices NxN, and at least one to be symmetric.
    // No stride or whatsoever. Continous allocation
    // Uses CBLAS dot product.
    inline
    double trace_dsymm(const double *A, const double *B, int N)
    {return cblas_ddot(N*N, A, 1, B, 1);}

    // Trace of A.B
    // Assumes A square matrix, B is diagonal. 
    // Only pass diagonal terms of B
    // If A is NxN, then B is N
    // Uses CBLAS dot product.
    inline
    double trace_ddiagmv(const double *A, const double *B, int N)
    {return cblas_ddot(N, A, N+1, B, 1);}

    // vT . S . v
    // Assumes S is square symmetric matrix NxN
    double my_cblas_dsymvdot(const double *v, const double *S,
        double *temp_vector, int N);

    void printfMatrix(const double *A, int N1, int N2);
    void fprintfMatrix(const char *fname, const double *A, int N1, int N2);
    std::vector<double> fscanfMatrix(const char *fname, int &N1, int &N2);

    // LAPACKE functions
    // In-place invert by first LU factorization
    void LAPACKE_InvertMatrixLU(double *A, int N);

    // Return orthogonal vector in rows of A.
    // A is assumed to have n vectors in its rows.
    // vector size is m, which is npixels in spectrum
    void LAPACKE_svd(double *A, double *svals, int m, int n);

    class DiaMatrix
    {
        std::unique_ptr<int[]> offsets;
        double *values, *sandwich_buffer;
        int size;

        double* _getDiagonal(int d);
        // void _getRowIndices(int i, int *indices);
        void _getRowIndices(int i, std::vector<int> &indices);

    public:
        int ndim, ndiags;

        DiaMatrix(int nm, int ndia);
        ~DiaMatrix();
        DiaMatrix(DiaMatrix &&rhs) = delete;
        DiaMatrix(const DiaMatrix &rhs) = delete;

        double* matrix() const { return values; };
        void getRow(int i, double *row);
        void getRow(int i, std::vector<double> &row);
        // Swap diagonals
        void transpose();
        void deconvolve(double m); // bool byCol

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
        void multiplyLeft(char TRANS_A, const double* A, double *B);

        // R . inplace . R^T
        void sandwich(double *inplace);

        double getMinMemUsage();
        double getBufMemUsage();
    };

    class OversampledMatrix
    {
        double *values, *sandwich_buffer;
        int nvals;

        double* _getRow(int i);
    public:
        int nrows, ncols;
        int nelem_per_row, oversampling;
        double fine_dlambda;

        // n1 : Number of rows.
        // nelem_prow : Number of elements per row. Should be odd.
        // osamp : Oversampling coefficient.
        // dlambda : Linear wavelength spacing of the original grid (i.e. rows)
        OversampledMatrix(int n1, int nelem_prow, int osamp, double dlambda);
        ~OversampledMatrix();
        OversampledMatrix(OversampledMatrix &&rhs) = delete;
        OversampledMatrix(const OversampledMatrix &rhs) = delete;

        double* matrix() const { return values; };
        int getNCols() const { return ncols; };

        // R . A = B
        // A should be ncols x ncols symmetric matrix. 
        // B should be nrows x ncols, will be initialized to zero
        void multiplyLeft(const double* A, double *B);
        // A . R^T = B
        // A should be nrows x ncols matrix. 
        // B should be nrows x nrows, will be initialized to zero
        void multiplyRight(const double* A, double *B);

        // B = R . Temp . R^T
        void sandwichHighRes(double *B, const double *temp_highres_mat);

        void freeBuffer();
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
        std::unique_ptr<DiaMatrix> dia_matrix;
        std::unique_ptr<OversampledMatrix> osamp_matrix;
    public:
        Resolution(int nm, int ndia);
        // n1 : Number of rows.
        // nelem_prow : Number of elements per row. Should be odd.
        // osamp : Oversampling coefficient.
        // dlambda : Linear wavelength spacing of the original grid (i.e. rows)
        Resolution(int n1, int nelem_prow, int osamp, double dlambda);
        Resolution(const Resolution *rmaster, int i1, int i2);
        Resolution(Resolution &&rhs) = default;
        Resolution(const Resolution &rhs) = delete;

        int getNCols() const { return ncols; };
        bool isDiaMatrix() const { return is_dia_matrix; };
        double* matrix() const;
        int getNElemPerRow() const;

        void cutBoundary(int i1, int i2);

        void transpose();
        void orderTranspose();
        void deconvolve(double m);
        void oversample(int osamp, double dlambda);

        // B = R . Temp . R^T
        void sandwich(double *B, const double *temp_highres_mat);

        void freeBuffer();
        double getMinMemUsage();
        double getBufMemUsage();

        void fprintfMatrix(const char *fname);
    };
}

#endif
