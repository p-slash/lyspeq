#ifndef MATRIX_HELPER_H
#define MATRIX_HELPER_H

#include <memory>
#include <vector>

#ifdef USE_MKL_CBLAS
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#else
#include "cblas.h"
// These three lines somehow fix OpenBLAS compilation error on macos
// #include <complex.h>
// #define lapack_complex_float    float _Complex
// #define lapack_complex_double   double _Complex
#include "lapacke.h"
#endif

#include "core/omp_manager.hpp"

namespace glmemory {
    extern double* getSandwichBuffer(int size);
}

namespace mxhelp
{
    // Copy upper triangle of matrix A to its lower triangle
    // A is NxN
    void copyUpperToLower(double *A, int N);

    void transpose_copy(const double *A, double *B, int M, int N);

    // v always starts at 0, ends at N-1-abs(d)
    // A is NxN
    void getDiagonal(const double *A, int N, int d, double *v);

    inline void vector_multiply(
            int N, const double *x, int incx, const double *y, int incy,
            double *out
    ) {
        for (int i = 0; i < N; ++i)
            out[i] = x[i * incx] * y[i * incy];
    }

    inline void vector_multiply(
            int N, const double *x, const double *y, double *out
    ) {
        for (int i = 0; i < N; ++i)
            out[i] = x[i] * y[i];
    }

    inline void normalize_vector(int N, double *x) {
        double norm = 1.0 / cblas_dnrm2(N, x, 1);
        cblas_dscal(N, norm, x, 1);
    }

    // Trace of A.B
    // Both are assumed to general and square NxN
    double trace_dgemm(const double *A, const double *B, int N);

    // Trace of A.B
    // Assumes A square matrix, B is diagonal. 
    // Only pass diagonal terms of B
    // If A is NxN, then B is N
    // Uses CBLAS dot product.
    inline
    double trace_ddiagmv(const double *A, const double *B, int N)
    {return cblas_ddot(N, A, N+1, B, 1);}

    // vT . A . v
    double my_cblas_dsymvdot(
        const double *v, const double *S,
        double *temp_vector, int N);
    double my_cblas_dgemvdot(
        const double *x, int Nx, const double* y, int Ny,
        const double *A, double *temp_vector);

    void printfMatrix(const double *A, int N1, int N2);
    void fprintfMatrix(const char *fname, const double *A, int N1, int N2);
    std::vector<double> fscanfMatrix(const char *fname, int &N1, int &N2);

    // LAPACKE functions
    // In-place invert by first LU factorization
    void LAPACKE_InvertMatrixLU(double *A, int N);
    void LAPACKE_InvertMatrixCholesky(double *U, int N);
    void LAPACKE_InvertSymMatrixLU_damped(double *S, int N, double damp);

    void LAPACKE_sym_eigens(double *A, int N, double *evals, double *evecs);
    void LAPACKE_sym_posdef_sqrt(double *A, int N, double *evals, double *evecs);
    // Return condition number
    // if sjump != nullptr, finds the adjacent ratio of s values larger than 8
    // fromthe right side
    double LAPACKE_RcondSvd(
            const double *A, int N, double *sjump=nullptr
    );

    // Replace zero diagonals with one, then invert
    // Return new number of degrees of freedom
    int LAPACKE_InvertMatrixLU_safe(double *A, int N);
    // S is symmetric
    // returns warning code (1 if damping used). DOF and damp as well
    // S is damped solution and Sinv is direct inverse on return
    int stableInvertSym(
        std::unique_ptr<double[]> &S, std::unique_ptr<double[]> &Sinv,
        int N, int &dof, double &damp);

    void LAPACKE_safeSolveCho(double *S, int N, double *b);
    // S is symmetric. Only upper addressed
    void LAPACKE_stableSymSolve(double *S, int N, double *b);

    // Return orthogonal vector in rows of A.
    // A is assumed to have n vectors in its rows.
    // vector size is m, which is npixels in spectrum
    void LAPACKE_svd(double *A, double *svals, int m, int n);

    class DiaMatrix
    {
        std::unique_ptr<int[]> offsets;
        double *values;
        int size;

        double* _getDiagonal(int d);
        // void _getRowIndices(int i, int *indices);
        void _getRowIndices(int i, std::vector<int> &indices);

    public:
        int ndim, ndiags;

        DiaMatrix(int nm, int ndia);
        ~DiaMatrix() { delete [] values; };
        DiaMatrix(DiaMatrix &&rhs) = delete;
        DiaMatrix(const DiaMatrix &rhs) = delete;

        double* matrix() const { return values; };
        int getSize() const { return size; };

        void getRow(int i, double *row);
        void getRow(int i, std::vector<double> &row);
        // Swap diagonals
        void transpose();
        void deconvolve(double m); // bool byCol

        void constructGaussian(double *v, double R_kms, double a_kms);
        void fprintfMatrix(const char *fname);
        void orderTranspose();

        // B initialized to zero
        // SIDE = CblasLeft,    B = op( R ) . A,
        // SIDE = CblasRight,   B = A . op( R ).
        // TRANSR = CblasNoTrans,  op( R ) = R.
        // TRANSR = CblasTrans,    op( R ) = R^T.
        void multiply(
            CBLAS_SIDE SIDER, CBLAS_TRANSPOSE TRANSR,
            const double* A, double *B);
        void multiplyLeft(const double* A, double *B, int M=0);
        void multiplyRightT(const double* A, double *B, int M=0);

        // R . inplace . R^T
        void sandwich(double *inplace);

        double getMinMemUsage();
        double getBufMemUsage() { return (double)sizeof(double) * ndim * ndim / 1048576.; };
    };

    class OversampledMatrix
    {
        double *values;
        int size;

        double* _getRow(int i) { return values + i * nelem_per_row; };
    public:
        int nrows, ncols;
        int nelem_per_row, oversampling;
        double fine_dlambda;

        // n1 : Number of rows.
        // nelem_prow : Number of elements per row. Should be odd.
        // osamp : Oversampling coefficient.
        // dlambda : Linear wavelength spacing of the original grid (i.e. rows)
        OversampledMatrix(int n1, int nelem_prow, int osamp, double dlambda);
        ~OversampledMatrix() { delete [] values; };
        OversampledMatrix(OversampledMatrix &&rhs) = delete;
        OversampledMatrix(const OversampledMatrix &rhs) = delete;

        double* matrix() const { return values; };
        int getNCols() const { return ncols; };
        int getSize() const { return size; };

        // R . A = B
        // A should be ncols x ncols symmetric matrix. 
        // B should be nrows x ncols, will be initialized to zero
        void multiplyLeft(const double* A, double *B);
        // A . R^T = B
        // A should be nrows x ncols matrix. 
        // B should be nrows x nrows, will be initialized to zero
        // Assumes B will be symmetric
        void multiplyRight(const double* A, double *B);

        // B = R . S . R^T, S is symmetric
        void sandwich(const double *S, double *B);

        double getMinMemUsage() { return (double)sizeof(double) * size / 1048576.; };
        double getBufMemUsage() {
            return (double)sizeof(double) * nrows * myomp::getMaxNumThreads() / 1048576.;
        };

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

        DiaMatrix* getDiaMatrixPointer() const { return dia_matrix.get(); };

        int getNCols() const { return ncols; };
        bool isDiaMatrix() const { return is_dia_matrix; };
        double* matrix() const {
            if (is_dia_matrix) return dia_matrix->matrix();
            else               return osamp_matrix->matrix();
        }
        int getSize() const {
            if (is_dia_matrix) return dia_matrix->getSize();
            else               return osamp_matrix->getSize();
        }
        int getNElemPerRow() const {
            if (is_dia_matrix) return dia_matrix->ndiags;
            else               return osamp_matrix->nelem_per_row;
        }

        void cutBoundary(int i1, int i2);

        void transpose() { if (is_dia_matrix) dia_matrix->transpose(); };
        void orderTranspose() {  if (is_dia_matrix) dia_matrix->orderTranspose(); };
        void deconvolve(double m) { if (is_dia_matrix) dia_matrix->deconvolve(m); };
        void oversample(int osamp, double dlambda);

        // B = R . S . R^T, S is symmetric
        void sandwich(const double *S, double *B);

        double getMinMemUsage();
        double getBufMemUsage();

        void fprintfMatrix(const char *fname);
    };
}

#endif
