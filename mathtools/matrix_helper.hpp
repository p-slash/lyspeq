#ifndef MATRIX_HELPER_H
#define MATRIX_HELPER_H

#include <memory>
#include <vector>
#include <stdexcept>

#ifdef USE_MKL_CBLAS
#include "mkl_cblas.h"
#else
#include "cblas.h"
#endif

#include "cuda_runtime.h"
#include "cublas_v2.h"

// namespace cudaspace {
//     template<class T>
//     T* myCudaMalloc(int N) {
//         cudaError_t cudaStat;
//         T* devPtr;
//         cudaStat = cudaMalloc((void**)&devPtr, N*sizeof(T));
//         if (cudaStat != cudaSuccess)
//             throw std::runtime_error("cudaMalloc failed.\n");
//         return devPtr;
//     }

//     template<class T>
//     void myCudaFree(T* devPtr) { cudaFree(devPtr); }

//     template<class T[]>
//     auto make_unique_cuda<T[]>(int N) {
//         std::unique_ptr<T*, decltype(&myCudaFree)> uptr(myCudaMalloc(N), &myCudaFree);
//         return std::move(uptr);
//     }
// }

class CuHelper
{
    // cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

public:
    CuHelper() {
        stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("CUBLAS initialization failed.\n");
    };
    ~CuHelper() { cublasDestroy(handle); };

    // Trace of A.B
    // Assumes A and B square matrices NxN, and at least one to be symmetric.
    // No stride or whatsoever. Continous allocation
    // Uses BLAS dot product.
    double trace_dsymm(const double *A, const double *B, int N) {
        double result;
        stat = cublasDdot(handle, N, A, 1, B, 1, &result);
        if (stat != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("trace_dsymm/cublasDdot failed.\n");
        return result;
    }

    double trace_ddiagmv(const double *A, const double *B, int N) {
        double result;
        stat = cublasDdot(handle, N, A, N+1, B, 1, &result);
        if (stat != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("trace_ddiagmv/cublasDdot failed.\n");
        return result;
    }

    // vT . S . v
    // Assumes S is square symmetric matrix NxN
    double my_cublas_dsymvdot(const double *v, const double *S, double *temp_vector, int N) {
        dsmyv(CUBLAS_FILL_MODE_UPPER, N, 1., S, N, v, 1, 0, temp_vector, 1);
        double result;
        stat = cublasDdot(handle, N, v, 1, temp_vector, 1, &result);
        if (stat != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("my_cublas_dsymvdot/cublasDdot failed.\n");
        return result;
    }

    void dcopy(const double *x, double *y, int N) {
        stat = cublasDcopy(handle, N, x, 1, y, 1);
        if (stat != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasDcopy failed.\n");
    }

    void daxpy( const double *alpha,
                const double *x, double *y,
                int N,
                int incx=1, int incy=1) {
        stat = cublasDaxpy(handle, N, alpha, x, incx, y, incy);
        if (stat != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasDaxpy failed.\n");
    }
    
    void dsymm(cublasSideMode_t side, cublasFillMode_t uplo,
               int m, int n, double alpha,
               const double *A, int lda,
               const double *B, int ldb,
               double beta, double *C, int ldc) {
        stat = cublasDsymm(handle, side, uplo,
            m, n, &alpha,
            A, lda, B, ldb,
            &beta, C, ldc);
        if (stat != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasDsymm failed.\n");
    }

    void dsmyv( cublasFillMode_t uplo,
                int n, double alpha,
                const double *A, int lda,
                const double *x, int incx, double beta,
                double *y, int incy) {
        stat = cublasDsymv(handle, uplo,
            n, &alpha, A, lda, x, incx,
            &beta, y, incy);
        if (stat != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasDsymv failed.\n");
    }
};

extern std::unique_ptr<CuHelper> cuhelper;

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
