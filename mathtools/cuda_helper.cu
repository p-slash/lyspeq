#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <stdexcept>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

class MyCuDouble
{
    double *dev_ptr;

    void _alloc(int n) {
        cudaError_t stat = cudaMalloc((void**) &dev_ptr, n*sizeof(double));
        if (stat != cudaSuccess) {
            dev_ptr = nullptr;
            throw std::runtime_error("cudaMalloc failed.\n");
        }
    }
public:
    MyCuDouble() { dev_ptr = nullptr; }
    MyCuDouble(int n) { _alloc(n); }
    MyCuDouble(int n, double *cpu_ptr) { 
        _alloc(n); asyncCpy(cpu_ptr, n);
    }
    ~MyCuDouble() { cudaFree(dev_ptr); }

    double& operator[](int i) { return dev_ptr[i]; }
    MyCuDouble(const MyCuDouble& udev_ptr) = delete;
    MyCuDouble& operator=(const MyCuDouble& udev_ptr) = delete;

    double* get() const { return dev_ptr; }

    void asyncCpy(double *cpu_ptr, int n, int offset=0) {
        cudaMemcpyAsync(dev_ptr + offset, cpu_ptr, n*sizeof(double), cudaMemcpyHostToDevice);
    }
    void reset() {
        if (dev_ptr != nullptr) {
            cudaFree(dev_ptr);
            dev_ptr = nullptr;
        }
    }

    void realloc(int n, double *cpu_ptr=nullptr) {
        reset(); _alloc(n);
        if (cpu_ptr != nullptr)
            asyncCpy(cpu_ptr, n);
    }
};

class CuHelper
{
    // cudaError_t cudaStat;
    cublasStatus_t blas_stat;
    cusolverStatus_t solver_stat;

public:
    cublasHandle_t blas_handle;
    cusolverDnHandle_t solver_handle;

    CuHelper() {
        blas_stat = cublasCreate(&blas_handle);
        if (blas_stat != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("CUBLAS initialization failed.\n");

        solver_stat = cusolverDnCreate(&solver_handle);
        if (solver_stat != CUSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("CUSOLVER initialization failed.\n");
    };
    ~CuHelper() { cublasDestroy(blas_handle); cusolverDnDestroy(solver_handle); };

    // Trace of A.B
    // Assumes A and B square matrices NxN, and at least one to be symmetric.
    // No stride or whatsoever. Continous allocation
    // Uses BLAS dot product.
    double trace_dsymm(const double *A, const double *B, int N) {
        double result;
        blas_stat = cublasDdot(blas_handle, N*N, A, 1, B, 1, &result);
        if (blas_stat != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("trace_dsymm/cublasDdot failed.\n");
        return result;
    }

    double trace_ddiagmv(const double *A, const double *B, int N) {
        double result;
        blas_stat = cublasDdot(blas_handle, N, A, N+1, B, 1, &result);
        if (blas_stat != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("trace_ddiagmv/cublasDdot failed.\n");
        return result;
    }

    // vT . S . v
    // Assumes S is square symmetric matrix NxN
    double my_cublas_dsymvdot(const double *v, const double *S, double *temp_vector, int N) {
        dsmyv(CUBLAS_FILL_MODE_UPPER, N, 1., S, N, v, 1, 0, temp_vector, 1);
        double result;
        blas_stat = cublasDdot(blas_handle, N, v, 1, temp_vector, 1, &result);
        if (blas_stat != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("my_cublas_dsymvdot/cublasDdot failed.\n");
        return result;
    }

    void dcopy(const double *x, double *y, int N) {
        blas_stat = cublasDcopy(blas_handle, N, x, 1, y, 1);
        if (blas_stat != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasDcopy failed.\n");
    }

    void daxpy( double alpha,
                const double *x, double *y,
                int N,
                int incx=1, int incy=1) {
        blas_stat = cublasDaxpy(blas_handle, N, &alpha, x, incx, y, incy);
        if (blas_stat != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasDaxpy failed.\n");
    }
    
    void dsymm(cublasSideMode_t side, cublasFillMode_t uplo,
               int m, int n, double alpha,
               const double *A, int lda,
               const double *B, int ldb,
               double beta, double *C, int ldc) {
        blas_stat = cublasDsymm(blas_handle, side, uplo,
            m, n, &alpha,
            A, lda, B, ldb,
            &beta, C, ldc);
        if (blas_stat != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasDsymm failed.\n");
    }

    void dsmyv( cublasFillMode_t uplo,
                int n, double alpha,
                const double *A, int lda,
                const double *x, int incx, double beta,
                double *y, int incy) {
        blas_stat = cublasDsymv(blas_handle, uplo,
            n, &alpha, A, lda, x, incx,
            &beta, y, incy);
        if (blas_stat != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasDsymv failed.\n");
    }

    // In-place invert by Cholesky factorization
    void invert_cholesky(double *A, int N) {
        int lworkf = 0, lworki = 0; /* size of workspace */
        /* If devInfo = 0, the Cholesky factorization is successful.
        if devInfo = -i, the i-th parameter is wrong (not counting handle).
        if devInfo = i, the leading minor of order i is not positive definite. */
        __device__ int devInfo = -1;

        solver_stat = cusolverDnDpotrf_bufferSize(
            solver_handle, CUBLAS_FILL_MODE_UPPER,
            N, A, N, &lworkf);
        if (solver_stat != CUSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("cusolverDnDpotrf_bufferSize failed.\n");

        MyCuDouble d_work(lworkf); /* device workspace for getrf */

        solver_stat = cusolverDnDpotrf(
            solver_handle, CUBLAS_FILL_MODE_UPPER,
            N, A, N, d_work.get(), lworkf, &devInfo);
        if (solver_stat != CUSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("cusolverDnDpotrf failed.\n");
        if (devInfo != 0)
            throw std::runtime_error("Cholesky factorization is not successful.\n");

        solver_stat = cusolverDnDpotri_bufferSize(
            solver_handle, CUBLAS_FILL_MODE_UPPER,
            N, A, N, &lworki);
        if (solver_stat != CUSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("cusolverDnDpotri_bufferSize failed.\n");

        if (lworki > lworkf)
            d_work.realloc(lworki);

        solver_stat = cusolverDnDpotri(
            solver_handle, CUBLAS_FILL_MODE_UPPER,
            N, A, N, d_work.get(), lworki, &devInfo);
        if (solver_stat != CUSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("cusolverDnDpotri failed.\n");
        if (devInfo != 0)
            throw std::runtime_error("Cholesky inversion is not successful.\n");
    }

    void svd(double *A, double *svals, int m, int n) {
        int lwork = 0; /* size of workspace */
        __device__ int devInfo = -1;

        solver_stat = cusolverDnDgesvd_bufferSize(
            solver_handle, m, n, &lwork);
        if (solver_stat != CUSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("cusolverDnDgesvd_bufferSize failed.\n");

        MyCuDouble d_work(lwork); /* device workspace */

        solver_stat = cusolverDnDgesvd(
            solver_handle, 'O', 'N', m, n, A, m, svals,
            nullptr, m, nullptr, n, d_work.get(), lwork,
            nullptr, &devInfo);
        if (solver_stat != CUSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("cusolverDnDgesvd failed.\n");
        if (devInfo != 0)
            throw std::runtime_error("SVD is not successful.\n");
    }

};

#endif