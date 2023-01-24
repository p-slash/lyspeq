#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <stdexcept>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>


template<typename T>
class MyCuPtr
{
    T *dev_ptr;

    void _alloc(int n) {
        cudaError_t stat = cudaMalloc((void**) &dev_ptr, n*sizeof(T));
        if (stat != cudaSuccess) {
            dev_ptr = nullptr;
            throw std::runtime_error("cudaMalloc failed.");
        }
    }
public:
    MyCuPtr() { dev_ptr = nullptr; }
    MyCuPtr(int n) { _alloc(n); }
    MyCuPtr(int n, T *cpu_ptr, cudaStream_t stream=NULL) { 
        _alloc(n); asyncCpy(cpu_ptr, n, 0, stream);
    }
    ~MyCuPtr() { cudaFree(dev_ptr); }

    T& operator[](int i) { return dev_ptr[i]; }
    MyCuPtr(const MyCuPtr& udev_ptr) = delete;
    MyCuPtr& operator=(const MyCuPtr& udev_ptr) = delete;

    T* get() const { return dev_ptr; }

    void asyncCpy(T *cpu_ptr, int n, int offset=0, cudaStream_t stream=NULL) {
        cudaMemcpyAsync(
            dev_ptr + offset, cpu_ptr, n*sizeof(T), cudaMemcpyHostToDevice, stream);
    }
    void asyncDwn(T *cpu_ptr, int n, int offset=0, cudaStream_t stream=NULL) {
        cudaMemcpyAsync(
            cpu_ptr, dev_ptr + offset, sizeof(T) * n, cudaMemcpyDeviceToHost, stream);
    }
    void syncDownload(T *cpu_ptr, int n, int offset=0) {
        cudaMemcpy(
            cpu_ptr, dev_ptr + offset, n*sizeof(T), cudaMemcpyDeviceToHost);
    }
    void reset() {
        if (dev_ptr != nullptr) {
            cudaFree(dev_ptr);
            dev_ptr = nullptr;
        }
    }

    void realloc(int n, T *cpu_ptr=nullptr, cudaStream_t stream=NULL) {
        reset(); _alloc(n);
        if (cpu_ptr != nullptr)
            asyncCpy(cpu_ptr, n, 0, stream);
    }
};


class MyCuStream {
    cudaStream_t stream;
    cudaError_t cuda_stat;

    void check_cuda_error(std::string err_msg) {
        if (cuda_stat != cudaSuccess) {
            throw std::runtime_error(err_msg);
        }
    }
public:
    MyCuStream() {
        cuda_stat = cudaStreamCreateWithFlags(
            &stream, cudaStreamNonBlocking);
        check_cuda_error("cudaStreamCreateWithFlags: ");
    }
    ~MyCuStream() { cudaStreamDestroy(stream); }

    cudaStream_t get() const {
        return stream;
    }

    void setCuBLAS(cublasHandle_t blas_handle) {
        cuda_stat = cublasSetStream(blas_handle, stream);
        check_cuda_error("cublasSetStream: ");
    }

    void setCuSOLVER(cusolverDnHandle_t solver_handle) {
        cuda_stat = cusolverDnSetStream(solver_handle, stream);
        check_cuda_error("cusolverDnSetStream: ");
    }

    void sync() {
        cuda_stat = cudaStreamSynchronize(stream);
        check_cuda_error("cudaStreamSynchronize: ");
    }
}


class CuHelper
{
    // cudaError_t cudaStat;
    cublasStatus_t blas_stat;
    cusolverStatus_t solver_stat;

    void check_cublas_error(std::string err_msg) {
        if (blas_stat != CUBLAS_STATUS_SUCCESS) {
            err_msg += std::string(cublasGetStatusString(blas_stat));
            throw std::runtime_error(err_msg);
        }
    }

    void check_cusolver_error(std::string err_msg) {
        if (solver_stat == CUSOLVER_STATUS_SUCCESS)
            return;
        switch (solver_stat) {
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            err_msg += "The library was not initialized.";  break;
        case CUSOLVER_STATUS_INVALID_VALUE:
            err_msg += "Invalid parameters were passed.";   break;
        case CUSOLVER_STATUS_INTERNAL_ERROR:
            err_msg += "An internal operation failed.";     break;
        }

        throw std::runtime_error(err_msg);
    }

public:
    cublasHandle_t blas_handle;
    cusolverDnHandle_t solver_handle;

    CuHelper() {
        blas_stat = cublasCreate(&blas_handle);
        check_cublas_error("CUBLAS initialization failed: ");

        solver_stat = cusolverDnCreate(&solver_handle);
        check_cusolver_error("CUSOLVER initialization failed: ");
    };
    ~CuHelper() {
        cublasDestroy(blas_handle);
        cusolverDnDestroy(solver_handle);
        cudaDeviceReset();
    };

    // Trace of A.B
    // Assumes A and B square matrices NxN, and at least one to be symmetric.
    // No stride or whatsoever. Continous allocation
    // Uses BLAS dot product.
    void trace_dsymm(const double *A, const double *B, int N, double *c_res) {
        blas_stat = cublasDdot(blas_handle, N*N, A, 1, B, 1, c_res);
        check_cublas_error("trace_dsymm/cublasDdot: ");
    }

    void trace_ddiagmv(
            const double *A, const double *B, int N, double *c_res) {
        blas_stat = cublasDdot(blas_handle, N, A, N+1, B, 1, c_res);
        check_cublas_error("trace_ddiagmv/cublasDdot: ");
    }

    // vT . S . v
    // Assumes S is square symmetric matrix NxN
    void my_cublas_dsymvdot(
            const double *v, const double *S, double *temp_vector, int N,
            double *c_res,
            const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER) {
        dsmyv(uplo, N, 1., S, N, v, 1, 0, temp_vector, 1);
        blas_stat = cublasDdot(blas_handle, N, v, 1, temp_vector, 1, c_res);
        check_cublas_error("my_cublas_dsymvdot/cublasDdot: ");
    }

    void dcopy(const double *x, double *y, int N) {
        blas_stat = cublasDcopy(blas_handle, N, x, 1, y, 1);
        check_cublas_error("cublasDcopy: ");
    }

    void daxpy( 
            double alpha,
            const double *x, double *y,
            int N,
            int incx=1, int incy=1) {
        blas_stat = cublasDaxpy(blas_handle, N, &alpha, x, incx, y, incy);
        check_cublas_error("cublasDaxpy: ");
    }

    void dsymm(
            cublasSideMode_t side, cublasFillMode_t uplo,
            int m, int n, double alpha,
            const double *A, int lda,
            const double *B, int ldb,
            double beta, double *C, int ldc) {
        blas_stat = cublasDsymm(
            blas_handle, side, uplo,
            m, n, &alpha,
            A, lda, B, ldb,
            &beta, C, ldc);
        check_cublas_error("cublasDsymm: ");
    }

    void dsmyv(
            cublasFillMode_t uplo,
            int n, double alpha,
            const double *A, int lda,
            const double *x, int incx, double beta,
            double *y, int incy) {
        blas_stat = cublasDsymv(
            blas_handle, uplo,
            n, &alpha, A, lda, x, incx,
            &beta, y, incy);
        check_cublas_error("cublasDsymv: ");
    }

    void potrf(
            double *A, int N,
            const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER) {
        int lworkf = 0; /* size of workspace */
        /* If devInfo = 0, the Cholesky factorization is successful.
        if devInfo = -i, the i-th parameter is wrong (not counting handle).
        if devInfo = i, the leading minor of order i is not positive definite. */
        MyCuPtr<int> devInfo(1);

        solver_stat = cusolverDnDpotrf_bufferSize(
            solver_handle, uplo,
            N, A, N, &lworkf);
        check_cusolver_error("cusolverDnDpotrf_bufferSize: ");

        MyCuPtr<double> d_workf(lworkf); /* device workspace for getrf */

        solver_stat = cusolverDnDpotrf(
            solver_handle, uplo,
            N, A, N, d_workf.get(), lworkf, devInfo.get());
        check_cusolver_error("cusolverDnDpotrf: ");
        // if (devInfo != 0)
        //     throw std::runtime_error("Cholesky factorization is not successful.");
    }

    void potri(
            double *A, int N,
            const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER) {
        int lworki;
        MyCuPtr<int> devInfo(1);
        solver_stat = cusolverDnDpotri_bufferSize(
            solver_handle, uplo,
            N, A, N, &lworki);
        check_cusolver_error("cusolverDnDpotri_bufferSize: ");

        MyCuPtr<double> d_worki(lworki);

        solver_stat = cusolverDnDpotri(
            solver_handle, uplo,
            N, A, N, d_worki.get(), lworki, devInfo.get());
        check_cusolver_error("cusolverDnDpotri: ");
        // if (devInfo != 0)
        //     throw std::runtime_error("Cholesky inversion is not successful.");
    }

    // In-place invert by Cholesky factorization
    void invert_cholesky(
            double *A, int N,
            const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER) {
        potrf(A, N, uplo);
        potri(A, N, uplo);
    }

    void svd(double *A, double *svals, int nrows, int ncols) {
        int lwork = 0; /* size of workspace */
        MyCuPtr<int> devInfo(1);
        // __device__ int devInfo = -1;

        solver_stat = cusolverDnDgesvd_bufferSize(
            solver_handle, nrows, ncols, &lwork);
        check_cusolver_error("cusolverDnDgesvd_bufferSize: ");

        MyCuPtr<double> d_work(lwork), d_rwork(ncols); /* device workspace */

        solver_stat = cusolverDnDgesvd(
            solver_handle, 'O', 'N', nrows, ncols, A, nrows, svals,
            nullptr, nrows, nullptr, ncols, d_work.get(), lwork,
            d_rwork.get(), devInfo.get());
        check_cusolver_error("cusolverDnDgesvd: ");
        // if (devInfo != 0)
        //     throw std::runtime_error("SVD is not successful.");
    }
};

#endif