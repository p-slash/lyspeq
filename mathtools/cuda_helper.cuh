#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <memory>
#include <stdexcept>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

const static int MYCU_BLOCK_SIZE = 256;

template<typename T>
class MyCuPtr {
    cudaError_t cuda_stat;
    T *dev_ptr;
    size_t size;

    void check_cuda_error(std::string err_msg) {
        if (cuda_stat != cudaSuccess)
            throw std::runtime_error(err_msg);
    }

    void _alloc(int n) {
        cuda_stat = cudaMalloc((void**) &dev_ptr, n*sizeof(T));
        if (cuda_stat != cudaSuccess) {
            dev_ptr = nullptr;
            throw std::runtime_error("cudaMalloc failed.");
        }
        size = n;
    }
public:
    MyCuPtr() { dev_ptr = nullptr; size = 0; }
    MyCuPtr(int n) { _alloc(n); }
    MyCuPtr(int n, T *cpu_ptr, cudaStream_t stream=NULL) { 
        _alloc(n); asyncCpy(cpu_ptr, n, 0, stream);
    }
    ~MyCuPtr() { cudaFree(dev_ptr); }

    T& operator[](int i) { return dev_ptr[i]; }
    MyCuPtr(const MyCuPtr& udev_ptr) = delete;
    MyCuPtr& operator=(const MyCuPtr& udev_ptr) = delete;

    T* get() const { return dev_ptr; }

    void memset(int value=0) {
        cuda_stat = cudaMemset(dev_ptr, value, size * sizeof(T));
        check_cuda_error("cudaMemset: ");
    }

    void asyncCpy(T *cpu_ptr, int n, int offset=0, cudaStream_t stream=NULL) {
        cuda_stat = cudaMemcpyAsync(
            dev_ptr + offset, cpu_ptr, n * sizeof(T), cudaMemcpyHostToDevice,
            stream);
        check_cuda_error("asyncCpy::cudaMemcpyAsync: ");
    }

    void asyncDwn(T *cpu_ptr, int n, int offset=0, cudaStream_t stream=NULL) {
        cuda_stat = cudaMemcpyAsync(
            cpu_ptr, dev_ptr + offset, n * sizeof(T), cudaMemcpyDeviceToHost,
            stream);
        check_cuda_error("asyncDwn::cudaMemcpyAsync: ");
    }

    void syncDownload(T *cpu_ptr, int n, int offset=0) {
        cuda_stat = cudaMemcpy(
            cpu_ptr, dev_ptr + offset, n * sizeof(T), cudaMemcpyDeviceToHost);
        check_cuda_error("syncDownload::cudaMemcpy: ");
    }

    void reset() {
        if (dev_ptr != nullptr) {
            cudaFree(dev_ptr);
            dev_ptr = nullptr;
            size = 0;
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
        if (cuda_stat != cudaSuccess)
            throw std::runtime_error(err_msg);
    }
public:
    MyCuStream(bool default_stream=false) {
        if (default_stream) {
            stream = 0;
            return;
        }

        cuda_stat = cudaStreamCreate(&stream);
        check_cuda_error("cudaStreamCreate: ");
    }

    ~MyCuStream() {
        if (stream == 0)
            return;
        cudaStreamDestroy(stream);
    }

    cudaStream_t get() const {
        return stream;
    }

    void sync() {
        cuda_stat = cudaStreamSynchronize(stream);
        check_cuda_error("cudaStreamSynchronize: ");
    }

    static void syncMainStream() {
        if (cudaStreamSynchronize(0) != cudaSuccess)
            throw std::runtime_error("cudaStreamSynchronize: ");
    }

    static void syncDevice() {
        if (cudaDeviceSynchronize() != cudaSuccess)
            throw std::runtime_error("cudaDeviceSynchronize: ");
    }
};

static double _CPU_ONE_VALUE_ = 1, _CPU_ZERO_VALUE_ = 0;
static MyCuPtr<double> _DEV_ONE_VALUE_(1, &_CPU_ONE_VALUE_),
                       _DEV_ZERO_VALUE_(1, &_CPU_ZERO_VALUE_);


class CuBlasHelper {
    cublasStatus_t blas_stat;
    double *_one_ptr, *_zero_ptr;

    void check_cublas_error(std::string err_msg) {
        if (blas_stat != CUBLAS_STATUS_SUCCESS) {
            err_msg += std::string(cublasGetStatusString(blas_stat));
            throw std::runtime_error(err_msg);
        }
    }
public:
    cublasHandle_t blas_handle;

    CuBlasHelper() : _one_ptr(&_CPU_ONE_VALUE_), _zero_ptr(&_CPU_ZERO_VALUE_) {
        blas_stat = cublasCreate(&blas_handle);
        check_cublas_error("CUBLAS initialization failed: ");
    }

    ~CuBlasHelper() {
        cublasDestroy(blas_handle);
    }

    void setStream(MyCuStream& stream) {
        blas_stat = cublasSetStream(blas_handle, stream.get());
        check_cublas_error("cublasSetStream: ");
    }

    void resetStream() {
        blas_stat = cublasSetStream(blas_handle, 0);
        check_cublas_error("cublasSetStream: ");
    }

    void setPointerMode2Host() {
        blas_stat = cublasSetPointerMode(blas_handle, CUBLAS_POINTER_MODE_HOST);
        check_cublas_error("cublasSetPointerMode - host:");
        _one_ptr = &_CPU_ONE_VALUE_;
        _zero_ptr = &_CPU_ZERO_VALUE_;
    }

    void setPointerMode2Device() {
        blas_stat = cublasSetPointerMode(
            blas_handle, CUBLAS_POINTER_MODE_DEVICE);
        check_cublas_error("cublasSetPointerMode - device:");
        _one_ptr = _DEV_ONE_VALUE_.get();
        _zero_ptr = _DEV_ZERO_VALUE_.get();
    }

    // Trace of A.B
    // Assumes A and B square matrices NxN, and at least one to be symmetric.
    // No stride or whatsoever. Continous allocation
    // Uses BLAS dot product.
    void trace_dsymm(const double *A, const double *B, int N, double *c_res) {
        blas_stat = cublasDdot(blas_handle, N*N, A, 1, B, 1, c_res);
        check_cublas_error("trace_dsymm/cublasDdot: ");
    }

    void trace_ddiagmv(
            const double *A, const double *B, int N, double *c_res
    ) {
        blas_stat = cublasDdot(blas_handle, N, A, N+1, B, 1, c_res);
        check_cublas_error("trace_ddiagmv/cublasDdot: ");
    }

    // vT . S . v
    // Assumes S is square symmetric matrix NxN
    void my_cublas_dsymvdot(
            const double *v, const double *S, double *temp_vector, int N,
            double *c_res,
            const cublasFillMode_t uplo=CUBLAS_FILL_MODE_LOWER
    ) {
        dsymv(N, S, N, v, 1, temp_vector, 1, uplo);
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
            int incx=1, int incy=1
    ) {
        blas_stat = cublasDaxpy(blas_handle, N, &alpha, x, incx, y, incy);
        check_cublas_error("cublasDaxpy: ");
    }

    void dsymm(
            cublasSideMode_t side,
            int m, int n,
            const double *A, int lda,
            const double *B, int ldb,
            double *C, int ldc,
            const cublasFillMode_t uplo=CUBLAS_FILL_MODE_LOWER
    ) {
        blas_stat = cublasDsymm(
            blas_handle, side, uplo,
            m, n, _one_ptr,
            A, lda, B, ldb,
            _zero_ptr, C, ldc);
        check_cublas_error("cublasDsymm: ");
    }

    void dsymv(
            int n,
            const double *A, int lda,
            const double *x, int incx,
            double *y, int incy,
            const cublasFillMode_t uplo=CUBLAS_FILL_MODE_LOWER
    ) {
        blas_stat = cublasDsymv(
            blas_handle, uplo,
            n, _one_ptr, A, lda, x, incx,
            _zero_ptr, y, incy);
        check_cublas_error("cublasDsymv: ");
    }
};


class CuSolverHelper
{
    MyCuPtr<int> devInfo;
    int cpuInfo;
    cusolverStatus_t solver_stat;

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

    void check_info(std::string err_msg) {
        cpuInfo = 0;
        devInfo.syncDownload(&cpuInfo, 1);

        if (cpuInfo != 0)
            throw std::runtime_error(err_msg);
    }

public:
    cusolverDnHandle_t solver_handle;

    CuSolverHelper() {
        solver_stat = cusolverDnCreate(&solver_handle);
        check_cusolver_error("CUSOLVER initialization failed: ");
        devInfo.realloc(1);
    }

    ~CuSolverHelper() {
        cusolverDnDestroy(solver_handle);
        // cudaDeviceReset();
    }

    void setStream(MyCuStream& stream) {
        solver_stat = cusolverDnSetStream(solver_handle, stream.get());
        check_cusolver_error("cusolverDnSetStream: ");
    }

    void resetStream() {
        solver_stat = cusolverDnSetStream(solver_handle, 0);
        check_cusolver_error("cusolverDnSetStream: ");
    }

    void potrf(
            double *A, int N,
            const cublasFillMode_t uplo=CUBLAS_FILL_MODE_LOWER
    ) {
        int lworkf = 0; /* size of workspace */
        /* If devInfo = 0, the Cholesky factorization is successful.
        if devInfo = -i, the i-th parameter is wrong (not counting handle).
        if devInfo = i, the leading minor of order i is not positive definite.
        */

        solver_stat = cusolverDnDpotrf_bufferSize(
            solver_handle, uplo,
            N, A, N, &lworkf);
        check_cusolver_error("cusolverDnDpotrf_bufferSize: ");

        MyCuPtr<double> d_workf(lworkf); /* device workspace for getrf */

        solver_stat = cusolverDnDpotrf(
            solver_handle, uplo,
            N, A, N, d_workf.get(), lworkf, devInfo.get());
        check_cusolver_error("cusolverDnDpotrf: ");

        check_info("Cholesky factorization is not successful.");
    }

    void potri(
            double *A, int N,
            const cublasFillMode_t uplo=CUBLAS_FILL_MODE_LOWER
    ) {
        int lworki;

        solver_stat = cusolverDnDpotri_bufferSize(
            solver_handle, uplo,
            N, A, N, &lworki);
        check_cusolver_error("cusolverDnDpotri_bufferSize: ");

        MyCuPtr<double> d_worki(lworki);

        solver_stat = cusolverDnDpotri(
            solver_handle, uplo,
            N, A, N, d_worki.get(), lworki, devInfo.get());
        check_cusolver_error("cusolverDnDpotri: ");

        check_info("Cholesky inversion is not successful.");
    }

    // In-place invert by Cholesky factorization
    void invert_cholesky(
            double *A, int N,
            const cublasFillMode_t uplo=CUBLAS_FILL_MODE_LOWER
    ) {
        potrf(A, N, uplo);
        potri(A, N, uplo);
    }

    void svd(double *A, double *svals, int nrows, int ncols) {
        int lwork = 0; /* size of workspace */

        solver_stat = cusolverDnDgesvd_bufferSize(
            solver_handle, nrows, ncols, &lwork);
        check_cusolver_error("cusolverDnDgesvd_bufferSize: ");

        MyCuPtr<double> d_work(lwork), d_rwork(ncols); /* device workspace */

        solver_stat = cusolverDnDgesvd(
            solver_handle, 'O', 'N', nrows, ncols, A, nrows, svals,
            nullptr, nrows, nullptr, ncols, d_work.get(), lwork,
            d_rwork.get(), devInfo.get());
        check_cusolver_error("cusolverDnDgesvd: ");

        check_info("SVD is not successful.");
    }
};

static CuBlasHelper cublas_helper;
static CuSolverHelper cusolver_helper;

#endif