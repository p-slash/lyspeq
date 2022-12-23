#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

class MyCuDouble
{
    double *dev_ptr;

    void _alloc(int n);
public:
    MyCuDouble() { dev_ptr = nullptr; }
    MyCuDouble(int n) { _alloc(n); }
    MyCuDouble(int n, double *cpu_ptr) { 
        _alloc(n); asyncCpy(cpu_ptr, n);
    }
    ~MyCuDouble();

    double& operator[](int i) { return dev_ptr[i]; }
    MyCuDouble(const MyCuDouble& udev_ptr) = delete;
    MyCuDouble& operator=(const MyCuDouble& udev_ptr) = delete;

    double* get() const { return dev_ptr; }

    void asyncCpy(double *cpu_ptr, int n, int offset=0);
    void reset();

    void realloc(int n, double *cpu_ptr=nullptr);
};

class CuHelper
{
    cublasStatus_t blas_stat;
    cusolverStatus_t solver_stat;

public:
    cublasHandle_t blas_handle;
    cusolverDnHandle_t solver_handle;

    CuHelper();
    ~CuHelper();

    // Trace of A.B
    // Assumes A and B square matrices NxN, and at least one to be symmetric.
    // No stride or whatsoever. Continous allocation
    // Uses BLAS dot product.
    double trace_dsymm(const double *A, const double *B, int N);

    double trace_ddiagmv(const double *A, const double *B, int N);

    // vT . S . v
    // Assumes S is square symmetric matrix NxN
    double my_cublas_dsymvdot(const double *v, const double *S, double *temp_vector, int N);

    void dcopy(const double *x, double *y, int N);

    void daxpy( double alpha,
                const double *x, double *y,
                int N,
                int incx=1, int incy=1);
    
    void dsymm(cublasSideMode_t side, cublasFillMode_t uplo,
               int m, int n, double alpha,
               const double *A, int lda,
               const double *B, int ldb,
               double beta, double *C, int ldc);

    void dsmyv( cublasFillMode_t uplo,
                int n, double alpha,
                const double *A, int lda,
                const double *x, int incx, double beta,
                double *y, int incy);

    // In-place invert by Cholesky factorization
    void invert_cholesky(double *A, int N);
    void svd(double *A, double *svals, int m, int n);
};

extern CuHelper cuhelper;
#endif
