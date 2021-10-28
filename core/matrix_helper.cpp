#include "core/matrix_helper.hpp"

#include <stdexcept>
#include <algorithm>
#include <utility>
#include <cmath>
#include <memory>

#ifdef USE_MKL_CBLAS
#include "mkl_lapacke.h"
#else
// These three lines somehow fix OpenBLAS compilation error on macos
#include <complex.h>
#define lapack_complex_float    float _Complex
#define lapack_complex_double   double _Complex
#include "lapacke.h"
#endif

#define SQRT_2 1.41421356237
#define SQRT_PI 1.77245385091

double _window_fn_v(double x, double R, double a)
{
    double gamma_p = (x + (a/2))/R/SQRT_2,
           gamma_m = (x - (a/2))/R/SQRT_2;

    return (erf(gamma_p)-erf(gamma_m))/2;
}

double _integral_erf(double x)
{
    return exp(-x*x)/SQRT_PI + x * erf(x);   
}

double _integrated_window_fn_v(double x, double R, double a)
{
    double xr = x/R/SQRT_2, ar = a/R/SQRT_2;

    return (R/a/SQRT_2) * (_integral_erf(xr+ar) + _integral_erf(xr-ar) - 2*_integral_erf(xr));
}

#undef SQRT_PI
#undef SQRT_2

namespace mxhelp
{
    void copyUpperToLower(double *A, int N)
    {
        for (int i = 1; i < N; ++i)
            for (int j = 0; j < i; ++j)
                *(A+j+N*i) = *(A+i+N*j);
    }

    void vector_add(double *target, const double *source, int size)
    {
        for (int i = 0; i < size; ++i)
            *(target+i) += *(source+i);
    }

    void vector_sub(double *target, const double *source, int size)
    {
        for (int i = 0; i < size; ++i)
            *(target+i) -= *(source+i);
    }

    double trace_dgemm(const double *A, const double *B, int N)
    {
        double result = 0.;

        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                result += (*(A+j+N*i)) * (*(B+i+N*j));

        return result;
    }

    // Assume A and B square symmetric matrices.
    // No stride or whatsoever. Continous allocation
    // Uses CBLAS dot product.
    double trace_dsymm(const double *A, const double *B, int N)
    {
        return cblas_ddot(N*N, A, 1, B, 1);  
    }

    double trace_ddiagmv(const double *A, const double *B, int N)
    {
        return cblas_ddot(N, A, N+1, B, 1);  
    }

    double my_cblas_dsymvdot(const double *v, const double *S, int N)
    {
        double *temp_vector = new double[N], r;

        cblas_dsymv(CblasRowMajor, CblasUpper, N, 1., S, N, v, 1, 0, temp_vector, 1);

        r = cblas_ddot(N, v, 1, temp_vector, 1);

        delete [] temp_vector;

        return r;
    }

    void LAPACKErrorHandle(const char *base, int info)
    {
        if (info != 0)
        {
            char err_msg[50];
            if (info < 0)   sprintf(err_msg, "%s", "Illegal value.");
            else            sprintf(err_msg, "%s", "Singular.");

            fprintf(stderr, "%s: %s\n", base, err_msg);
            throw std::runtime_error(err_msg);
        }
    }

    void LAPACKE_InvertMatrixLU(double *A, int N)
    {
        lapack_int LIN = N, *ipiv, info;
        ipiv = new lapack_int[N];
       
        // Factorize A
        // the LU factorization of a general m-by-n matrix.
        info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, LIN, LIN, A, LIN, ipiv);
        
        LAPACKErrorHandle("ERROR in LU decomposition.", info);

        info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, LIN, A, LIN, ipiv);
        LAPACKErrorHandle("ERROR in LU invert.", info);

        delete [] ipiv;
        // dpotrf(CblasUpper, N, A, N); 
        // the Cholesky factorization of a symmetric positive-definite matrix
    }

    void printfMatrix(const double *A, int nrows, int ncols)
    {
        for (int i = 0; i < nrows; ++i)
        {
            for (int j = 0; j < ncols; ++j)
                printf("%.6le ", *(A+j+ncols*i));
            printf("\n");
        }
    }

    void fprintfMatrix(const char *fname, const double *A, int nrows, int ncols)
    {
        FILE *toWrite;
        
        toWrite = fopen(fname, "w");

        fprintf(toWrite, "%d %d\n", nrows, ncols);
        
        for (int i = 0; i < nrows; ++i)
        {
            for (int j = 0; j < ncols; ++j)
                fprintf(toWrite, "%14le ", *(A+j+ncols*i));
            fprintf(toWrite, "\n");
        }

        fclose(toWrite);
    }

    void fscanfMatrix(const char *fname, double *& A, int &nrows, int &ncols)
    {
        FILE *toRead;

        toRead = fopen(fname, "r");

        fscanf(toRead, "%d %d\n", &nrows, &ncols);
        A = new double[nrows*ncols];

        for (int i = 0; i < nrows; ++i)
        {
            for (int j = 0; j < ncols; ++j)
                fscanf(toRead, "%le ", &A[j+ncols*i]);
            fscanf(toRead, "\n");
        }

        fclose(toRead);
    }

    // class Resolution
    Resolution::Resolution(int n1, int nelem_prow, int osamp, double dlambda) : 
        nrows(n1), nelem_per_row(nelem_prow), oversampling(osamp),
        sandwich_buffer(NULL), temp_highres_mat(NULL)
    {
        nvals = nrows*nelem_per_row;
        ncols = nrows*oversampling + nelem_per_row-1;
        fine_dlambda = dlambda/oversampling;
        values  = new double[nvals];
        
        // nptrs = nrows+1;
        // indices = new int[nvals];
        // iptrs   = new int[nptrs];

        // for (int i = 0; i < nvals; ++i)
        //     indices[i] = int(i/nelem_per_row)*oversampling + (i%nelem_per_row);
        // for (int i = 0; i < nptrs; ++i)
        //     iptrs[i]   = i*nelem_per_row;
    }

    Resolution::~Resolution()
    {
        // delete [] indices;
        // delete [] iptrs;
        delete [] values;
        freeBuffers();
    }

    double* Resolution::_getRow(int i)
    {
        return values+i*nelem_per_row;
    }

    // R . A = B
    // A should be ncols x ncols symmetric matrix. 
    // B should be nrows x ncols, will be initialized to zero
    void Resolution::multiplyLeft(const double* A, double *B)
    {
        double *bsub = B, *rrow=values;
        const double *Asub = A;

        for (int i = 0; i < nrows; ++i)
        {
            // double *rrow = _getRow(i), *bsub = B + i*ncols;
            // const double *Asub = A + i*ncols*oversampling;

            cblas_dgemv(CblasRowMajor, CblasTrans,
                nelem_per_row, ncols, 1., Asub, ncols, 
                rrow, 1, 0, bsub, 1);

            bsub += ncols;
            Asub += ncols*oversampling;
            rrow += nelem_per_row;
        }
    }
    // A . R^T = B
    // A should be nrows x ncols matrix. 
    // B should be nrows x nrows, will be initialized to zero
    void Resolution::multiplyRight(const double* A, double *B)
    {
        double *bsub = B, *rrow=values;
        const double *Asub = A;

        for (int i = 0; i < nrows; ++i)
        {
            // double *rrow = _getRow(i), *bsub = B + i;
            // const double *Asub = A + i*oversampling;

            cblas_dgemv(CblasRowMajor, CblasNoTrans,
                nrows, nelem_per_row, 1., Asub, ncols, 
                rrow, 1, 0, bsub, nrows);

            ++bsub;
            Asub += oversampling;
            rrow += nelem_per_row;
        }
    }

    void Resolution::sandwichHighRes(double *B)
    {
        if (sandwich_buffer == NULL)
            sandwich_buffer = new double[nrows*ncols];

        multiplyLeft(temp_highres_mat, sandwich_buffer);
        multiplyRight(sandwich_buffer, B);
    }

    double* Resolution::allocWaveGrid(double w1)
    {
        double *oversamp_wave = new double[ncols];

        for (int i = 0; i < ncols; ++i)
            oversamp_wave[i] = w1 - (i + int(nelem_per_row/2))*fine_dlambda;

        return oversamp_wave;
    }

    double Resolution::getMinMemUsage()
    {
        // Convert to MB by division of 1048576
        return (double)sizeof(double) * nvals / 1048576.;
    }

    double Resolution::getBufMemUsage()
    {
        // Convert to MB by division of 1048576
        double highressize  = (double)sizeof(double) * ncols * (ncols+1) / 1048576.,
               sandwichsize = (double)sizeof(double) * nrows * ncols / 1048576.;

        return highressize+sandwichsize;
    }

    void Resolution::allocateTempHighRes()
    {
        if (temp_highres_mat == NULL)
            temp_highres_mat = new double[ncols*ncols];
    }

    void Resolution::fprintfMatrix(const char *fname)
    {
        FILE *toWrite;
    
        toWrite = fopen(fname, "w");

        fprintf(toWrite, "%d %d\n", nrows, nelem_per_row);
        
        for (int i = 0; i < nrows; ++i)
        {
            for (int j = 0; j < nrows; ++j)
            {
                int off = j-i*oversampling;

                if (off>=0 && off < nelem_per_row)
                    fprintf(toWrite, "%14le ", *(values+off+nelem_per_row*i));
                else
                    fprintf(toWrite, "0 ");
            }
            fprintf(toWrite, "\n");
        }

        fclose(toWrite);
    }

    void Resolution::freeBuffers()
    {
        if (sandwich_buffer != NULL)
        {
            delete [] sandwich_buffer;
            sandwich_buffer = NULL;
        }
        if (temp_highres_mat != NULL)
        {
            delete [] temp_highres_mat;
            temp_highres_mat = NULL;
        }
    }
}















