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

    // class DiaMatrix
    DiaMatrix::DiaMatrix(int nm, int ndia) : ndim(nm), ndiags(ndia),
        sandwich_buffer(NULL)
    {
        // e.g. ndim=724, ndiags=11
        // offsets: [ 5  4  3  2  1  0 -1 -2 -3 -4 -5]
        // when offsets[i]>0, remove initial offsets[i] elements from resomat.T[i]
        // when offsets[i]<0, remove last |offsets[i]| elements from resomat.T[i]
        if (ndiags%2 == 0)
            throw std::runtime_error("DiaMatrix ndiagonal cannot be even!");

        size = ndim*ndiags;
        matrix = new double[size]();

        offsets = new int[ndiags];
        for (int i=ndiags/2, j=0; i > -(ndiags/2)-1; --i, ++j)
            offsets[j] = i;
    }

    void DiaMatrix::orderTranspose()
    {
        double* newmat = new double[size];

        for (int d = 0; d < ndiags; ++d)
            for (int i = 0; i < ndim; ++i)
                *(newmat + i+d*ndim) = *(matrix + i*ndiags+d);

        delete [] matrix;
        matrix = newmat;
    }

    double* DiaMatrix::_getDiagonal(int d)
    {
        int off = offsets[d], od1 = 0;
        if (off > 0)  od1 = off;

        return matrix+(d*ndim+od1);
    }

    // Normalizing this row by row is not yielding somewhat wrong signal matrix
    void DiaMatrix::constructGaussian(double *v, double R_kms, double a_kms)
    {
        // std::unique_ptr<double[]> rownorm(new double[ndim]());

        for (int d = 0; d < ndiags; ++d)
        {
            int off = offsets[d], nelem = ndim - abs(off);
                // , row = (off < 0) ? -off : 0;
            double *dia_slice = _getDiagonal(d);

            for (int i = 0; i < nelem; ++i) //, ++row)
            {
                int j = i+abs(off);
                *(dia_slice+i) = _window_fn_v(v[j]-v[i], R_kms, a_kms);
                // rownorm[row] += *(dia_slice+i);
            }
        }

        /*
        // Normalize row by row
        for (int d = 0; d < ndiags; ++d)
        {
            int off = offsets[d], nelem = ndim - abs(off),
                row = (off < 0) ? -off : 0;;
            double *dia_slice = _getDiagonal(d);

            for (int i = 0; i < nelem; ++i, ++row)
                *(dia_slice+i) /= rownorm[row];
        }
        */
    }

    void DiaMatrix::fprintfMatrix(const char *fname)
    {
        FILE *toWrite;
        
        toWrite = fopen(fname, "w");

        fprintf(toWrite, "%d %d\n", ndim, ndiags);
        
        for (int i = 0; i < ndim; ++i)
        {
            for (int j = 0; j < ndim; ++j)
            {
                int off = j-i, d=ndiags/2-off;

                if (abs(off)>ndiags/2)
                    fprintf(toWrite, "0 ");
                else
                    fprintf(toWrite, "%14le ", *(matrix+d*ndim+j));
            }
            fprintf(toWrite, "\n");
        }

        fclose(toWrite);
    }

    DiaMatrix::~DiaMatrix()
    {
        delete [] offsets;
        delete [] matrix;
        freeBuffer();
    }

    void DiaMatrix::freeBuffer()
    {
        if (sandwich_buffer != NULL)
        {
            delete [] sandwich_buffer;
            sandwich_buffer = NULL;
        }
    }

    void DiaMatrix::multiply(int N, char SIDER, char TRANSR, const double* A, 
        double *B)
    {
        if (N != ndim)
            throw std::runtime_error("DiaMatrix multiply operation dimension do not match!");

        std::for_each(B, B+N*N, [&](double &b) { b=0; });

        int transpose = 1;

        if (TRANSR == 'N' || TRANSR == 'n')
            transpose = 1;
        else if (TRANSR == 'T' || TRANSR == 't')
            transpose = -1;
        else
            throw std::runtime_error("DiaMatrix multiply transpose wrong character!");

        bool lside = (SIDER == 'L' || SIDER == 'l'),
             rside = (SIDER == 'R' || SIDER == 'r');
        
        if (!lside && !rside)
            throw std::runtime_error("DiaMatrix multiply SIDER wrong character!");

        /* Left Side:
        if offset > 0 (upper off-diagonals), 
            remove initial offsets[i] elements from DiaMatrix matrix
                when transposed remove last |offsets[i]|
            start from offset row in A to add row by row, 
            add to 0th row of B, end by offset
        if offset < 0 (lower off-diagonals), 
            remove last |offsets[i]| elements from DiaMatrix matrix
                when transposed remove initial offsets[i]
            start from 0th row in A, but end by |offset|, 
            add to offset row in B
        =======================================================
         * Right Side:
        if offset > 0 (upper off-diagonals), 
            remove initial offsets[i] elements from DiaMatrix matrix
                when transposed remove last |offsets[i]|
            start from 0th col in A, but end by |offset|, to add col by col, 
            add to offset col of B
        if offset < 0 (lower off-diagonals), 
            remove last |offsets[i]| elements from DiaMatrix matrix
                when transposed remove initial offsets[i]
            start from offset col in A 
            add to 0th col in B, but end by |offset|
        */

        for (int d = 0; d < ndiags; ++d)
        {
            int off = transpose*offsets[d], 
                nmult = ndim - abs(off),
                A1 = abs(off), B1 = 0;

            if (off < 0) std::swap(A1, B1);
            // if (rside)   std::swap(A1, B1);
            
            // Here's a shorter code. See long version to unpack.
            int i, j, Ni=nmult, Nj=ndim;
            int* di = &i;
            const double *Aslice, *dia_slice = _getDiagonal(d);
            double       *Bslice;

            if (lside)
            {
                Aslice = A + A1*ndim;
                Bslice = B + B1*ndim;
            }
            else
            {
                std::swap(A1, B1);
                std::swap(Ni, Nj);
                Aslice = A + A1;
                Bslice = B + B1;
                di = &j;
            }

            for (i = 0; i < Ni; ++i)
            {
                for (j = 0; j < Nj; ++j, ++Aslice, ++Bslice)
                    *Bslice += *(dia_slice+*di) * *Aslice;

                Bslice += (ndim-Nj);
                Aslice += (ndim-Nj);
            }
        }
    }

    void DiaMatrix::sandwich(int N, double *inplace)
    {
        if (N != ndim)
            throw std::runtime_error("DiaMatrix sandwich operation dimensions do not match!");

        if (sandwich_buffer == NULL)
            sandwich_buffer = new double[ndim*ndim];

        multiply(ndim, 'L', 'N', inplace, sandwich_buffer);
        multiply(ndim, 'R', 'T', sandwich_buffer, inplace);
    }

    double DiaMatrix::getMinMemUsage()
    {
        // Convert to MB by division of 1048576
        double diasize = (double)sizeof(double) * ndim * ndiags / 1048576.;
        double offsize = (double)sizeof(int) * (ndiags+3) / 1048576.;

        return diasize + offsize;
    }

    double DiaMatrix::getBufMemUsage()
    {
        // Convert to MB by division of 1048576
        double bufsize = (double)sizeof(double) * ndim * ndim / 1048576.;

        return bufsize;
    }

    // class OversampledMatrix
    OversampledMatrix::OversampledMatrix(int n1, int nelem_prow, int osamp, double dlambda) : 
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

    OversampledMatrix::~OversampledMatrix()
    {
        // delete [] indices;
        // delete [] iptrs;
        delete [] values;
        freeBuffers();
    }

    double* OversampledMatrix::_getRow(int i)
    {
        return values+i*nelem_per_row;
    }

    // R . A = B
    // A should be ncols x ncols symmetric matrix. 
    // B should be nrows x ncols, will be initialized to zero
    void OversampledMatrix::multiplyLeft(const double* A, double *B)
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
    void OversampledMatrix::multiplyRight(const double* A, double *B)
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

    void OversampledMatrix::sandwichHighRes(double *B)
    {
        if (sandwich_buffer == NULL)
            sandwich_buffer = new double[nrows*ncols];

        multiplyLeft(temp_highres_mat, sandwich_buffer);
        multiplyRight(sandwich_buffer, B);
    }

    double* OversampledMatrix::allocWaveGrid(double w1)
    {
        double *oversamp_wave = new double[ncols];

        for (int i = 0; i < ncols; ++i)
            oversamp_wave[i] = w1 + (i - int(nelem_per_row/2))*fine_dlambda;

        return oversamp_wave;
    }

    double OversampledMatrix::getMinMemUsage()
    {
        // Convert to MB by division of 1048576
        return (double)sizeof(double) * nvals / 1048576.;
    }

    double OversampledMatrix::getBufMemUsage()
    {
        // Convert to MB by division of 1048576
        double highressize  = (double)sizeof(double) * ncols * (ncols+1) / 1048576.,
               sandwichsize = (double)sizeof(double) * nrows * ncols / 1048576.;

        return highressize+sandwichsize;
    }

    void OversampledMatrix::allocateTempHighRes()
    {
        if (temp_highres_mat == NULL)
            temp_highres_mat = new double[ncols*ncols];
    }

    void OversampledMatrix::fprintfMatrix(const char *fname)
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

    void OversampledMatrix::freeBuffers()
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

    // Main resolution object
    Resolution::Resolution(int nm, int ndia)
    {
        dia_matrix   = new DiaMatrix(nm, ndia);
        osamp_matrix = NULL;
        values = dia_matrix->matrix;
        ncols = nm;
    }

    Resolution::Resolution(int n1, int nelem_prow, int osamp, double dlambda)
    {
        osamp_matrix = new OversampledMatrix(n1, nelem_prow, osamp, dlambda);
        dia_matrix   = NULL;
        values = osamp_matrix->values;
        temp_highres_mat = osamp_matrix->temp_highres_mat;
        ncols = osamp_matrix->getNCols();
    }

    Resolution::~Resolution()
    {
        delete dia_matrix;
        delete osamp_matrix;
    }

    void Resolution::orderTranspose()
    {
        if (dia_matrix != NULL)
            dia_matrix->orderTranspose();
    }

    void Resolution::allocateTempHighRes()
    {
        if (temp_highres_mat == NULL)
            temp_highres_mat = new double[ncols*ncols];
    }

    double* Resolution::allocWaveGrid(double w1)
    {
        if (osamp_matrix != NULL) return osamp_matrix->allocWaveGrid(w1);
        else return NULL;
    }

    void Resolution::sandwich(double *B)
    {
        if (dia_matrix != NULL)
            dia_matrix->sandwich(ncols, B);
        else if (osamp_matrix != NULL)
            osamp_matrix->sandwichHighRes(B);
        else throw std::runtime_error("No matrix is allocated.");
    }

    void Resolution::freeBuffers()
    {
        if (dia_matrix != NULL)   dia_matrix->freeBuffer();
        if (osamp_matrix != NULL) osamp_matrix->freeBuffers();
        if (temp_highres_mat != NULL)
        {
            delete [] temp_highres_mat;
            temp_highres_mat = NULL;
        }
    }

    double Resolution::getMinMemUsage()
    {
        if (dia_matrix != NULL)
            return dia_matrix->getMinMemUsage();
        else if (osamp_matrix != NULL)
            return osamp_matrix->getMinMemUsage();
        else throw  std::runtime_error("No matrix is allocated.");
    }

    double Resolution::getBufMemUsage()
    {
        if (dia_matrix != NULL)
            return dia_matrix->getBufMemUsage();
        else if (osamp_matrix != NULL)
            return osamp_matrix->getBufMemUsage();
        else throw  std::runtime_error("No matrix is allocated.");
    }

    void Resolution::fprintfMatrix(const char *fname)
    {
        if (dia_matrix != NULL)
            return dia_matrix->fprintfMatrix(fname);
        else if (osamp_matrix != NULL)
            return osamp_matrix->fprintfMatrix(fname);
        else throw  std::runtime_error("No matrix is allocated.");
    }
}















