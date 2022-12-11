#include "mathtools/matrix_helper.hpp"
#include "mathtools/real_field.hpp"

#include <stdexcept>
#include <algorithm>
#include <limits>
#include <utility>
#include <cmath>

#include <gsl/gsl_interp.h>

#ifdef USE_MKL_CBLAS
#include "mkl_lapacke.h"
#else
// These three lines somehow fix OpenBLAS compilation error on macos
// #include <complex.h>
// #define lapack_complex_float    float _Complex
// #define lapack_complex_double   double _Complex
#include "lapacke.h"
#endif

const double
MY_SQRT_2 = 1.41421356237,
MY_SQRT_PI = 1.77245385091,
MY_EPSILON_D = std::numeric_limits<double>::epsilon();

// cblas_dcopy(N, sour, isour, tar, itar); 
template<class InputIt>
constexpr double nonzero_min_element(InputIt first, InputIt last)
{
    if (first == last) return *first;
 
    double smallest = fabs(*first);
    while (++first != last)
    {
        double tmp = fabs(*first);

        if (smallest < MY_EPSILON_D)
            smallest = tmp;
        else if (tmp < smallest && tmp > MY_EPSILON_D) 
            smallest = tmp;
    }

    if (smallest < MY_EPSILON_D)
        smallest = MY_EPSILON_D;

    return smallest;
}

double _window_fn_v(double x, double R, double a)
{
    double gamma_p = (x + (a/2))/R/MY_SQRT_2,
           gamma_m = (x - (a/2))/R/MY_SQRT_2;

    return (erf(gamma_p)-erf(gamma_m))/2;
}

double _integral_erf(double x)
{
    return exp(-x*x)/MY_SQRT_PI + x * erf(x);   
}

double _integrated_window_fn_v(double x, double R, double a)
{
    double xr = x/R/MY_SQRT_2, ar = a/R/MY_SQRT_2;

    return (R/a/MY_SQRT_2) * (_integral_erf(xr+ar) + _integral_erf(xr-ar) - 2*_integral_erf(xr));
}

namespace mxhelp
{
    void copyUpperToLower(double *A, int N)
    {
        for (int i = 1; i < N; ++i)
            cblas_dcopy(i, A+i, N, A+N*i, 1);
    }

    // v always starts at 0, ends at N-1-abs(d)
    void getDiagonal(const double *A, int N, int d, double *v)
    {
        int rowi = (d>=0) ? 0 : -d, counter=0;
        double *vi=v;
        const double *Ai=A+(N+1)*rowi+d;

        while (counter<N-abs(d))
        {
            *vi=*Ai;
            ++vi;
            ++counter;
            Ai+=(N+1);
        }
    }

    double trace_dgemm(const double *A, const double *B, int N)
    {
        double result = 0.;

        for (int i = 0; i < N; ++i)
            result += cblas_ddot(N, A+N*i, 1, B+i, N);

        return result;
    }

    double my_cblas_dsymvdot(const double *v, const double *S,
        double *temp_vector, int N)
    {
        cblas_dsymv(CblasRowMajor, CblasUpper, N, 1., S, N, v, 1, 0,
            temp_vector, 1);

        return cblas_ddot(N, v, 1, temp_vector, 1);
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
        lapack_int *ipiv = new lapack_int[N];
        lapack_int LIN = N, info;

        // Factorize A
        // the LU factorization of a general m-by-n matrix.
        info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, LIN, LIN, A, LIN,
            ipiv);

        LAPACKErrorHandle("ERROR in LU decomposition.", info);

        info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, LIN, A, LIN,
            ipiv);
        LAPACKErrorHandle("ERROR in LU invert.", info);

        delete [] ipiv;
        // dpotrf(CblasUpper, N, A, N); 
        // the Cholesky factorization of a symmetric positive-definite matrix
    }

    void LAPACKE_svd(double *A, double *svals, int m, int n)
    {
        auto superb = std::make_unique<double[]>(n-1);
        lapack_int M = m, N = n, info;
        info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'O', 'N', M, N, A, M, svals, 
            NULL, M, NULL, N, superb.get());
        LAPACKErrorHandle("ERROR in SVD.", info);
    }

    void printfMatrix(const double *A, int nrows, int ncols)
    {
        for (int i = 0; i < nrows; ++i)
        {
            for (int j = 0; j < ncols; ++j)
                printf("%13.5e ", *(A+j+ncols*i));
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

    std::vector<double> fscanfMatrix(const char *fname,int &nrows, int &ncols)
    {
        FILE *toRead;

        toRead = fopen(fname, "r");

        fscanf(toRead, "%d %d\n", &nrows, &ncols);
        std::vector<double> A;
        double tmp;
        A.reserve(nrows*ncols);

        for (int i = 0; i < nrows; ++i)
        {
            for (int j = 0; j < ncols; ++j)
            {
                fscanf(toRead, "%le ", &tmp);
                A.push_back(tmp);
            }
            fscanf(toRead, "\n");
        }

        fclose(toRead);

        return A;
    }

    // class DiaMatrix
    DiaMatrix::DiaMatrix(int nm, int ndia)
        : sandwich_buffer(NULL), ndim(nm), ndiags(ndia)
    {
        // e.g. ndim=724, ndiags=11
        // offsets: [ 5  4  3  2  1  0 -1 -2 -3 -4 -5]
        // when offsets[i]>0, remove initial offsets[i] elements from resomat.T[i]
        // when offsets[i]<0, remove last |offsets[i]| elements from resomat.T[i]
        if (ndiags%2 == 0)
            throw std::runtime_error("DiaMatrix ndiagonal cannot be even!");

        size = ndim*ndiags;
        values = new double[size];

        offsets = std::make_unique<int[]>(ndiags);
        for (int i=ndiags/2, j=0; i > -(ndiags/2)-1; --i, ++j)
            offsets[j] = i;
    }

    DiaMatrix::~DiaMatrix()
    {
        delete [] values;
        freeBuffer();
    }

    void DiaMatrix::orderTranspose()
    {
        double *newmat = new double[size];

        for (int d = 0; d < ndiags; ++d)
            cblas_dcopy(ndim, matrix()+d, ndiags,
                newmat+d*ndim, 1);

        // values = std::move(newmat);
        delete [] values;
        values = newmat;
    }

    double* DiaMatrix::_getDiagonal(int d)
    {
        int off = offsets[d], od1 = 0;
        if (off > 0)  od1 = off;

        return matrix()+(d*ndim+od1);
    }

    // Normalizing this row by row is not yielding somewhat wrong signal matrix
    void DiaMatrix::constructGaussian(double *v, double R_kms, double a_kms)
    {
        for (int d = 0; d < ndiags; ++d)
        {
            int off = offsets[d], nelem = ndim - abs(off);
                // , row = (off < 0) ? -off : 0;
            double *dia_slice = _getDiagonal(d);

            for (int i = 0; i < nelem; ++i) //, ++row)
            {
                int j = i+abs(off);
                *(dia_slice+i) = _window_fn_v(v[j]-v[i], R_kms, a_kms);
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
                    fprintf(toWrite, "%10.3e ", 0.);
                else
                    fprintf(toWrite, "%10.3e ", *(_getDiagonal(d)+j));
            }
            fprintf(toWrite, "\n");
        }

        fclose(toWrite);
    }

    // void DiaMatrix::_getRowIndices(int i, int *indices)
    // {
    //     int noff = ndiags/2;
    //     for (int j = ndiags-1; j >= 0; --j)
    //         indices[ndiags-1-j] = j*ndim+offsets[j]+i;
    //     if (i < noff)
    //         for (int j = 0; j < noff-i; ++j)
    //             indices[j]=indices[ndiags-1-j];
    //     else if (i > ndim-noff-1)
    //         for (int j = 0; j < i-ndim+noff+1; ++j)
    //             indices[ndiags-1-j]=indices[j];
    // }

    void DiaMatrix::_getRowIndices(int i, std::vector<int> &indices)
    {
        indices.clear();
        int noff = ndiags/2;
        for (int j = ndiags-1; j >= 0; --j)
            indices.push_back(j*ndim+offsets[j]+i);
        if (i < noff)
            for (int j = 0; j < noff-i; ++j)
                indices[j]=indices[ndiags-1-j];
        else if (i > ndim-noff-1)
            for (int j = 0; j < i-ndim+noff+1; ++j)
                indices[ndiags-1-j]=indices[j];
    }

    void DiaMatrix::getRow(int i, double *row)
    {
        std::vector<int> indices(ndiags);
        _getRowIndices(i, indices);

        for (int j = 0; j < ndiags; ++j)
            row[j] = values[indices[j]];
    }

    void DiaMatrix::getRow(int i, std::vector<double> &row)
    {
        row.clear();
        std::vector<int> indices(ndiags);
        _getRowIndices(i, indices);

        for (int j = 0; j < ndiags; ++j)
            row.push_back(values[indices[j]]);
    }

    void DiaMatrix::transpose()
    {
        int noff = ndiags/2, nsize;
        for (int d = 0; d < noff; ++d)
        {
            double *v1 = _getDiagonal(d), *v2 = _getDiagonal(ndiags-1-d);
            nsize = ndim - abs(offsets[d]);

            std::swap_ranges(v1, v1+nsize, v2);
        }
    }

    void DiaMatrix::deconvolve(double m) //bool byCol
    {
        const int HALF_PAD_NO = 5;
        int input_size = ndiags+2*HALF_PAD_NO;
        std::unique_ptr<double[]> row = std::make_unique<double[]>(input_size);
        std::vector<int> indices(ndiags);

        // if (byCol)  transpose();

        RealField deconvolver(input_size, 1, row.get());

        for (int i = 0; i < ndim; ++i)
        {
            _getRowIndices(i, indices);
            for (int p = 0; p < ndiags; ++p)
                row[p+HALF_PAD_NO] = values[indices[p]];

            // Set padded regions to zero
            for (int p = 0; p < HALF_PAD_NO; ++p)
            { row[p] = 0;  row[p+ndiags+HALF_PAD_NO] = 0; }

            // deconvolve sinc^-2 factor using fftw
            deconvolver.deconvolveSinc(m);

            for (int p = 0; p < ndiags; ++p)
                values[indices[p]] = row[p+HALF_PAD_NO];
        }

        // if (byCol)  transpose();
    }

    void DiaMatrix::freeBuffer()
    {
        if (sandwich_buffer != NULL)
        {
            delete [] sandwich_buffer;
            sandwich_buffer = NULL;
        }
    }

    void DiaMatrix::multiply(char SIDER, char TRANSR, const double* A, 
        double *B)
    {
        std::fill_n(B, ndim*ndim, 0);

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

            const double *Aslice, *dia_slice = _getDiagonal(d);
            double       *Bslice;

            if (lside)
            {
                Aslice = A + A1*ndim;
                Bslice = B + B1*ndim;

                for (int i = 0; i < nmult; ++i)
                {
                    cblas_daxpy(ndim, *(dia_slice+i), Aslice, 1, Bslice, 1);
                    Bslice += ndim;
                    Aslice += ndim;
                }
            }
            else
            {
                std::swap(A1, B1);

                Aslice = A + A1;
                Bslice = B + B1;

                for (int i = 0; i < ndim; ++i)
                {
                    for (int j = 0; j < nmult; ++j, ++Aslice, ++Bslice)
                        *Bslice += *(dia_slice+j) * *Aslice;

                    Bslice += (ndim-nmult);
                    Aslice += (ndim-nmult);
                }
            }
        }
    }

    void DiaMatrix::sandwich(double *inplace)
    {
        if (sandwich_buffer == NULL)
            sandwich_buffer = new double[ndim*ndim];

        multiply('L', 'N', inplace, sandwich_buffer);
        multiply('R', 'T', sandwich_buffer, inplace);
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
        sandwich_buffer(NULL), nrows(n1), nelem_per_row(nelem_prow), oversampling(osamp)
    {
        ncols = nrows*oversampling + nelem_per_row-1;
        nvals = nrows*nelem_per_row;
        fine_dlambda = dlambda/oversampling;
        values  = new double[nvals];
    }

    OversampledMatrix::~OversampledMatrix()
    {
        delete [] values;
        freeBuffer();
    }

    double* OversampledMatrix::_getRow(int i)
    {
        return matrix()+i*nelem_per_row;
    }

    // R . A = B
    // A should be ncols x ncols symmetric matrix. 
    // B should be nrows x ncols, will be initialized to zero
    void OversampledMatrix::multiplyLeft(const double* A, double *B)
    {
        double *bsub = B;
        const double *Asub = A, *rrow=matrix();

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
        double *bsub = B;
        const double *Asub = A, *rrow=matrix();

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

    void OversampledMatrix::sandwichHighRes(double *B, const double *temp_highres_mat)
    {
        if (sandwich_buffer == NULL)
            sandwich_buffer = new double[nrows*ncols];

        multiplyLeft(temp_highres_mat, sandwich_buffer);
        multiplyRight(sandwich_buffer, B);
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

    void OversampledMatrix::fprintfMatrix(const char *fname)
    {
        FILE *toWrite;
    
        toWrite = fopen(fname, "w");

        fprintf(toWrite, "%d %d\n", nrows, nelem_per_row);
        
        for (int i = 0; i < nrows; ++i)
        {
            for (int j = 0; j < nelem_per_row; ++j)
                fprintf(toWrite, "%3e ", *(_getRow(i)+j));
            fprintf(toWrite, "\n");
        }

        fclose(toWrite);
    }

    void OversampledMatrix::freeBuffer()
    {
        if (sandwich_buffer != NULL)
        {
            delete [] sandwich_buffer;
            sandwich_buffer = NULL;
        }
    }

    // Main resolution object
    Resolution::Resolution(int nm, int ndia) : is_dia_matrix(true), ncols(nm)
    {
        dia_matrix = std::make_unique<DiaMatrix>(nm, ndia);
    }

    Resolution::Resolution(int n1, int nelem_prow, int osamp, double dlambda) :
        is_dia_matrix(false)
    {
        osamp_matrix = std::make_unique<OversampledMatrix>(n1, nelem_prow, osamp, dlambda);
        ncols  = osamp_matrix->getNCols();
    }

    Resolution::Resolution(const Resolution *rmaster, int i1, int i2) :
        is_dia_matrix(rmaster->is_dia_matrix)
    {
        int newsize = i2-i1;

        if (is_dia_matrix)
        {
            int ndiags = rmaster->dia_matrix->ndiags;
            dia_matrix = std::make_unique<DiaMatrix>(newsize, ndiags);

            for (int d = 0; d < ndiags; ++d)
                std::copy_n(rmaster->matrix()+(rmaster->ncols*d)+i1, 
                    newsize, matrix()+newsize*d);

            ncols = newsize;
        }
        else
        {
            int nelemprow = rmaster->osamp_matrix->nelem_per_row, 
                osamp     = rmaster->osamp_matrix->oversampling;

            osamp_matrix = std::make_unique<OversampledMatrix>(newsize, nelemprow, osamp, 1.);
            osamp_matrix->fine_dlambda = rmaster->osamp_matrix->fine_dlambda;

            std::copy_n(rmaster->matrix()+(i1*nelemprow), newsize*nelemprow, matrix());

            ncols = osamp_matrix->getNCols();
        }
    }

    void Resolution::cutBoundary(int i1, int i2)
    {
        int newsize = i2-i1;

        if (is_dia_matrix)
        {
            auto new_dia_matrix = std::make_unique<DiaMatrix>(newsize, dia_matrix->ndiags);

            for (int d = 0; d < dia_matrix->ndiags; ++d)
                std::copy_n(dia_matrix->matrix()+(ncols*d)+i1, newsize, 
                    new_dia_matrix->matrix()+newsize*d);

            dia_matrix = std::move(new_dia_matrix);
            ncols  = newsize;
        }
        else
        {
            auto new_osamp_matrix = std::make_unique<OversampledMatrix>(newsize, 
                osamp_matrix->nelem_per_row, osamp_matrix->oversampling, 1.);
            new_osamp_matrix->fine_dlambda = osamp_matrix->fine_dlambda;

            std::copy_n(osamp_matrix->matrix()+(i1*osamp_matrix->nelem_per_row),
                newsize*osamp_matrix->nelem_per_row, new_osamp_matrix->matrix());

            osamp_matrix = std::move(new_osamp_matrix);
            ncols  = osamp_matrix->getNCols();
        }
    }

    void Resolution::transpose()
    {
        if (is_dia_matrix) dia_matrix->transpose();
    }

    void Resolution::orderTranspose()
    {
        if (is_dia_matrix)
            dia_matrix->orderTranspose();
    }

    double* Resolution::matrix() const
    {
        if (is_dia_matrix)
            return dia_matrix->matrix();
        else
            return osamp_matrix->matrix();

        return NULL;
    }

    int Resolution::getNElemPerRow() const
    {
        if (is_dia_matrix)
            return dia_matrix->ndiags;
        else
            return osamp_matrix->nelem_per_row;

        return 0;
    }

    void Resolution::oversample(int osamp, double dlambda)
    {
        if (!is_dia_matrix) return;

        int noff = dia_matrix->ndiags/2, nelem_per_row = 2*noff*osamp + 1;
        // Using the following simple scaling yields biased results
        // double rescalor = (double) dia_matrix->ndiags / (double) nelem_per_row;
        osamp_matrix = std::make_unique<OversampledMatrix>(ncols, nelem_per_row, osamp, dlambda);

        double *newrow;
        std::vector<double> row, win, wout;
        row.reserve(dia_matrix->ndiags);
        win.reserve(dia_matrix->ndiags);
        wout.reserve(nelem_per_row);

        for (int i = 0; i < dia_matrix->ndiags; ++i)
            win.push_back(i-noff);
        for (int i = 0; i < nelem_per_row; ++i)
            wout.push_back(i*1./osamp-noff);

        gsl_interp *interp_cubic = gsl_interp_alloc(gsl_interp_cspline, dia_matrix->ndiags);
        gsl_interp_accel *acc = gsl_interp_accel_alloc();

        // ncols == nrows for dia matrix
        for (int i = 0; i < ncols; ++i)
        {
            dia_matrix->getRow(i, row);

            newrow = osamp_matrix->matrix()+i*nelem_per_row;

            // interpolate log, shift before log
            double _shift = *std::min_element(row.begin(), row.end())
                - nonzero_min_element(row.begin(), row.end());

            std::for_each(row.begin(), row.end(),
                [_shift](double &f) { f = log(f-_shift); }
            );

            gsl_interp_init(interp_cubic, win.data(), row.data(), dia_matrix->ndiags);

            // Paranoid that std::transform lambda is problematic
            double sum=0;
            for (int jj = 0; jj < nelem_per_row; ++jj)
            {
                newrow[jj] = _shift + exp(gsl_interp_eval(
                    interp_cubic, win.data(), row.data(), wout[jj], acc));
                sum += newrow[jj];
            }

            std::for_each(newrow, newrow+nelem_per_row,
                [sum](double &X) { X/=sum; }
            );

            gsl_interp_accel_reset(acc);
        }

        is_dia_matrix = false;
        ncols = osamp_matrix->getNCols();

        // Clean up
        gsl_interp_free(interp_cubic);
        gsl_interp_accel_free(acc);
        dia_matrix.reset();
    }

    void Resolution::deconvolve(double m)
    {
        if (is_dia_matrix) dia_matrix->deconvolve(m);
    }

    void Resolution::sandwich(double *B, const double *temp_highres_mat)
    {
        if (is_dia_matrix)
        {
            const double *tmat __attribute__((unused)) = temp_highres_mat;
            dia_matrix->sandwich(B);
        }
        else
            osamp_matrix->sandwichHighRes(B, temp_highres_mat);
    }

    void Resolution::freeBuffer()
    {
        if (dia_matrix)   dia_matrix->freeBuffer();
        if (osamp_matrix) osamp_matrix->freeBuffer();
    }

    double Resolution::getMinMemUsage()
    {
        if (is_dia_matrix)  return dia_matrix->getMinMemUsage();
        else                return osamp_matrix->getMinMemUsage();
    }

    double Resolution::getBufMemUsage()
    {
        if (is_dia_matrix)  return dia_matrix->getBufMemUsage();
        else                return osamp_matrix->getBufMemUsage();
    }

    void Resolution::fprintfMatrix(const char *fname)
    {
        if (is_dia_matrix)  return dia_matrix->fprintfMatrix(fname);
        else                return osamp_matrix->fprintfMatrix(fname);
    }
}















