#include "mathtools/matrix_helper.hpp"
#include "mathtools/real_field.hpp"

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>

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

#if defined(ENABLE_OMP)
#include "omp.h"
#endif

const double
MY_SQRT_2 = 1.41421356237,
MY_SQRT_PI = 1.77245385091,
MY_EPSILON_D = 1e-15;

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
    #define BLOCK_SIZE 32
    void copyUpperToLower(double *A, int N) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < N; i += BLOCK_SIZE) {
            for (int j = i; j < N; j += BLOCK_SIZE) {
                int kmax = std::min(i + BLOCK_SIZE, N),
                    lmax = std::min(j + BLOCK_SIZE, N);

                for (int k = i; k < kmax; ++k)
                    #pragma omp simd
                    for (int l = std::max(k, j); l < lmax; ++l)
                        A[k + l * N] = A[l + k * N];
            }
        }
    }

    void transpose_copy(const double *A, double *B, int M, int N) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < M; i += BLOCK_SIZE) {
            for (int j = 0; j < N; j += BLOCK_SIZE) {
                int kmax = std::min(i + BLOCK_SIZE, M),
                    lmax = std::min(j + BLOCK_SIZE, N);

                #pragma omp simd collapse(2)
                for (int l = j; l < lmax; ++l)
                    for (int k = i; k < kmax; ++k)
                        B[k + l * M] = A[l + k * N];
            }
        }
    }
    #undef BLOCK_SIZE

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

    double my_cblas_dsymvdot(
            const double *v, const double *S, double *temp_vector, int N
    ) {
        cblas_dsymv(
            CblasRowMajor, CblasUpper, N, 1.,
            S, N, v, 1, 0, temp_vector, 1);

        return cblas_ddot(N, v, 1, temp_vector, 1);
    }


    // Slow!
    double my_cblas_dsymvdot(const double *v, const double *S, int N) {
        double sum = 0;

        #pragma omp parallel for schedule(static, 1) reduction(+:sum)
        for (int i = 0; i < N; ++i)
        {
            sum += S[i + i * N] * v[i] * v[i];
            for (int j = i; j < N; ++j)
                sum += 2 * S[j + i * N] * v[i] * v[j];
        }

        return sum;
    }

    void LAPACKErrorHandle(const char *base, int info) {
        if (info != 0) {
            std::string err_msg(base);
            if (info < 0)   err_msg += ": Illegal value.";
            else            err_msg += ": Singular.";

            fprintf(stderr, "%s\n", err_msg.c_str());
            throw std::runtime_error(err_msg);
        }
    }

    void LAPACKE_InvertMatrixLU(double *A, int N) {
        static std::vector<lapack_int> ipiv;
        ipiv.resize(N);
        lapack_int LIN = N, info = 0;

        // Factorize A
        // the LU factorization of a general m-by-n matrix.
        info = LAPACKE_dgetrf(
            LAPACK_ROW_MAJOR, LIN, LIN, A, LIN, ipiv.data());
        LAPACKErrorHandle("ERROR in LU decomposition.", info);

        info = LAPACKE_dgetri(
            LAPACK_ROW_MAJOR, LIN, A, LIN, ipiv.data());
        LAPACKErrorHandle("ERROR in LU invert.", info);

        // dpotrf(CblasUpper, N, A, N); 
        // the Cholesky factorization of a symmetric positive-definite matrix
    }

    std::vector<int> _setEmptyIndices(double *S, int N) {
        std::vector<int> empty_indx;

        for (int i = 0; i < N; ++i) {
            if (S[(N + 1) * i] == 0) {
                empty_indx.push_back(i);
                S[(N + 1) * i] = 1.;
            }
        }

        return empty_indx;
    }

    int LAPACKE_InvertMatrixLU_safe(double *A, int N)
    {
        // find empty diagonals
        // assert all elements on that row and col are zero
        // replace with 1, then invert, then replace with zero
        std::vector<int> empty_indx = _setEmptyIndices(A, N);

        LAPACKE_InvertMatrixLU(A, N);

        for (const auto &i : empty_indx)
            A[(N + 1) * i] = 0;

        return N - empty_indx.size();
    }

    void LAPACKE_InvertSymMatrixLU_damped(double *S, int N, double damp) {
        int size = N * N;
        auto _Amat = std::make_unique<double[]>(size),
             _Bmat = std::make_unique<double[]>(size);
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            N, N, N, 1., S, N,
            S, N, 0, _Amat.get(), N);

        // Scaling damping
        // cblas_dscal(N, 1. + damp, _Amat.get(), N + 1);

        // Additive damping
        for (int i = 0; i < size; i += N + 1)
            _Amat[i] += damp;

        LAPACKE_InvertMatrixLU(_Amat.get(), N);

        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            N, N, N, 0.5, _Amat.get(), N,
            S, N, 0, _Bmat.get(), N);

        transpose_copy(_Bmat.get(), S, N, N);

        cblas_daxpy(N, 1, _Bmat.get(), 1, S, 1);
    }

    double LAPACKE_RcondSvd(const double *A, int N, double *sjump) {
        int size = N * N;
        auto B = std::make_unique<double[]>(size),
             svals = std::make_unique<double[]>(N),
             superb = std::make_unique<double[]>(N - 1);

        lapack_int LIN = N, info = 0;
        std::copy_n(A, size, B.get());

        info = LAPACKE_dgesvd(
            LAPACK_ROW_MAJOR, 'N', 'N', LIN, LIN, B.get(), LIN, svals.get(), 
            NULL, LIN, NULL, LIN, superb.get());
        LAPACKErrorHandle("ERROR in SVD.", info);

        if (sjump == nullptr)
            return svals[N - 1] / svals[0];

        *sjump = 0;
        for (int i = N - 1; i > 0; --i)
        {
            if ((svals[i - 1] / svals[i]) > 8.)
                *sjump = svals[i];
            else
                break;
        }

        return svals[N - 1] / svals[0];
    }


    int stableInvertSym(double *S, int N, int &dof, double &damp) {
        int warn = 0;
        std::vector<int> empty_indx = _setEmptyIndices(S, N);
        dof = N - empty_indx.size();
        damp = 0;

        double rcond = 0;
        rcond = LAPACKE_RcondSvd(S, N, &damp);

        if (damp != 0) {
            warn = 1;
            LAPACKE_InvertSymMatrixLU_damped(S, N, damp);
        } else {
            LAPACKE_InvertMatrixLU(S, N);
        }

        for (const auto &i : empty_indx)
            S[(N + 1) * i] = 0;

        return warn;
    }


    void LAPACKE_solve(double *A, int N, double *b) {
        static std::vector<lapack_int> ipiv;
        ipiv.resize(N);

        lapack_int LIN = N, info = 0;
        info = LAPACKE_dgesv(
            LAPACK_ROW_MAJOR, LIN, 1, A, LIN, ipiv.data(), b, 1);
        // info = LAPACKE_dposv(LAPACK_ROW_MAJOR, 'U', LIN, 1, S, LIN,b, 1);

        LAPACKErrorHandle("ERROR in solve_safe.", info);
    }


    void LAPACKE_solve_safe(double *S, int N, double *b) {
        std::vector<int> empty_indx = _setEmptyIndices(S, N);

        LAPACKE_solve(S, N, b);

        for (const auto &i : empty_indx)
            b[i] = 0;
    }


    void LAPACKE_safeSolveCho(double *S, int N, double *b) {
        std::vector<int> empty_indx = _setEmptyIndices(S, N);

        lapack_int LIN = N, info = 0;
        info = LAPACKE_dposv(LAPACK_ROW_MAJOR, 'U', LIN, 1, S, LIN, b, 1);

        LAPACKErrorHandle("ERROR in safeSolveCho.", info);
        for (const auto &i : empty_indx)
            b[i] = 0;
    }


    void LAPACKE_solve_damped(const double *S, int N, double *b, double damp) {
        auto _Amat = std::make_unique<double[]>(N * N),
             _Ab = std::make_unique<double[]>(N);

        cblas_dsymv(
            CblasRowMajor, CblasUpper,
            N, 1., S, N, 
            b, 1, 0, _Ab.get(), 1);
        std::copy_n(_Ab.get(), N, b);

        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            N, N, N, 1., S, N,
            S, N, 0, _Amat.get(), N);

        // Additive damping
        for (int i = 0; i < N; ++i)
            _Amat[i * (N + 1)] += damp;

        LAPACKE_solve(_Amat.get(), N, b);
    }


    void LAPACKE_stableSymSolve(double *S, int N, double *b) {
        std::vector<int> empty_indx = _setEmptyIndices(S, N);

        double rcond = 0, sjump = 0;
        rcond = LAPACKE_RcondSvd(S, N, &sjump);

        if (sjump != 0)
            LAPACKE_solve_damped(S, N, b, sjump);
        else
            LAPACKE_solve(S, N, b);

        for (const auto &i : empty_indx)
            b[i] = 0;
    }

    void LAPACKE_svd(double *A, double *svals, int m, int n)
    {
        auto superb = std::make_unique<double[]>(n-1);
        lapack_int M = m, N = n, info;
        info = LAPACKE_dgesvd(
            LAPACK_COL_MAJOR, 'O', 'N', M, N, A, M, svals, 
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

    void DiaMatrix::orderTranspose()
    {
        double *newmat = new double[size];

        transpose_copy(matrix(), newmat, ndim, ndiags);

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
        int input_size = ndiags + 2 * HALF_PAD_NO;
        RealField deconvolver(input_size, 1);
        std::vector<int> indices(ndiags);

        // if (byCol)  transpose();

        for (int i = 0; i < ndim; ++i)
        {
            _getRowIndices(i, indices);
            for (int p = 0; p < ndiags; ++p)
                deconvolver.field_x[p + HALF_PAD_NO] = values[indices[p]];

            // Set padded regions to zero
            for (int p = 0; p < HALF_PAD_NO; ++p) {
                deconvolver.field_x[p] = 0;
                deconvolver.field_x[p + ndiags + HALF_PAD_NO] = 0;
            }

            // deconvolve sinc^-2 factor using fftw
            deconvolver.deconvolveSinc(m);

            for (int p = 0; p < ndiags; ++p)
                values[indices[p]] = deconvolver.field_x[p + HALF_PAD_NO];
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

    void DiaMatrix::multiply(
            CBLAS_SIDE SIDER, CBLAS_TRANSPOSE TRANSR,
            const double* A, double *B) {
        std::fill_n(B, ndim*ndim, 0);

        int transpose = 1;

        if (TRANSR == CblasNoTrans)
            transpose = 1;
        else if (TRANSR == CblasTrans)
            transpose = -1;
        else
            throw std::runtime_error(
                "DiaMatrix multiply transpose wrong character!");

        bool lside = (SIDER == CblasLeft), rside = (SIDER == CblasRight);
        
        if (!lside && !rside)
            throw std::runtime_error(
                "DiaMatrix multiply SIDER wrong character!");

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

    void DiaMatrix::multiplyLeft(const double* A, double *B) {
        std::fill_n(B, ndim * ndim, 0);

        for (int d = 0; d < ndiags; ++d)
        {
            int off = offsets[d],
                poff = std::max(0, off),
                moff = std::max(0, -off);

            #pragma omp parallel for simd collapse(2)
            for (int i = 0; i < ndim - abs(off); ++i)
                for (int j = 0; j < ndim; ++j){
                    B[j + (i + moff) * ndim] += 
                        values[poff + d * ndim + i] * A[j + (i + poff) * ndim];
                }
        }
    }

    void DiaMatrix::multiplyRightT(const double* A, double *B) {
        std::fill_n(B, ndim * ndim, 0);

        for (int d = 0; d < ndiags; ++d)
        {
            int off = offsets[d],
                moff = std::max(0, -off),
                poff = std::max(0, off);

            #pragma omp parallel for simd collapse(2)
            for (int i = 0; i < ndim; ++i)
                for (int j = 0; j < ndim - abs(off); ++j) {
                    B[j + i * ndim + moff] +=
                        values[j + poff + d * ndim] * A[j + i * ndim + poff];
                }
        }
    }

    void DiaMatrix::sandwich(double *inplace)
    {
        if (sandwich_buffer == NULL)
            sandwich_buffer = new double[ndim*ndim];

        multiplyLeft(inplace, sandwich_buffer);
        multiplyRightT(sandwich_buffer, inplace);
    }

    double DiaMatrix::getMinMemUsage()
    {
        // Convert to MB by division of 1048576
        double diasize = (double)sizeof(double) * ndim * ndiags / 1048576.;
        double offsize = (double)sizeof(int) * (ndiags+3) / 1048576.;

        return diasize + offsize;
    }

    // class OversampledMatrix
    OversampledMatrix::OversampledMatrix(
            int n1, int nelem_prow, int osamp, double dlambda
    ) : sandwich_buffer(NULL), nrows(n1), nelem_per_row(nelem_prow),
        oversampling(osamp)
    {
        ncols = nrows*oversampling + nelem_per_row-1;
        size = nrows*nelem_per_row;
        fine_dlambda = dlambda/oversampling;
        values  = new double[size];
    }


    // R . A = B
    // A should be ncols x ncols symmetric matrix. 
    // B should be nrows x ncols, will be initialized to zero
    void OversampledMatrix::multiplyLeft(const double* A, double *B)
    {
        #pragma omp parallel for
        for (int i = 0; i < nrows; ++i)
            cblas_dgemv(
                CblasRowMajor, CblasTrans,
                nelem_per_row, ncols, 1., A + i * oversampling * ncols, ncols, 
                values + i * nelem_per_row, 1, 0, B + i * ncols, 1);
    }

    // A . R^T = B
    // A should be nrows x ncols matrix. 
    // B should be nrows x nrows, will be initialized to zero
    // Assumes B will be symmetric
    void OversampledMatrix::multiplyRight(const double* A, double *B)
    {
        std::fill_n(B, nrows * nrows, 0);
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < nrows; ++i)
            for (int j = i; j < nrows; ++j)
                for (int k = 0; k < nelem_per_row; ++k)
                    B[j + i * nrows] +=
                        A[k + j * oversampling + i * ncols]
                        * values[k + j * nelem_per_row];

        copyUpperToLower(B, nrows);
    }

    void OversampledMatrix::sandwichHighRes(double *B, const double *A)
    {
        if (sandwich_buffer == NULL)
            sandwich_buffer = new double[nrows*ncols];
 
        multiplyLeft(A, sandwich_buffer);
        multiplyRight(sandwich_buffer, B);
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


    void Resolution::oversample(int osamp, double dlambda)
    {
        if (!is_dia_matrix) return;

        int noff = dia_matrix->ndiags / 2,
            nelem_per_row = 2 * noff * osamp + 1;
        // Using the following simple scaling yields biased results
        // double rescalor = (double) dia_matrix->ndiags / (double) nelem_per_row;
        osamp_matrix = std::make_unique<OversampledMatrix>(
            ncols, nelem_per_row, osamp, dlambda);

        double *newrow;
        std::vector<double> row, win, wout;
        row.reserve(dia_matrix->ndiags);
        win.reserve(dia_matrix->ndiags);
        wout.reserve(nelem_per_row);

        for (int i = 0; i < dia_matrix->ndiags; ++i)
            win.push_back((i - noff) * dlambda);
        for (int i = 0; i < nelem_per_row; ++i)
            wout.push_back((i * 1. / osamp - noff) * dlambda);

        gsl_interp *interp_cubic = gsl_interp_alloc(
            gsl_interp_cspline, dia_matrix->ndiags);
        gsl_interp_accel *acc = gsl_interp_accel_alloc();

        // ncols == nrows for dia matrix
        for (int i = 0; i < ncols; ++i)
        {
            dia_matrix->getRow(i, row);

            newrow = osamp_matrix->matrix() + i * nelem_per_row;

            // interpolate log, shift before log
            double _shift =
                *std::min_element(row.begin(), row.end())
                - nonzero_min_element(row.begin(), row.end());

            std::for_each(
                row.begin(), row.end(),
                [_shift](double &f) { f = log(f - _shift); }
            );

            gsl_interp_init(
                interp_cubic, win.data(), row.data(), dia_matrix->ndiags);

            for (int jj = 0; jj < nelem_per_row; ++jj)
                newrow[jj] = 
                    _shift + exp(gsl_interp_eval(
                        interp_cubic, win.data(), row.data(), wout[jj], acc
                ));

            double isum = 1. / std::reduce(newrow, newrow + nelem_per_row);
            cblas_dscal(nelem_per_row, isum, newrow, 1);

            gsl_interp_accel_reset(acc);
        }

        is_dia_matrix = false;
        ncols = osamp_matrix->getNCols();

        // Clean up
        gsl_interp_free(interp_cubic);
        gsl_interp_accel_free(acc);
        dia_matrix.reset();
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















