#include "core/matrix_helper.hpp"

#include <stdexcept>
#include <algorithm>

#ifdef USE_MKL_CBLAS
#include "mkl_lapacke.h"
#else
// These three lines somehow fix OpenBLAS compilation error on macos
#include <complex.h>
#define lapack_complex_float    float _Complex
#define lapack_complex_double   double _Complex
#include "lapacke.h"
#endif


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
        // dpotrf(CblasUpper, N, A, N); // the Cholesky factorization of a symmetric positive-definite matrix
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
    Resolution::Resolution(int nm, int ndia) : ndim(nm), ndiags(ndia)
    {
        size = ndim*ndiags;
        matrix = new double[size];
        buffer_mat = NULL;

        offsets = new int[ndiags];
        for (int i=ndiags/2, j=0; i > -(ndiags/2)-1; --i, ++j)
            offsets[j] = i;
    }

    Resolution::~Resolution()
    {
        delete [] offsets;
        delete [] matrix;

        if (buffer_mat != NULL)
            delete [] buffer_mat;
    }

    void Resolution::multiply(char SIDER, char TRANSR, const double* A, double *B, int N)
    {
        if (N != ndim)
            std::runtime_error("Resolution multiply operation dimension do not match!");

        std::for_each(B, B+N*N, [&](double &b) { b=0; });

        int transpose;

        if (TRANSR == 'N' || TRANSR == 'n')
            transpose = 1;
        else if (TRANSR == 'T' || TRANSR == 't')
            transpose = -1;
        else
            std::runtime_error("Resolution multiply transpose wrong character!");

        if (SIDER == 'L' || SIDER == 'l')
        {
            /* 
            if offset > 0 (upper off-diagonals), 
                remove initial offsets[i] elements from resolution matrix
                    when transposed remove last |offsets[i]|
                start from offset row in A to add row by row, 
                add to 0th row of B, end by offset
            if offset < 0 (lower off-diagonals), 
                remove last |offsets[i]| elements from resolution matrix
                    when transposed remove initial offsets[i]
                start from 0th row in A, but end by |offset|, 
                add to offset row in B
            */
            for (int d = 0; d < ndiags; ++d)
            {
                int off = transpose*offsets[d], 
                    nmult = ndim - abs(off),
                    Arow1, Brow1, od1;

                if (off >= 0)
                {
                    Arow1 = off;
                    Brow1 = 0;
                    od1   = Arow1*(1+transpose)/2;
                }
                else
                {
                    Arow1 = 0;
                    Brow1 = -off;
                    od1   = -off*(1-transpose)/2;
                }

                const double *Aslice = A+Arow1*N,
                             *dia_slice = matrix+(d*ndim+od1);
                double       *Bslice = B+Brow1*N;

                for (int nrow = 0; nrow < nmult; ++nrow)
                {
                    cblas_daxpy(ndim, dia_slice[nrow], Aslice, 1, Bslice, 1);
                    Aslice+=N;
                    Bslice+=N;
                }
            }
        }
        else if (SIDER == 'R' || SIDER == 'r')
        {
            /* 
            if offset > 0 (upper off-diagonals), 
                remove initial offsets[i] elements from resolution matrix
                    when transposed remove last |offsets[i]|
                start from 0th col in A, but end by |offset|, to add col by col, 
                add to offset col of B
            if offset < 0 (lower off-diagonals), 
                remove last |offsets[i]| elements from resolution matrix
                    when transposed remove initial offsets[i]
                start from offset col in A 
                add to 0th col in B, but end by |offset|
            */
            for (int d = 0; d < ndiags; ++d)
            {
                int off = transpose*offsets[d],
                    nmult = ndim - abs(off),
                    Acol1, Bcol1, od1;

                if (off < 0)
                {
                    Acol1 = -off;
                    Bcol1 = 0;
                    od1   = -off*(1-transpose)/2;
                }
                else
                {
                    Acol1 = 0;
                    Bcol1 = off;
                    od1   = Bcol1*(1+transpose)/2;
                }

                const double *odiag = matrix + (d*ndim+od1);

                for (int nrow = 0; nrow < ndim; ++nrow)
                {
                    const double  *Aslice = A + (Acol1+nrow*ndim);
                    double        *Bslice = B + (Bcol1+nrow*ndim);
                    
                    for (int ncol=0; ncol < nmult; ++ncol)
                        *(Bslice+ncol) += *(Aslice+ncol) * *(odiag+ncol);
                }
            }
        }
    }

    void Resolution::sandwich(double *inplace, int N)
    {
        // e.g. n1=724, ntotdiag=11
        // offsets: [ 5  4  3  2  1  0 -1 -2 -3 -4 -5]
        // when offsets[i]>0, remove initial offsets[i] elements from resomat.T[i]
        // when offsets[i]<0, remove last |offsets[i]| elements from resomat.T[i]

        if (N != ndim)
            std::runtime_error("Resolution sandwich operation dimension do not match!");
        
        if (buffer_mat == NULL)
            buffer_mat = new double[N*N];
        
        multiply('L', 'N', inplace, buffer_mat, N);
        multiply('R', 'T', buffer_mat, inplace, N);
    }

    /*
    void invertMatrixCholesky2(gsl_matrix *A)
    {
        int size = A->size1, status;

        gsl_vector *S = gsl_vector_alloc(size);
        gsl_linalg_cholesky_scale(A, S);

        status = gsl_linalg_cholesky_decomp2(A, S);

        if (status)
        {
            const char *err_msg = gsl_strerror(status);
            // fprintf(stderr, "ERROR in Cholesky Decomp: %s\n", err_msg);
            gsl_vector_free(S);
            throw std::runtime_error(err_msg);
        }

        gsl_linalg_cholesky_invert(A);
        gsl_linalg_cholesky_scale_apply(A, S);
        gsl_vector_free(S);
    }

    void invertMatrixCholesky(gsl_matrix *A)
    {
        int status = gsl_linalg_cholesky_decomp(A); 

        if (status)
        {
            const char *err_msg = gsl_strerror(status);
            // fprintf(stderr, "ERROR in Cholesky Decomp: %s\n", err_msg);

            throw std::runtime_error(err_msg);
        }

        status = gsl_linalg_cholesky_invert(A);

        if (status)
        {
            const char *err_msg = gsl_strerror(status);
            fprintf(stderr, "ERROR in Cholesky Invert: %s\n", err_msg);
            throw std::runtime_error(err_msg);
        }
    }

    void invertMatrixLU(gsl_matrix *A, gsl_matrix *Ainv)
    {
        int size = A->size1, signum, status;

        gsl_permutation *p = gsl_permutation_alloc(size);

        status = gsl_linalg_LU_decomp(A, p, &signum);

        if (status)
        {
            const char *err_msg = gsl_strerror(status);
            fprintf(stderr, "ERROR in LU Decomp: %s\n", err_msg);

            throw std::runtime_error(err_msg);
        }

        status = gsl_linalg_LU_invert(A, p, Ainv);

        gsl_permutation_free(p);
        
        if (status)
        {
            const char *err_msg = gsl_strerror(status);
            fprintf(stderr, "ERROR in LU Invert: %s\n", err_msg);
            throw std::runtime_error(err_msg);
        }
    }
    */
}
