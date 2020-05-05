#include "core/matrix_helper.hpp"

#include <stdexcept>

#ifdef USE_MKL_CBLAS
#include "mkl_lapacke.h"
#else
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
        lapack_int LIN = N, ipiv, info;
        // Factorize A
        // the LU factorization of a general m-by-n matrix.
        info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, LIN, LIN, A, LIN, &ipiv);
        
        LAPACKErrorHandle("ERROR in LU Decomp", info);

        info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, LIN, A, LIN, &ipiv); //, work, lwork, info);
        LAPACKErrorHandle("ERROR in LU Decomp", info);

        // dpotrf(CblasUpper, N, A, N); // the Cholesky factorization of a symmetric positive-definite matrix
    }

    void printfMatrix(const double *A, int nrows, int ncols)
    {
        for (int i = 0; i < nrows; ++i)
        {
            for (int j = 0; j < ncols; ++j)
                printf("%.6le ", *(A+i+nrows*j));
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
                fprintf(toWrite, "%20le ", *(A+i+nrows*j));
            fprintf(toWrite, "\n");
        }

        fclose(toWrite);
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
