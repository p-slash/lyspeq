#include "matrix_helper.hpp"

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_permutation.h>

void copy_upper2lower(gsl_matrix *A)
{
    double temp;
    int size = A->size1;

    for (int i = 1; i < size; i++)
    {
        for (int j = 0; j < i; j++)
        {
            temp = gsl_matrix_get(A, j, i);
            gsl_matrix_set(A, i, j, temp);
        }
    }
}

double trace_of_2matrices(const gsl_matrix *A, const gsl_matrix *B)
{
    int size = A->size1;

    double result = 0.;

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            result += gsl_matrix_get(A, i, j) * gsl_matrix_get(B, j, i);
        }
    }

    return result;
}

double trace_of_2_sym_matrices(const gsl_matrix *A, const gsl_matrix *B)
{
    int size = A->size1;

    double result = 0.;

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            result += gsl_matrix_get(A, i, j) * gsl_matrix_get(B, i, j);
        }
    }

    return result;   
}
double trace_of_2matrices(const gsl_matrix *A, const double *noise)
{
    int size = A->size1;

    double result = 0., nn;

    for (int i = 0; i < size; i++)
    {
        nn = noise[i];

        result += gsl_matrix_get(A, i, i) * nn * nn;
    }

    return result;
}

void invert_matrix_cholesky(gsl_matrix *A)
{
    int status = gsl_linalg_cholesky_decomp(A); 

    if (status)
    {
        const char *err_msg = gsl_strerror(status);
        fprintf(stderr, "ERROR in Cholesky Decomp: %s\n", err_msg);

        throw err_msg;
    }

    status = gsl_linalg_cholesky_invert(A);

    if (status)
    {
        const char *err_msg = gsl_strerror(status);
        fprintf(stderr, "ERROR in Cholesky Invert: %s\n", err_msg);
        throw err_msg;
    }
}

void invert_matrix_LU(const gsl_matrix *A, gsl_matrix *Ainv)
{
    int size = A->size1, signum, status;

    gsl_permutation *p     = gsl_permutation_alloc(size);
    gsl_matrix      *Acopy = gsl_matrix_alloc(size, size);
    gsl_matrix_memcpy(Acopy, A);

    status = gsl_linalg_LU_decomp(Acopy, p, &signum);

    if (status)
    {
        const char *err_msg = gsl_strerror(status);
        fprintf(stderr, "ERROR in LU Decomp: %s\n", err_msg);

        throw err_msg;
    }

    status = gsl_linalg_LU_invert(Acopy, p, Ainv);

    gsl_matrix_free(Acopy);
    gsl_permutation_free(p);
    
    if (status)
    {
        const char *err_msg = gsl_strerror(status);
        fprintf(stderr, "ERROR in LU Invert: %s\n", err_msg);
        throw err_msg;
    }
}

void printf_matrix(const gsl_matrix *m)
{
    int nrows = m->size1, ncols = m->size2;

    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            printf("%le ", gsl_matrix_get(m, i, j));
        }
        printf("\n");
    }
}

void fprintf_matrix(const char *fname, const gsl_matrix *m)
{
    FILE *toWrite;
    
    toWrite = fopen(fname, "w");

    int nrows = m->size1, ncols = m->size2;

    fprintf(toWrite, "%d %d\n", nrows, ncols);
    
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            fprintf(toWrite, "%le ", gsl_matrix_get(m, i, j));
        }
        fprintf(toWrite, "\n");
    }

    fclose(toWrite);
}
