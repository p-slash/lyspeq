#include "matrix_helper.hpp"

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_cblas.h>
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

double trace_dgemm(const gsl_matrix *A, const gsl_matrix *B)
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

// Assume A and B square symmetric matrices.
// No stride or whatsoever. Continous allocation
// Uses CBLAS dot product.
double trace_dsymm(const gsl_matrix *A, const gsl_matrix *B)
{
    int size = A->size1;
    size *= size;
    return cblas_ddot(size, A->data, 1, B->data, 1);  
}

double trace_ddiagmv(const gsl_matrix *A, const double *B)
{
    int size = A->size1;

    double result = 0.;

    for (int i = 0; i < size; i++)
        result += gsl_matrix_get(A, i, i) * B[i];

    return result;
}

double my_cblas_dsymvdot(const gsl_vector *v, const gsl_matrix *S)
{
    int size = v->size;

    double *temp_vector = new double[size], r;

    cblas_dsymv(CblasRowMajor, CblasUpper, \
                size, 1., S->data, size, \
                v->data, 1, \
                0, temp_vector, 1);

    r = cblas_ddot(size, v->data, 1, temp_vector, 1);

    delete [] temp_vector;

    return r;
}

void invert_matrix_cholesky_2(gsl_matrix *A)
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
        throw err_msg;
    }

    gsl_linalg_cholesky_invert(A);
    gsl_linalg_cholesky_scale_apply(A, S);
    gsl_vector_free(S);
}

void invert_matrix_cholesky(gsl_matrix *A)
{
    int status = gsl_linalg_cholesky_decomp(A); 

    if (status)
    {
        const char *err_msg = gsl_strerror(status);
        // fprintf(stderr, "ERROR in Cholesky Decomp: %s\n", err_msg);

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

void invert_matrix_LU(gsl_matrix *A, gsl_matrix *Ainv)
{
    int size = A->size1, signum, status;

    gsl_permutation *p = gsl_permutation_alloc(size);

    status = gsl_linalg_LU_decomp(A, p, &signum);

    if (status)
    {
        const char *err_msg = gsl_strerror(status);
        fprintf(stderr, "ERROR in LU Decomp: %s\n", err_msg);

        throw err_msg;
    }

    status = gsl_linalg_LU_invert(A, p, Ainv);

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
