#include "matrix_helper.hpp"

#include <gsl/gsl_linalg.h>

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

        // if (status == GSL_EDOM)
        //     fprintf(stderr, "The matrix is not positive definite.\n");

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
    
    toWrite = open_file(fname, "w");

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
