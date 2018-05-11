#include "matrix_helper.hpp"

#include <gsl/gsl_linalg.h>

double trace_of_2matrices(const gsl_matrix *A, const gsl_matrix *B, int size)
{
    double result = 0;

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            result += gsl_matrix_get(A, i, j) * gsl_matrix_get(B, j, i);
        }
    }

    return result;
}

double trace_of_2matrices(const gsl_matrix *A, const double *noise, int size)
{
    double result = 0, nn;

    for (int i = 0; i < size; i++)
    {
        nn = noise[i];

        result += gsl_matrix_get(A, i, i) * nn * nn;
    }

    return result;
}

int invert_matrix_cholesky(gsl_matrix *A)
{
    int r;

    r = gsl_linalg_cholesky_decomp(A);

    gsl_linalg_cholesky_invert(A);

    return r;
}

