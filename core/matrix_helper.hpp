#ifndef MATRIX_HELPER_H
#define MATRIX_HELPER_H

#include <gsl/gsl_matrix.h>

double trace_of_2matrices(const gsl_matrix *A, const gsl_matrix *B, int size);

/* optimized for diagonal matrix B */
double trace_of_2matrices(const gsl_matrix *A, const double *noise, int size);

int invert_matrix_cholesky(gsl_matrix *A);

void printf_matrix(const gsl_matrix *m, int size);

#endif
