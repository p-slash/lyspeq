#ifndef MATRIX_HELPER_H
#define MATRIX_HELPER_H

#include <gsl/gsl_matrix.h>
#include <cstdio>

double trace_of_2matrices(const gsl_matrix *A, const gsl_matrix *B);

/* optimized for diagonal matrix B */
double trace_of_2matrices(const gsl_matrix *A, const double *noise);

void invert_matrix_cholesky(gsl_matrix *A);
void invert_matrix_LU(const gsl_matrix *A, gsl_matrix *Ainv);

void printf_matrix(const gsl_matrix *m);
void fprintf_matrix(const char *fname, const gsl_matrix *m);

#endif
