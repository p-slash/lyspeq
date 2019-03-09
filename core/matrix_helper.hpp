#ifndef MATRIX_HELPER_H
#define MATRIX_HELPER_H

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

void copy_upper2lower(gsl_matrix *A);

double trace_dgemm(const gsl_matrix *A, const gsl_matrix *B);
double trace_dsymm(const gsl_matrix *A, const gsl_matrix *B);
/* optimized for diagonal matrix B */
double trace_ddiagmv(const gsl_matrix *A, const double *B);

/* vT . S . v */
double my_cblas_dsymvdot(const gsl_vector *v, const gsl_matrix *S);

void invert_matrix_cholesky_2(gsl_matrix *A);
void invert_matrix_cholesky(gsl_matrix *A);
void invert_matrix_LU(gsl_matrix *A, gsl_matrix *Ainv);

void printf_matrix(const gsl_matrix *m);
void fprintf_matrix(const char *fname, const gsl_matrix *m);

#endif
