#ifndef MATRIX_HELPER_H
#define MATRIX_HELPER_H

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

namespace mxhelp
{
    // Copy upper triangle of matrix A to its lower triangle
    // Call gsl_matrix_get and set, passing -DHAVE_INLINE to precompiler make them inline (faster)
    void copyUpperToLower(gsl_matrix *A);
    void copyUpperToLower(double *A, int size);

    void vector_add(double *target, const double *source, int size);
    void vector_sub(double *target, const double *source, int size);

    // Trace of A.B
    // Both are assumed to general and square
    // Call gsl_matrix_get, passing -DHAVE_INLINE to precompiler make them inline (faster)
    double trace_dgemm(const double *A, const double *B, int N);

    // Trace of A.B
    // Assumes A and B square matrices, and at least one to be symmetric.
    // No stride or whatsoever. Continous allocation
    // Uses CBLAS dot product.
    double trace_dsymm(const double *A, const double *B, int N);

    // Trace of A.B
    // Assumes A square matrix, B is diagonal. 
    // Only pass diagonal terms of B
    // If A is NxN, then B is N
    // Uses CBLAS dot product.
    double trace_ddiagmv(const double *A, const double *B, int N);

    // vT . S . v
    // Assumes S is square symmetric matrix
    double my_cblas_dsymvdot(const double *v, const double *S, int N);

    // In-place invert A
    // Apply scale as gsl_linalg_cholesky_scale
    // Assuming A is NxN, allocates and frees a vector of size N
    void invertMatrixCholesky2(gsl_matrix *A);

    // In-place invert A with Cholesky
    // No scaling
    void invertMatrixCholesky(gsl_matrix *A);

    // Invert A into Ainv using LU decomposition
    // A is changed, do not use it again
    // Assuming A is NxN, allocates and frees a gsl_permutation of size N
    void invertMatrixLU(gsl_matrix *A, gsl_matrix *Ainv);

    void printfMatrix(const gsl_matrix *m);
    void fprintfMatrix(const char *fname, const gsl_matrix *m);
}

#endif
