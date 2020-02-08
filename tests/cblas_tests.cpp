#include "core/matrix_helper.hpp"
#include <gsl/gsl_cblas.h>
#include <cmath>

#define N 8
#define NA 4

int main()
{
    // Test cblas_ddot
    double A[N], B[N], C=0, r=-10, rel_err = 0;
    for (int i = 0; i < N; ++i)
    {
        A[i] = 0.1*i;
        B[i] = 2.5*i + 1;
        C += A[i] * B[i];
    }

    r = cblas_ddot(N, A, 1, B, 1);
    printf("True: %e\nCBLAS: %e\n", C, r);

    rel_err = fabs(C-r)/C;
    if (rel_err > 1e-8)
        fprintf(stderr, "ERROR cblas_ddot: True and CBLAS does not match. Rel error: %e.\n", rel_err);

    // Test cblas_dsymv
    double smy_matrix_A[] = { 4, 6, 7, 8,
                            6, 9, 2, 1,
                            7, 2, 0, 1,
                            8, 1, 1, 5};
    gsl_matrix_view gmv_symA = gsl_matrix_view_array(smy_matrix_A, NA, NA);

    double vector_B[] = {4, 5, 6, 7}, vector_R[NA];
    // gsl_vector_view gvv_B = gsl_vector_view_array(vector_B, NA);

    cblas_dsymv(CblasRowMajor, CblasUpper,
                NA, 0.5, smy_matrix_A, NA,
                vector_B, 1,
                0, vector_R, 1);

    for (int i = 0; i < NA; ++i)
        printf("%lf ", vector_R[i]);

    printf("\n");

    // Test cblas_dsymm
    double matrix_B[] = {3, 1, 9, 0,
                        4, 8, 8, 8,
                        4, 3, 2, 0,
                        5, 5, 9, 2};
    double result[NA*NA];

    cblas_dsymm( CblasRowMajor, CblasLeft, CblasUpper,
                 NA, NA, 1., smy_matrix_A, NA,
                 matrix_B, NA,
                 0, result, NA);
    for (int i = 0; i < NA; ++i)
    {
        for (int j = 0; j < NA; ++j)
            printf("%lf ", result[j + NA*i]);
        printf("\n");
    }

    // Test trace_ddiagmv
    printf("%lf\n", mxhelp::trace_ddiagmv(smy_matrix_A, vector_B, NA));

    // my_cblas_dsymvdot
    printf("%lf\n", mxhelp::my_cblas_dsymvdot(vector_B, smy_matrix_A, NA));
    
    // Test LU invert
    gsl_matrix *copy_A = gsl_matrix_alloc(NA, NA);
    gsl_matrix_memcpy(copy_A, &gmv_symA.matrix);
    
    mxhelp::invertMatrixLU(copy_A, &gmv_symA.matrix);
    mxhelp::printfMatrix(&gmv_symA.matrix);

    gsl_matrix_free(copy_A);

    return 0;
}
