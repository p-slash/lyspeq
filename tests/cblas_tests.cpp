#include "core/matrix_helper.hpp"
#include <cmath>
#include <cstdio>
#include <algorithm>

#define N 8
#define NA 4
#define NR 7
#define Ndiag 5

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
    mxhelp::LAPACKE_InvertMatrixLU(smy_matrix_A, NA);
    mxhelp::printfMatrix(smy_matrix_A, NA, NA);

    // Test diamatrix multiplication
    /*double matrix_R[] = {1, 4, 5, 0, 0, 0, 0,
                         2, 8, 1, 2, 0, 0, 0,
                         9, 4, 1, 7, 7, 0, 0,
                         0, 1, 1, 3, 2, 7, 0,
                         0, 0, 1, 7, 7, 4, 4,
                         0, 0, 0, 2, 1, 4, 3,
                         0, 0, 0, 0, 7, 4, 4},*/
    double  dia_R[] = {-1, -1, 5, 2, 7, 7, 4,
                       -1,  4, 1, 7, 2, 4, 3,
                        1,  8, 1, 3, 7, 4, 4,
                        2,  4, 1, 7, 1, 4, -1,
                        9, 1, 1, 2, 7, -1, -1},
            matrix_BR[] = {3, -4, 7, 7, -5, -2, -4,
                         -2, -7, 4, -2, 0, -9, -7, 
                         6, 6, 5, -6, -3, -9, -4, 
                         3, 2, 4, -2, 8, -8, -4, 
                         1, 2, -5, 7, -1, 6, 1, 
                         3, -7, 0, 9, 1, -6, -2, 
                         -3, 5, 9, -6, 4, 1, 8},
            result_R[NR*NR];

    // mxhelp::printfMatrix(matrix_R, NR, NR);

    // printf("-----\n");
    // mxhelp::printfMatrix(matrix_BR, NR, NR);

    // printf("-----\n");

    mxhelp::Resolution rmat(NR, Ndiag);
    std::copy(&dia_R[0], &dia_R[0]+NR*Ndiag, rmat.matrix);
    // mxhelp::printfMatrix(rmat.matrix, Ndiag, NR);
    
    // printf("-----\n");
    printf("LN-----\n");
    rmat.multiply('L', 'N', matrix_BR, result_R, NR);
    mxhelp::printfMatrix(result_R, NR, NR);
    printf("LT-----\n");
    rmat.multiply('L', 'T', matrix_BR, result_R, NR);
    mxhelp::printfMatrix(result_R, NR, NR);
    printf("RN-----\n");
    rmat.multiply('R', 'N', matrix_BR, result_R, NR);
    mxhelp::printfMatrix(result_R, NR, NR);
    printf("RT-----\n");
    rmat.multiply('R', 'T', matrix_BR, result_R, NR);
    mxhelp::printfMatrix(result_R, NR, NR);

    return 0;
}
