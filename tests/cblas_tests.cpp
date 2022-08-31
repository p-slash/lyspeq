#include "mathtools/matrix_helper.hpp"
#include <cmath>
#include <cstdio>
#include <algorithm>

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
    auto temp_vector = std::make_unique<double[]>(NA);
    printf("%lf\n", mxhelp::my_cblas_dsymvdot(vector_B, smy_matrix_A,
        temp_vector.get(), NA));
    
    // Test LU invert
    mxhelp::LAPACKE_InvertMatrixLU(smy_matrix_A, NA);
    mxhelp::printfMatrix(smy_matrix_A, NA, NA);

    // Oversampled matrix tests
    #define Nrows 7
    #define Ncols 18
    #define Nelemprow 5
    mxhelp::OversampledMatrix ovrmat(Nrows, Nelemprow, 2, 2);
    double valss[] = {0.25, 0.29, 0.31, 0.29, 0.25, 0.25, 0.29, 0.31, 0.29, 0.25, 0.25, 
        0.29, 0.31, 0.29, 0.25, 0.25, 0.29, 0.31, 0.29, 0.25, 0.25, 0.29, 0.31, 0.29, 
        0.25, 0.25, 0.29, 0.31, 0.29, 0.25, 0.25, 0.29, 0.31, 0.29, 0.25};
    double matrxA[] = {0.42,0.08,0.15,0.73,0.72,0.25,0.5,0.05,0.87,0.11,0.33,0.74,0.1,0.63,
        0.67,0.67,0.0,0.84,0.38,0.76,0.74,0.21,0.4,0.59,0.72,0.11,0.48,0.33,0.91,0.3,0.26,
        0.62,0.1,0.57,0.81,0.51,0.05,0.67,0.24,0.89,0.32,0.87,0.31,0.27,0.32,0.19,0.72,0.2,
        0.33,0.34,0.05,0.91,0.8,0.68,0.15,0.1,0.15,0.14,0.36,0.56,0.21,0.72,0.17,0.76,0.06,
        0.94,0.03,0.09,0.05,0.98,0.93,0.5,0.26,0.62,0.24,0.83,0.75,0.02,0.49,0.76,0.45,0.67,
        0.96,0.72,0.23,0.65,0.88,0.09,0.03,0.38,0.88,0.17,0.13,0.21,0.5,0.6,0.09,0.31,0.13,
        0.75,0.08,0.21,0.11,0.01,0.33,0.48,0.21,0.22,0.85,0.59,0.05,0.18,0.27,0.62,0.58,0.5,
        0.18,0.25,0.07,0.91,0.54,0.49,0.53,0.93,0.98,0.76,0.33,0.4,0.44,0.4,0.44,0.92,0.8,0.26,
        0.49,0.44,0.7,0.18,0.42,0.5,0.12,0.61,0.71,0.95,0.48,0.75,0.24,0.76,0.18,0.07,0.5,0.37,
        0.96,0.3,0.43,0.63,0.95,0.04,0.87,0.32,0.03,0.03,0.92,0.46,0.54,0.33,0.6,0.28,0.63,0.18,
        0.28,0.88,0.19,0.69,0.87,0.74,0.92,0.16,0.23,0.26,0.72,0.56,0.56,0.57,0.52,0.94,0.83,0.51,
        0.16,0.96,0.43,0.57,0.97,0.7,0.01,0.04,0.0,0.37,0.09,0.89,0.7,0.4,0.04,0.13,0.67,0.82,0.42,
        0.31,0.02,0.47,0.68,0.59,0.78,0.76,0.29,0.65,0.89,0.6,0.41,0.91,0.11,0.62,0.34,0.88,0.5,
        0.75,0.98,0.45,0.97,0.54,0.14,0.99,0.49,0.48,0.42,0.77,0.11,0.36,0.29,0.07,0.85,0.06,0.96,
        0.41,0.42,0.35,0.9,0.66,0.09,0.65,0.89,0.48,0.94,0.31,0.09,0.36,0.43,0.17,0.9,0.91,0.52,
        0.35,0.52,0.42,0.16,0.95,0.12,0.21,0.53,0.29,0.61,0.85,0.67,0.42,0.11,0.96,0.53,0.78,
        0.37,0.29,0.12,0.79,0.73,0.5,0.24,0.04,0.21,0.85,0.03,0.73,0.64,0.41,0.42,0.34,0.13,
        0.36,0.29,0.69,0.53,0.13,0.64,0.07,0.66,0.99,0.83,0.51,0.07,0.36,0.64,0.26,0.76,0.96,
        0.73,0.43,0.63,0.22,0.71,0.86,0.32,0.93,1.0,0.27,0.37,0.83};
    double mtrxB1[Nrows*Ncols], mtrxB2[Nrows*Nrows];
    std::copy(&valss[0], &valss[0]+Nrows*Nelemprow, ovrmat.matrix());

    printf("Left multiplication-----\n");
    ovrmat.multiplyLeft(matrxA, mtrxB1);
    mxhelp::printfMatrix(mtrxB1, Nrows, Ncols);
    printf("Right multiplication-----\n");
    ovrmat.multiplyRight(mtrxB1, mtrxB2);
    mxhelp::printfMatrix(mtrxB2, Nrows, Nrows);

    // mxhelp::Resolution overmat2(Nrows, Nelemprow, 2, 2);
    // std::copy(&valss[0], &valss[0]+Nrows*Nelemprow, overmat2.values);
    // mxhelp::Resolution overmatSub(&overmat2, 1, 3);
    // overmat2.fprintfMatrix("testOrg.txt");
    // overmatSub.fprintfMatrix("testCopy.txt");

    // Dia matrix tests
    #define NR 7
    #define Ndiag 5
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

    mxhelp::DiaMatrix diarmat(NR, Ndiag);
    std::copy(&dia_R[0], &dia_R[0]+NR*Ndiag, diarmat.matrix());
    
    // printf("-----\n");
    printf("LN-----\n");
    diarmat.multiply('L', 'N', matrix_BR, result_R);
    mxhelp::printfMatrix(result_R, NR, NR);
    printf("LT-----\n");
    diarmat.multiply('L', 'T', matrix_BR, result_R);
    mxhelp::printfMatrix(result_R, NR, NR);
    printf("RN-----\n");
    diarmat.multiply('R', 'N', matrix_BR, result_R);
    mxhelp::printfMatrix(result_R, NR, NR);
    printf("RT-----\n");
    diarmat.multiply('R', 'T', matrix_BR, result_R);
    mxhelp::printfMatrix(result_R, NR, NR);

    // Unit test getRow of dia matrix
    std::vector<double> testrow(Ndiag);
    diarmat.getRow(0, testrow);
    for (int i = 0; i < Ndiag; ++i)
        printf("%.0f ", testrow[i]);
    printf("\n");

    diarmat.getRow(1, testrow);
    for (int i = 0; i < Ndiag; ++i)
        printf("%.0f ", testrow[i]);
    printf("\n");

    diarmat.getRow(3, testrow);
    for (int i = 0; i < Ndiag; ++i)
        printf("%.0f ", testrow[i]);
    printf("\n");

    diarmat.getRow(5, testrow);
    for (int i = 0; i < Ndiag; ++i)
        printf("%.0f ", testrow[i]);
    printf("\n");
    diarmat.getRow(6, testrow);
    for (int i = 0; i < Ndiag; ++i)
        printf("%.0f ", testrow[i]);
    printf("\n");

    // mxhelp::Resolution rmat(NR, Ndiag);
    // double v[NR], a_kms=20., R_kms=20.;
    // for (int i = 0; i < NR; ++i)
    //     v[i] = int(i - NR/2)*a_kms;

    // diarmat.constructGaussian(v, R_kms, a_kms);
    // std::copy(diarmat.matrix, diarmat.matrix+NR*Ndiag, rmat.values);
    // rmat.oversample(3, 1.0);
    // diarmat.fprintfMatrix("tests/output/diamat.txt");
    // rmat.fprintfMatrix("tests/output/osampmat.txt");

    // SVD tests
    double SVD_A[] = {
        8.79, 6.11,  -9.15,  9.57, -3.49,  9.84,
        9.93, 6.91,  -7.93,  1.64,  4.02,  0.15,
        9.83, 5.04,  4.86,  8.83,   9.8,   -8.99,
        5.45, -0.27, 4.85,  0.74,  10.00,  -6.02,
        3.16, 7.98,  3.01,  5.8,   4.27, -5.31
    };
    double svals[6];
    // ncols = 6, nrows =  5
    mxhelp::LAPACKE_svd(SVD_A, svals, 6, 5);
    mxhelp::printfMatrix(SVD_A, 5, 6);
    return 0;
}



  