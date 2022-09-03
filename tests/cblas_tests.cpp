#include "mathtools/matrix_helper.hpp"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cassert>

#define N 8
#define NA 4
#define NcolsSVD 6
#define NrowsSVD 5
const double relerr = 1e-6;

bool isClose(double a, double b)
{
    double diff = fabs(a-b), divi = std::max(fabs(a),fabs(b))/2;
    return (diff/divi) < relerr;
}

bool allClose(const double *a, const double *b, int size)
{
    bool result = true;
    for (int i = 0; i < size; ++i)
        result &= isClose(a[i], b[i]);
    return result;
}

const double
sym_matrix_A[] = {
    4, 6, 7, 8,
    6, 9, 2, 1,
    7, 2, 0, 1,
    8, 1, 1, 5 },
truth_cblas_dsymv_1[] = { 72.0, 44.0, 22.5, 39.0 },
vector_cblas_dsymv_b_1[] = {4, 5, 6, 7},
matrix_cblas_dsymm_B[] = {
    3, 1, 9, 0,
    4, 8, 8, 8,
    4, 3, 2, 0,
    5, 5, 9, 2},
truth_out_cblas_dsymm [] = {
    104., 113., 170., 64., 
    67., 89., 139., 74., 
    34., 28., 88., 18., 
    57., 44., 127., 18.},
truth_lu_inversion_A[] = {
    4.058442e-02,-8.603896e-02,3.214286e-01,-1.120130e-01,
    -8.603896e-02,2.224026e-01,-3.214286e-01,1.574675e-01,
    3.214286e-01,-3.214286e-01,7.857143e-01,-6.071429e-01,
    -1.120130e-01,1.574675e-01,-6.071429e-01,4.691558e-01},
matrix_for_SVD_A[] = {
    8.79, 6.11,  -9.15,  9.57, -3.49,  9.84,
    9.93, 6.91,  -7.93,  1.64,  4.02,  0.15,
    9.83, 5.04,  4.86,  8.83,   9.8,  -8.99,
    5.45, -0.27, 4.85,  0.74,  10.00, -6.02,
    3.16, 7.98,  3.01,  5.8,    4.27, -5.31},
truth_SVD_A[] = {
    -5.911424e-01, -3.975668e-01, -3.347897e-02, -4.297069e-01, -4.697479e-01, 2.933588e-01, 
    2.631678e-01, 2.437990e-01, -6.002726e-01, 2.361668e-01, -3.508914e-01, 5.762621e-01, 
    3.554302e-01, -2.223900e-01, -4.508393e-01, -6.858629e-01, 3.874446e-01, -2.085292e-02, 
    3.142644e-01, -7.534662e-01, 2.334497e-01, 3.318600e-01, 1.587356e-01, 3.790777e-01, 
    2.299383e-01, -3.635897e-01, -3.054757e-01, 1.649276e-01, -5.182574e-01, -6.525516e-01};

int test_cblas_ddot()
{
    double A[N], B[N], C=0, r=-10;
    for (int i = 0; i < N; ++i)
    {
        A[i] = 0.1*i;
        B[i] = 2.5*i + 1;
        C += A[i] * B[i];
    }

    r = cblas_ddot(N, A, 1, B, 1);

    if (isClose(C, r))
        return 0;

    fprintf(stderr, "ERROR cblas_ddot.\n");
    return 1;
}

int test_cblas_dsymv()
{
    double vector_R[NA];
    // gsl_vector_view gvv_B = gsl_vector_view_array(vector_B, NA);

    cblas_dsymv(CblasRowMajor, CblasUpper,
                NA, 0.5, sym_matrix_A, NA,
                vector_cblas_dsymv_b_1, 1,
                0, vector_R, 1);

    if (allClose(truth_cblas_dsymv_1, vector_R, NA))
        return 0;

    fprintf(stderr, "ERROR test_cblas_dsymv.\n");
    return 1;
}

int test_trace_ddiagmv_1()
{
    const double truth = 96.0;
    double result = mxhelp::trace_ddiagmv(sym_matrix_A,
        vector_cblas_dsymv_b_1, NA);

    if (isClose(result, truth))
        return 0;
    fprintf(stderr, "ERROR trace_ddiagmv_1.\n");
    return 1;
}

int test_trace_ddiagmv_2()
{
    const int ndim = 496;
    const double truth = -3522.0979676252823;
    std::vector<double> A, vec;
    int nrows, ncols;

    A = mxhelp::fscanfMatrix("tests/input/test_matrix_ddiagmv.txt", nrows, ncols);
    assert(nrows == ndim);
    assert(ncols == ndim);
    vec = mxhelp::fscanfMatrix("tests/input/test_diagonal_ddiagmv.txt", nrows, ncols);
    assert(nrows == ndim);
    assert(ncols == 1);

    double result =  mxhelp::trace_ddiagmv(A.data(), vec.data(), ndim);
    if (isClose(result, truth))
        return 0;
    fprintf(stderr, "ERROR trace_ddiagmv_2.\n");
    return 1;
}

int test_cblas_dsymm()
{
    double result[NA*NA];

    cblas_dsymm( CblasRowMajor, CblasLeft, CblasUpper,
                 NA, NA, 1., sym_matrix_A, NA,
                 matrix_cblas_dsymm_B, NA,
                 0, result, NA);

    if (allClose(truth_out_cblas_dsymm, result, NA*NA))
        return 0;
    fprintf(stderr, "ERROR cblas_dsymm.\n");
    return 1;
}

int test_my_cblas_dsymvdot()
{
    const double truth = 1832.;
    auto temp_vector = std::make_unique<double[]>(NA);
    double result = mxhelp::my_cblas_dsymvdot(
        vector_cblas_dsymv_b_1, sym_matrix_A,
        temp_vector.get(), NA);

    if (isClose(result, truth))
        return 0;
    fprintf(stderr, "ERROR my_cblas_dsymvdot.\n");
    return 1;
}

int test_LAPACKE_InvertMatrixLU()
{
    double invert_matrix[NA*NA];
    std::copy(sym_matrix_A, sym_matrix_A+NA*NA, invert_matrix);
    mxhelp::LAPACKE_InvertMatrixLU(invert_matrix, NA);
    if (allClose(truth_lu_inversion_A, invert_matrix, NA*NA))
        return 0;
    fprintf(stderr, "ERROR LAPACKE_InvertMatrixLU.\n");
    return 1;
}

int test_LAPACKE_SVD()
{
    // SVD tests
    double svals[NcolsSVD], svd_matrix[NcolsSVD*NrowsSVD];
    std::copy(
        matrix_for_SVD_A,
        matrix_for_SVD_A+NcolsSVD*NrowsSVD,
        svd_matrix);

    mxhelp::LAPACKE_svd(svd_matrix, svals, NcolsSVD, NrowsSVD);

    if (allClose(truth_SVD_A, svd_matrix, NcolsSVD*NrowsSVD))
        return 0;
    fprintf(stderr, "ERROR LAPACKE_svd.\n");
    return 1;
}

// Test resolution matrix
// Test OversampledMatrix
#define NrowsOversamp 7
#define NcolsOversamp 18
#define Nelemprow 5
const double
oversamp_values[] = {
    0.25, 0.29, 0.31, 0.29, 0.25, 0.25, 0.29,
    0.31, 0.29, 0.25, 0.25, 0.29, 0.31, 0.29,
    0.25, 0.25, 0.29, 0.31, 0.29, 0.25, 0.25,
    0.29, 0.31, 0.29, 0.25, 0.25, 0.29, 0.31,
    0.29, 0.25, 0.25, 0.29, 0.31, 0.29, 0.25},
oversample_multiplier_A[] = {
    0.42,0.08,0.15,0.73,0.72,0.25,0.5,0.05,0.87,0.11,0.33,0.74,0.1,0.63,
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
    0.73,0.43,0.63,0.22,0.71,0.86,0.32,0.93,1.0,0.27,0.37,0.83},
truth_oversample_left_multiplication[] = {
    3.3920e-01,6.3210e-01,4.3000e-01,7.6740e-01,6.8710e-01,6.7070e-01,
    6.1330e-01,5.2690e-01,6.1770e-01,5.7000e-01,8.2700e-01,7.8660e-01,
    2.6890e-01,6.3130e-01,4.4650e-01,9.2160e-01,7.6010e-01,8.0870e-01,
    6.0430e-01,5.8550e-01,2.2810e-01,6.2630e-01,6.2940e-01,7.1510e-01,
    4.6140e-01,7.2680e-01,3.5150e-01,7.5560e-01,5.3570e-01,8.3420e-01,
    3.2940e-01,4.3800e-01,5.2800e-01,9.1130e-01,7.8490e-01,6.8660e-01,
    7.9940e-01,6.9070e-01,3.0080e-01,6.3020e-01,5.8880e-01,6.5550e-01,
    6.8540e-01,6.0280e-01,5.8810e-01,6.6510e-01,5.9540e-01,7.3270e-01,
    6.1610e-01,4.7230e-01,7.3230e-01,7.0690e-01,5.8560e-01,6.7740e-01,
    9.0380e-01,7.6940e-01,5.1110e-01,6.3480e-01,5.5490e-01,7.5970e-01,
    9.2220e-01,4.9480e-01,6.0590e-01,7.7830e-01,5.1640e-01,8.1760e-01,
    1.0461e+00,6.6950e-01,7.0630e-01,5.6500e-01,5.2690e-01,6.4270e-01,
    8.5860e-01,9.0260e-01,6.9570e-01,8.0590e-01,4.1930e-01,5.8280e-01,
    8.4430e-01,7.6060e-01,6.1760e-01,9.0520e-01,5.4670e-01,7.8310e-01,
    1.2302e+00,7.4770e-01,7.4860e-01,6.0670e-01,2.8080e-01,5.0610e-01,
    8.3880e-01,8.8490e-01,5.2450e-01,7.3500e-01,3.6730e-01,5.2770e-01,
    9.7870e-01,8.8300e-01,7.2520e-01,7.6880e-01,6.6890e-01,6.2480e-01,
    1.0414e+00,9.4240e-01,3.2820e-01,7.7830e-01,6.2660e-01,6.4150e-01,
    8.2010e-01,8.9840e-01,5.1660e-01,6.6780e-01,3.8180e-01,5.9140e-01,
    7.9670e-01,8.3570e-01,7.4440e-01,6.7150e-01,6.9530e-01,6.0580e-01,
    9.2480e-01,7.8340e-01,3.3290e-01,7.6020e-01,8.1330e-01,7.2310e-01},
truth_oversample_right_multiplication[] = {
    7.957300e-1, 8.908750e-1, 8.636270e-1, 8.696630e-1, 8.714340e-1, 8.129250e-1, 8.460060e-1,
    7.305580e-1, 7.564950e-1, 8.064100e-1, 7.881360e-1, 7.973340e-1, 7.369770e-1, 8.335520e-1,
    8.233590e-1, 8.019310e-1, 8.716060e-1, 8.702020e-1, 8.909860e-1, 8.723660e-1, 8.694060e-1,
    9.303340e-1, 9.347490e-1, 9.398870e-1, 9.166780e-1, 1.035895e+0, 1.061225e+0, 9.702080e-1,
    1.030607e+0, 9.177060e-1, 9.105440e-1, 1.022288e+0, 1.121034e+0, 1.149119e+0, 1.002592e+0,
    9.338910e-1, 8.558460e-1, 9.856250e-1, 1.115734e+0, 1.053153e+0, 1.026597e+0, 1.017745e+0,
    9.148190e-1, 8.118510e-1, 9.423860e-1, 1.040852e+0, 1.003260e+0, 9.466060e-1, 9.853680e-1
};

int test_OversampledMatrix_multiplications()
{
    int r = 0;
    mxhelp::OversampledMatrix ovrmat(NrowsOversamp, Nelemprow, 2, 2);
    assert(NcolsOversamp == ovrmat.getNCols());
    std::copy(
        &oversamp_values[0],
        &oversamp_values[0]+NrowsOversamp*Nelemprow,
        ovrmat.matrix()
    );

    std::vector<double>
    mtrxB1(NrowsOversamp*NcolsOversamp, 0),
    mtrxB2(NrowsOversamp*NrowsOversamp, 0);

    // printf("Left multiplication-----\n");
    ovrmat.multiplyLeft(oversample_multiplier_A, mtrxB1.data());
    if (not allClose(truth_oversample_left_multiplication, mtrxB1.data(), mtrxB1.size()))
    {
        fprintf(stderr, "ERROR OversampledMatrix::multiplyLeft.\n");
        r += 1;
    }

    // printf("Right multiplication-----\n");
    ovrmat.multiplyRight(mtrxB1.data(), mtrxB2.data());
    if (not allClose(truth_oversample_right_multiplication, mtrxB2.data(), mtrxB2.size()))
    {
        fprintf(stderr, "ERROR OversampledMatrix::multiplyRight.\n");
        r += 1;
    }

    return r;
}

// Test DiaMatrix
#define NrowsDiag 7
#define NdiagDiag 5
const double
diamatrix_diagonals[] = {
    -1, -1, 5, 2, 7, 7, 4,
   -1,  4, 1, 7, 2, 4, 3,
    1,  8, 1, 3, 7, 4, 4,
    2,  4, 1, 7, 1, 4, -1,
    9, 1, 1, 2, 7, -1, -1},
diamatrix_rows[] = {
    5, 4, 1, 4, 5,
       2, 2, 8, 1, 2,
          9, 4, 1, 7, 7,
             1, 1, 3, 2, 7,
                1, 7, 7, 4, 4, 
                   2, 1, 4, 3, 2, 
                      7, 4, 4, 4, 7},
diamatrix_multiplier_B[] = {
    3, -4, 7, 7, -5, -2, -4,
    -2, -7, 4, -2, 0, -9, -7, 
    6, 6, 5, -6, -3, -9, -4, 
    3, 2, 4, -2, 8, -8, -4, 
    1, 2, -5, 7, -1, 6, 1, 
    3, -7, 0, 9, 1, -6, -2, 
    -3, 5, 9, -6, 4, 1, 8},
truth_diamatrix_LN_multiplication[] = {
    2.50e+01, -2.00e+00, 4.80e+01, -3.10e+01, -2.00e+01, -8.30e+01, -5.20e+01, 
    2.00e+00, -5.40e+01, 5.90e+01, -1.20e+01, 3.00e+00, -1.01e+02, -7.60e+01, 
    5.30e+01, -3.00e+01, 7.70e+01, 8.40e+01, 1.00e+00, -7.70e+01, -8.90e+01, 
    3.60e+01, -4.00e+01, 1.10e+01, 6.30e+01, 2.60e+01, -7.20e+01, -3.50e+01, 
    3.40e+01, 2.60e+01, 3.40e+01, 4.10e+01, 6.60e+01, -4.30e+01, -1.00e+00, 
    1.00e+01, -7.00e+00, 3.00e+01, 2.10e+01, 3.10e+01, -3.10e+01, 9.00e+00, 
    7.00e+00, 6.00e+00, 1.00e+00, 6.10e+01, 1.30e+01, 2.20e+01, 3.10e+01},
truth_diamatrix_LT_multiplication[] = {
    5.3e+01, 3.6e+01, 6.0e+01, -5.1e+01, -3.2e+01, -1.01e+02, -5.4e+01, 
    2.3e+01, -4.6e+01, 8.4e+01, -1.4e+01, -2.4e+01, -1.24e+02, -9.2e+01, 
    2.3e+01, -1.7e+01, 4.3e+01, 3.2e+01, -2.1e+01, -3.0e+01, -3.4e+01, 
    6.0e+01, 3.4e+01, 2.0e+01, 1.5e+01, -2.0e+00, -7.5e+01, -5.1e+01, 
    3.7e+01, 8.8e+01, 7.1e+01, -3.0e+01, 1.7e+01, -3.6e+01, 2.5e+01, 
    2.5e+01, 1.4e+01, 4.4e+01, 2.6e+01, 7.2e+01, -5.2e+01, 0.0e+00, 
    1.0e+00, 7.0e+00, 1.6e+01, 3.1e+01, 1.5e+01, 1.0e+01, 3.0e+01},
truth_diamatrix_RN_multiplication[] = {
    5.80e+01, 1.50e+01, 2.00e+01, 2.30e+01, -2.00e+00, 5.00e+00, -4.20e+01, 
    2.00e+01, -5.00e+01, -1.50e+01, -1.00e+01, -3.40e+01, -7.80e+01, -5.50e+01, 
    6.30e+01, 8.60e+01, 3.20e+01, -1.00e+01, -3.50e+01, -1.06e+02, -5.50e+01, 
    4.30e+01, 4.20e+01, 2.70e+01, 6.60e+01, 4.40e+01, -3.00e+01, -8.00e+00, 
    -4.00e+01, 7.00e+00, 8.00e+00, -5.00e+00, -1.50e+01, 7.30e+01, 1.80e+01, 
    -1.10e+01, -3.50e+01, 1.80e+01, 8.00e+00, 5.00e+00, 3.50e+01, -2.20e+01, 
    8.80e+01, 5.80e+01, -3.00e+00, 8.50e+01, 1.36e+02, 1.00e+01, 5.10e+01},
truth_diamatrix_RT_multiplication[] = {
    2.20e+01, -5.00e+00, 3.20e+01, 0.00e+00, -3.00e+00, -1.10e+01, -5.90e+01, 
    -1.00e+01, -6.00e+01, -5.60e+01, -7.20e+01, -7.40e+01, -6.10e+01, -6.40e+01, 
    5.50e+01, 5.30e+01, 2.00e+01, -7.60e+01, -1.10e+02, -6.30e+01, -7.30e+01, 
    3.10e+01, 2.20e+01, 8.10e+01, -4.00e+01, -2.00e+00, -4.00e+01, 8.00e+00, 
    -1.60e+01, 2.70e+01, 5.40e+01, 5.80e+01, 6.50e+01, 4.00e+01, 2.10e+01, 
    -2.50e+01, -3.20e+01, 6.90e+01, -2.00e+01, 3.80e+01, -1.10e+01, -2.50e+01, 
    6.20e+01, 3.10e+01, -1.20e+01, 1.10e+01, 3.10e+01, 2.00e+01, 6.40e+01};

int test_DiaMatrix_multiplications()
{
    int r = 0;
    double result_R[NrowsDiag*NrowsDiag];
    mxhelp::DiaMatrix diarmat(NrowsDiag, NdiagDiag);
    std::copy(
        &diamatrix_diagonals[0],
        &diamatrix_diagonals[0]+NrowsDiag*NdiagDiag,
        diarmat.matrix()
    );

    // printf("LN-----\n");
    diarmat.multiply('L', 'N', diamatrix_multiplier_B, result_R);
    if (not allClose(truth_diamatrix_LN_multiplication, result_R, NrowsDiag*NrowsDiag))
    {
        fprintf(stderr, "ERROR DiaMatrix::multiply('L', 'N').\n");
        r += 1;
    }

    // printf("LT-----\n");
    diarmat.multiply('L', 'T', diamatrix_multiplier_B, result_R);
    if (not allClose(truth_diamatrix_LT_multiplication, result_R, NrowsDiag*NrowsDiag))
    {
        fprintf(stderr, "ERROR DiaMatrix::multiply('L', 'T').\n");
        r += 1;
    }

    // printf("RN-----\n");
    diarmat.multiply('R', 'N', diamatrix_multiplier_B, result_R);
    if (not allClose(truth_diamatrix_RN_multiplication, result_R, NrowsDiag*NrowsDiag))
    {
        fprintf(stderr, "ERROR DiaMatrix::multiply('R', 'N').\n");
        r += 1;
    }

    // printf("RT-----\n");
    diarmat.multiply('R', 'T', diamatrix_multiplier_B, result_R);
    if (not allClose(truth_diamatrix_RT_multiplication, result_R, NrowsDiag*NrowsDiag))
    {
        fprintf(stderr, "ERROR DiaMatrix::multiply('R', 'T').\n");
        r += 1;
    }

    return r;
}

int test_DiaMatrix_getRow()
{
    int r = 0;
    std::vector<double> testrow(NdiagDiag);

    mxhelp::DiaMatrix diarmat(NrowsDiag, NdiagDiag);
    std::copy(
        &diamatrix_diagonals[0],
        &diamatrix_diagonals[0]+NrowsDiag*NdiagDiag,
        diarmat.matrix()
    );

    for (int row = 0; row < NrowsDiag; ++row)
    {
        diarmat.getRow(row, testrow);
        const double *truth_row = &diamatrix_rows[row*NdiagDiag];
        
        if (not allClose(truth_row, testrow.data(), testrow.size()))
        {
            fprintf(stderr, "ERROR DiaMatrix::getRow(%d).\n", row);
            r += 1;
        }
    }

    return r;
}

int main()
{
    int r = 0;
    r += test_cblas_ddot();
    r += test_cblas_dsymv();
    r += test_cblas_dsymm();
    r += test_trace_ddiagmv_1();
    r += test_trace_ddiagmv_2();
    r += test_my_cblas_dsymvdot();
    r += test_LAPACKE_InvertMatrixLU();
    
    r += test_OversampledMatrix_multiplications();

    // mxhelp::Resolution overmat2(Nrows, Nelemprow, 2, 2);
    // std::copy(&valss[0], &valss[0]+Nrows*Nelemprow, overmat2.values);
    // mxhelp::Resolution overmatSub(&overmat2, 1, 3);
    // overmat2.fprintfMatrix("testOrg.txt");
    // overmatSub.fprintfMatrix("testCopy.txt");
    r += test_DiaMatrix_multiplications();
    r += test_DiaMatrix_getRow();
    r += test_LAPACKE_SVD();

    // mxhelp::DiaMatrix diarmat2(300, 11);
    // mxhelp::Resolution rmat(300, 11);
    // std::vector<double> vel(300);
    // vel.clear();
    // double a_kms=20., R_kms=20.;
    // for (int i = 0; i < 300; ++i)
    //     vel.push_back((i - 300/2.)*a_kms);
    // for (auto v : vel)
    //     printf("v%.1e==", v);

    // diarmat2.constructGaussian(vel.data(), R_kms, a_kms);
    // std::copy(diarmat2.matrix(), diarmat2.matrix()+300*11, rmat.matrix());
    // rmat.oversample(3, 1.0);
    // diarmat2.fprintfMatrix("tests/output/diamat.txt");
    // rmat.fprintfMatrix("tests/output/osampmat.txt");

    
    return r;
}



  