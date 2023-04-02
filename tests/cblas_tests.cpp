#include "mathtools/matrix_helper.hpp"
#include "tests/test_utils.hpp"

#include <unordered_map>
#include <string>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cassert>

const double
sym_matrix_A[] = {
    4, 6, 7, 8,
    6, 9, 2, 1,
    7, 2, 0, 1,
    8, 1, 1, 5 },
diagonal_of_A[] = {
    4, 9, 0, 5,
    6, 2, 1, -1,
    7, 1, -1, -1,
    8, -1, -1, -1},
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

const int
NA = 4,
NcolsSVD = 6,
NrowsSVD = 5;

int test_cblas_ddot()
{
    const int N = 8;
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
    printValues(C, r);
    return 1;
}

int test_cblas_dsymv_1()
{
    double vector_R[NA];

    cblas_dsymv(CblasRowMajor, CblasUpper,
                NA, 0.5, sym_matrix_A, NA,
                vector_cblas_dsymv_b_1, 1,
                0, vector_R, 1);

    if (allClose(truth_cblas_dsymv_1, vector_R, NA))
        return 0;

    fprintf(stderr, "ERROR test_cblas_dsymv_1.\n");
    printMatrices(truth_cblas_dsymv_1, vector_R, NA, 1);
    return 1;
}

int test_cblas_dsymv_2()
{
    const int ndim = 496;
    std::vector<double> A, vec, truth_out_cblas_dsymv;
    int nrows, ncols;

    const std::string
    fname_matrix = std::string(SRCDIR) + "/tests/input/test_symmatrix.txt",
    fname_vector = std::string(SRCDIR) + "/tests/input/test_diagonal_ddiagmv.txt",
    fname_truth  = std::string(SRCDIR) + "/tests/truth/truth_dsymv.txt";

    A = mxhelp::fscanfMatrix(fname_matrix.c_str(), nrows, ncols);
    assert(nrows == ndim);
    assert(ncols == ndim);
    vec = mxhelp::fscanfMatrix(fname_vector.c_str(), nrows, ncols);
    assert(nrows == ndim);
    assert(ncols == 1);
    truth_out_cblas_dsymv = mxhelp::fscanfMatrix(fname_truth.c_str(), nrows, ncols);
    assert(nrows == ndim);
    assert(ncols == 1);

    std::vector<double> cblas_dsymv_result(ndim);
    cblas_dsymv(CblasRowMajor, CblasUpper,
                ndim, 1, A.data(), ndim,
                vec.data(), 1,
                0, cblas_dsymv_result.data(), 1);

    if (allClose(truth_out_cblas_dsymv.data(), cblas_dsymv_result.data(), ndim))
        return 0;
    fprintf(stderr, "ERROR test_cblas_dsymv_2.\n");
    // printMatrices(truth_out_cblas_dsymv.data(), cblas_dsymv_result.data(), ndim, 1);
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
    printMatrices(truth_out_cblas_dsymm, result, NA, NA);
    return 1;
}

int test_getDiagonal()
{
    double v[NA];
    int r = 0;
    for (int d = 0; d < NA; ++d)
    {
        mxhelp::getDiagonal(sym_matrix_A, NA, d, v);
        const double *truth_diag = &diagonal_of_A[d*NA];
        
        if (not allClose(truth_diag, v, NA-d))
        {
            fprintf(stderr, "ERROR getDiagonal(%d).\n", d);
            printMatrices(truth_diag, v, NA-d, 1);
            r += 1;
        }
    }

    return r;
}

int test_trace_ddiagmv_1()
{
    const double truth = 96.0;
    double result = mxhelp::trace_ddiagmv(sym_matrix_A,
        vector_cblas_dsymv_b_1, NA);

    if (isClose(result, truth))
        return 0;
    fprintf(stderr, "ERROR trace_ddiagmv_1.\n");
    printValues(truth, result);
    return 1;
}

int test_trace_ddiagmv_2()
{
    const int ndim = 496;
    const double truth = -3522.06;
    std::vector<double> A, vec;
    int nrows, ncols;

    const std::string
    fname_matrix = std::string(SRCDIR) + "/tests/input/test_matrix_ddiagmv.txt",
    fname_vector = std::string(SRCDIR) + "/tests/input/test_diagonal_ddiagmv.txt";
    A = mxhelp::fscanfMatrix(fname_matrix.c_str(), nrows, ncols);
    assert(nrows == ndim);
    assert(ncols == ndim);
    vec = mxhelp::fscanfMatrix(fname_vector.c_str(), nrows, ncols);
    assert(nrows == ndim);
    assert(ncols == 1);

    double result = mxhelp::trace_ddiagmv(A.data(), vec.data(), ndim);
    if (isClose(result, truth))
        return 0;
    fprintf(stderr, "ERROR trace_ddiagmv_2.\n");
    printValues(truth, result);
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
    printValues(truth, result);
    return 1;
}

int test_LAPACKE_InvertMatrixLU_1()
{
    double invert_matrix[NA*NA];
    std::copy(sym_matrix_A, sym_matrix_A+NA*NA, invert_matrix);
    mxhelp::LAPACKE_InvertMatrixLU(invert_matrix, NA);
    if (allClose(truth_lu_inversion_A, invert_matrix, NA*NA))
        return 0;
    fprintf(stderr, "ERROR LAPACKE_InvertMatrixLU.\n");
    printMatrices(truth_lu_inversion_A, invert_matrix, NA, NA);
    return 1;
}

int test_LAPACKE_InvertMatrixLU_2()
{
    const int ndim = 496;
    std::vector<double> A, truth_out_inverse;
    int nrows, ncols;

    const std::string
    fname_matrix = std::string(SRCDIR) + "/tests/input/test_symmatrix.txt",
    fname_truth  = std::string(SRCDIR) + "/tests/truth/truth_inverse_matrix.txt";

    A = mxhelp::fscanfMatrix(fname_matrix.c_str(), nrows, ncols);
    assert(nrows == ndim);
    assert(ncols == ndim);

    truth_out_inverse = mxhelp::fscanfMatrix(fname_truth.c_str(), nrows, ncols);
    assert(nrows == ndim);
    assert(ncols == ndim);

    mxhelp::LAPACKE_InvertMatrixLU(A.data(), ndim);

    if (allClose(A.data(), truth_out_inverse.data(), ndim*ndim))
        return 0;

    fprintf(stderr, "ERROR test_LAPACKE_InvertMatrixLU_2.\n");
    // printMatrices(truth_out_inverse.data(), A.data(), ndim, ndim);
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
    printMatrices(truth_SVD_A, svd_matrix, NrowsSVD, NcolsSVD);
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
        printMatrices(truth_oversample_left_multiplication,
            mtrxB1.data(), NrowsOversamp, NcolsOversamp);
        r += 1;
    }

    // printf("Right multiplication-----\n");
    ovrmat.multiplyRight(mtrxB1.data(), mtrxB2.data());
    if (not allClose(truth_oversample_right_multiplication, mtrxB2.data(), mtrxB2.size()))
    {
        fprintf(stderr, "ERROR OversampledMatrix::multiplyRight.\n");
        printMatrices(truth_oversample_right_multiplication,
            mtrxB2.data(), NrowsOversamp, NcolsOversamp);
        r += 1;
    }

    return r;
}

// Test DiaMatrix
#define NrowsDiag 7
#define NdiagDiag 5
const double
diamatrix_diagonals[] = {
    0, 0, 5, 2, 7, 7, 4,
    0, 4, 1, 7, 2, 4, 3,
    1, 8, 1, 3, 7, 4, 4,
    2, 4, 1, 7, 1, 4, 0,
    9, 1, 1, 2, 7, 0, 0},
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
diamatrix_multiplier_Sym[] = {
    -2, -5,  5,  3,  8,  5, -5,
    -5, -4, -3, 11,  2, -1,  9,
     5, -3, -4,  1,  5,  9,  5,
     3, 11,  1,  2,  9,  0,  3,
     8,  2,  5,  9,  6,  7,  9,
     5, -1,  9,  0,  7, -6,  9,
    -5,  9,  5,  3,  9,  9,  0},
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
    6.20e+01, 3.10e+01, -1.20e+01, 1.10e+01, 3.10e+01, 2.00e+01, 6.40e+01},
truth_diamatrix_Tl_multiplication[] = {
     22,  -10,   55,   31,  -16,  -25,   62,
     -5,  -60,   53,   22,   27,  -32,   31,
     32,  -56,   20,   81,   54,   69,  -12,
      0,  -72,  -76,  -40,   58,  -20,   11,
     -3,  -74, -110,   -2,   65,   38,   31,
    -11,  -61,  -63,  -40,   40,  -11,   20,
    -59,  -64,  -73,    8,   21,  -25,   64};

const std::unordered_map<std::string, const double*>
    diamatrix_truth_map ({
        {"LN", truth_diamatrix_LN_multiplication},
        {"LT", truth_diamatrix_LT_multiplication},
        {"RN", truth_diamatrix_RN_multiplication},
        {"RT", truth_diamatrix_RT_multiplication},
        {"Nl", truth_diamatrix_LN_multiplication},
        {"Tl", truth_diamatrix_Tl_multiplication}
    });

const std::unordered_map<char, CBLAS_SIDE>
    char2side ({
        {'L', CblasLeft}, {'R', CblasRight}
    });

const std::unordered_map<char, CBLAS_TRANSPOSE>
    char2trans ({
        {'N', CblasNoTrans}, {'T', CblasTrans}
    });

int _compare_one_DiaMatrix_multiplications(
        const std::string &combo, const double* result) {
    int r = 0;
    const double *truth = diamatrix_truth_map.at(combo);
    if (not allClose(truth, result, NrowsDiag*NrowsDiag))
    {
        fprintf(stderr, "ERROR DiaMatrix::multiply('%c', '%c').\n", combo[0], combo[1]);
        printMatrices(truth, result, NrowsDiag, NrowsDiag);
        r += 1;
    }
    return r;
}

int test_DiaMatrix_multiplications()
{
    int r = 0;
    std::vector<double> result_R(NrowsDiag*NrowsDiag, 0);
    mxhelp::DiaMatrix diarmat(NrowsDiag, NdiagDiag);
    std::copy(
        &diamatrix_diagonals[0],
        &diamatrix_diagonals[0]+NrowsDiag*NdiagDiag,
        diarmat.matrix()
    );

    const std::vector<std::string> combos {"LN", "LT", "RN", "RT"};
    for (const auto &combo: combos)
    {
        diarmat.multiply(
            char2side.at(combo[0]),
            char2trans.at(combo[1]),
            diamatrix_multiplier_B, result_R.data());
        r += _compare_one_DiaMatrix_multiplications(combo, result_R.data());
    }

    return r;
}

int test_DiaMatrix_multipyLeft()
{
    int r = 0;
    std::vector<double> result_R(NrowsDiag*NrowsDiag, 0);
    mxhelp::DiaMatrix diarmat(NrowsDiag, NdiagDiag);
    std::copy(
        &diamatrix_diagonals[0],
        &diamatrix_diagonals[0]+NrowsDiag*NdiagDiag,
        diarmat.matrix()
    );

    const std::vector<std::string> combos {"Nl"};
    for (const auto &combo: combos)
    {
        diarmat.multiplyLeft(diamatrix_multiplier_B, result_R.data());
        r += _compare_one_DiaMatrix_multiplications(combo, result_R.data());
    }
    return r;
}

int test_DiaMatrix_multiplyRightT()
{
    int r = 0;
    std::vector<double> result_R(NrowsDiag*NrowsDiag, 0);
    mxhelp::DiaMatrix diarmat(NrowsDiag, NdiagDiag);
    std::copy(
        &diamatrix_diagonals[0],
        &diamatrix_diagonals[0]+NrowsDiag*NdiagDiag,
        diarmat.matrix()
    );

    const std::vector<std::string> combos {"RT"};
    for (const auto &combo: combos)
    {
        diarmat.multiplyRightT(diamatrix_multiplier_B, result_R.data());
        r += _compare_one_DiaMatrix_multiplications(combo, result_R.data());
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
            printMatrices(truth_row, testrow.data(), testrow.size(), 1);
            r += 1;
        }
    }

    return r;
}

int test_Resolution_osamp()
{
    const int
    ndim=496, ndiags=11, osamp_factor=4;
    mxhelp::DiaMatrix diamat(ndim, ndiags);
    mxhelp::Resolution rmat(ndim, ndiags);
    auto vel = std::make_unique<double[]>(ndim);

    const double
    a_kms=20., R_kms=20.,
    truth_row[] = { 8.44661e-07,
        2.54207e-06, 7.50658e-06, 2.13402e-05, 5.73072e-05, 1.43213e-04,
        3.33479e-04, 7.27409e-04, 1.49424e-03, 2.90292e-03, 5.33307e-03,
        9.25408e-03, 1.51492e-02, 2.33753e-02, 3.40046e-02, 4.66622e-02,
        6.04319e-02, 7.38972e-02, 8.53227e-02, 9.30139e-02, 9.57301e-02,
        9.30139e-02, 8.53227e-02, 7.38972e-02, 6.04319e-02, 4.66622e-02,
        3.40046e-02, 2.33753e-02, 1.51492e-02, 9.25408e-03, 5.33307e-03,
        2.90292e-03, 1.49424e-03, 7.27409e-04, 3.33479e-04, 1.43213e-04,
        5.73072e-05, 2.13402e-05, 7.50658e-06, 2.54207e-06, 8.44661e-07 };

    for (int i = 0; i < ndim; ++i)
        vel[i] = (i - ndim/2.)*a_kms;

    diamat.constructGaussian(vel.get(), R_kms, a_kms);
    std::copy(diamat.matrix(), diamat.matrix()+ndim*ndiags, rmat.matrix());
    rmat.oversample(osamp_factor, a_kms);

    int nelemperrow = rmat.getNElemPerRow(), r=0;

    for (int row = 0; row < ndim; ++row)
    {
        double *this_row = rmat.matrix() + row*nelemperrow;
        if (not allClose(truth_row, this_row, nelemperrow))
        {
            fprintf(stderr, "ERROR Resolution::oversample.\n");
            printMatrices(truth_row, this_row, 1, nelemperrow);
            r += 1;
        }
    }

    return r;
}

int main()
{
    int r = 0;
    r += test_cblas_ddot();
    r += test_cblas_dsymv_1();
    r += test_cblas_dsymv_2();
    r += test_cblas_dsymm();
    r += test_getDiagonal();
    r += test_trace_ddiagmv_1();
    r += test_trace_ddiagmv_2();
    r += test_my_cblas_dsymvdot();
    r += test_LAPACKE_InvertMatrixLU_1();
    r += test_LAPACKE_InvertMatrixLU_2();
    r += test_OversampledMatrix_multiplications();
    r += test_DiaMatrix_multiplications();
    r += test_DiaMatrix_multipyLeft();
    r += test_DiaMatrix_multiplyRightT();
    r += test_DiaMatrix_getRow();
    r += test_LAPACKE_SVD();
    r += test_Resolution_osamp();

    if (r == 0)
        printf("Matrix operations work!\n");

    return r;
}



  