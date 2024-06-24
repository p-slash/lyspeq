#include "mathtools/matrix_helper.hpp"
#include "tests/test_utils.hpp"

#include "mathtools/discrete_interpolation.hpp"

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


int test_cblas_nrm2() {
    const int N = 8;
    double A[N], r;
    for (int i = 0; i < N; ++i)
        A[i] = 1;
    r = cblas_dnrm2(N, A, 1);
    if (isClose(sqrt(N), r))
        return 0;

    fprintf(stderr, "ERROR cblas_dnrm2.\n");
    printValues(sqrt(N), r);
    return 1;
}

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
        
        if (!allClose(truth_diag, v, NA-d))
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
    0.42 , 0.23 , 0.1  , 0.44 , 0.49 , 0.565, 0.675, 0.19 , 0.675,
    0.515, 0.525, 0.415, 0.495, 0.525, 0.805, 0.64 , 0.015, 0.455,
    0.23 , 0.76 , 0.705, 0.155, 0.51 , 0.38 , 0.655, 0.255, 0.615,
    0.395, 0.735, 0.595, 0.43 , 0.695, 0.205, 0.71 , 0.77 , 0.435,
    0.1  , 0.705, 0.24 , 0.52 , 0.28 , 0.5  , 0.18 , 0.355, 0.28 ,
    0.365, 0.64 , 0.45 , 0.37 , 0.225, 0.07 , 0.79 , 0.72 , 0.66 ,
    0.44 , 0.155, 0.52 , 0.14 , 0.595, 0.385, 0.195, 0.56 , 0.465,
    0.545, 0.315, 0.67 , 0.47 , 0.225, 0.205, 0.7  , 0.67 , 0.38 ,
    0.49 , 0.51 , 0.28 , 0.595, 0.75 , 0.26 , 0.38 , 0.6  , 0.315,
    0.635, 0.74 , 0.38 , 0.17 , 0.47 , 0.655, 0.1  , 0.225, 0.57 ,
    0.565, 0.38 , 0.5  , 0.385, 0.26 , 0.6  , 0.355, 0.615, 0.1  ,
    0.515, 0.51 , 0.17 , 0.365, 0.04 , 0.25 , 0.72 , 0.275, 0.59 ,
    0.675, 0.655, 0.18 , 0.195, 0.38 , 0.355, 0.58 , 0.65 , 0.34 ,
    0.44 , 0.45 , 0.79 , 0.44 , 0.67 , 0.715, 0.73 , 0.555, 0.745,
    0.19 , 0.255, 0.355, 0.56 , 0.6  , 0.615, 0.65 , 0.26 , 0.43 ,
    0.31 , 0.605, 0.5  , 0.65 , 0.28 , 0.515, 0.695, 0.535, 0.69 ,
    0.675, 0.615, 0.28 , 0.465, 0.315, 0.1  , 0.34 , 0.43 , 0.96 ,
    0.29 , 0.295, 0.525, 0.725, 0.5  , 0.695, 0.345, 0.16 , 0.33 ,
    0.515, 0.395, 0.365, 0.545, 0.635, 0.515, 0.44 , 0.31 , 0.29 ,
    0.88 , 0.575, 0.5  , 0.81 , 0.575, 0.635, 0.225, 0.46 , 0.24 ,
    0.525, 0.735, 0.64 , 0.315, 0.74 , 0.51 , 0.45 , 0.605, 0.295,
    0.575, 0.43 , 0.295, 0.975, 0.56 , 0.265, 0.08 , 0.265, 0.54 ,
    0.415, 0.595, 0.45 , 0.67 , 0.38 , 0.17 , 0.79 , 0.5  , 0.525,
    0.5  , 0.295, 0.47 , 0.565, 0.47 , 0.6  , 0.775, 0.21 , 0.755,
    0.495, 0.43 , 0.37 , 0.47 , 0.17 , 0.365, 0.44 , 0.65 , 0.725,
    0.81 , 0.975, 0.565, 0.97 , 0.72 , 0.15 , 0.86 , 0.565, 0.4  ,
    0.525, 0.695, 0.225, 0.225, 0.47 , 0.04 , 0.67 , 0.28 , 0.5  ,
    0.575, 0.56 , 0.47 , 0.72 , 0.66 , 0.52 , 0.575, 0.48 , 0.705,
    0.805, 0.205, 0.07 , 0.205, 0.655, 0.25 , 0.715, 0.515, 0.695,
    0.635, 0.265, 0.6  , 0.15 , 0.52 , 0.12 , 0.225, 0.595, 0.645,
    0.64 , 0.71 , 0.79 , 0.7  , 0.1  , 0.72 , 0.73 , 0.695, 0.345,
    0.225, 0.08 , 0.775, 0.86 , 0.575, 0.225, 0.04 , 0.6  , 0.56 ,
    0.015, 0.77 , 0.72 , 0.67 , 0.225, 0.275, 0.555, 0.535, 0.16 ,
    0.46 , 0.265, 0.21 , 0.565, 0.48 , 0.595, 0.6  , 0.83 , 0.44 ,
    0.455, 0.435, 0.66 , 0.38 , 0.57 , 0.59 , 0.745, 0.69 , 0.33 ,
    0.24 , 0.54 , 0.755, 0.4  , 0.705, 0.645, 0.56 , 0.44 , 0.83},
truth_oversample_left_multiplication[] = {
    4.52800e-01, 6.68900e-01, 5.24650e-01, 5.05500e-01, 7.17250e-01, 5.83100e-01,
    5.66050e-01, 5.43900e-01, 6.47500e-01, 6.73250e-01, 8.19150e-01, 7.05100e-01,
    5.41950e-01, 5.85300e-01, 5.05600e-01, 8.38800e-01, 7.00800e-01, 6.97200e-01,
    6.37100e-01, 6.53250e-01, 4.87600e-01, 5.15450e-01, 6.45450e-01, 5.80000e-01,
    4.67300e-01, 7.78000e-01, 4.16500e-01, 7.05500e-01, 7.41150e-01, 6.71400e-01,
    4.97350e-01, 4.46300e-01, 5.31250e-01, 8.22800e-01, 6.62550e-01, 8.09250e-01,
    7.19450e-01, 6.68450e-01, 4.43750e-01, 5.99500e-01, 6.33450e-01, 5.52400e-01,
    6.51250e-01, 7.12750e-01, 5.77850e-01, 6.06900e-01, 7.21600e-01, 6.65450e-01,
    6.54500e-01, 5.43000e-01, 7.81000e-01, 7.47900e-01, 5.03200e-01, 8.27150e-01,
    7.13700e-01, 7.26650e-01, 5.00600e-01, 5.92100e-01, 7.35800e-01, 5.74950e-01,
    6.79000e-01, 6.12350e-01, 6.65150e-01, 6.88750e-01, 6.53650e-01, 7.24000e-01,
    1.00190e+00, 7.10450e-01, 7.93950e-01, 5.76250e-01, 5.43150e-01, 6.93250e-01,
    7.24950e-01, 7.76200e-01, 5.97250e-01, 6.83750e-01, 6.45000e-01, 4.73000e-01,
    6.91200e-01, 6.92450e-01, 7.49050e-01, 8.53450e-01, 7.03100e-01, 6.45250e-01,
    1.12475e+00, 7.81650e-01, 6.51550e-01, 6.16050e-01, 4.57700e-01, 6.38450e-01,
    7.58550e-01, 7.42400e-01, 4.87950e-01, 5.35250e-01, 6.47950e-01, 3.64050e-01,
    8.51050e-01, 7.07700e-01, 7.69500e-01, 8.65350e-01, 7.23950e-01, 6.71500e-01,
    9.54600e-01, 8.20900e-01, 4.67550e-01, 7.34350e-01, 5.90250e-01, 8.43650e-01,
    7.14900e-01, 7.71000e-01, 5.88550e-01, 6.16800e-01, 4.67100e-01, 4.57900e-01,
    8.76400e-01, 7.38650e-01, 6.81750e-01, 7.46350e-01, 5.77750e-01, 7.40800e-01,
    8.88450e-01, 8.19350e-01, 4.39500e-01, 6.13100e-01, 8.46400e-01, 7.76800e-01},
truth_oversample_right_multiplication[] = {
    7.95730e-01, 8.10717e-01, 8.43493e-01, 8.99998e-01, 9.51020e-01, 8.73408e-01, 8.80412e-01,
    8.10716e-01, 7.56495e-01, 8.04170e-01, 8.61442e-01, 8.57520e-01, 7.96411e-01, 8.22701e-01,
    8.43493e-01, 8.04170e-01, 8.71606e-01, 9.05044e-01, 9.00765e-01, 9.28995e-01, 9.05896e-01,
    8.99998e-01, 8.61443e-01, 9.05045e-01, 9.16678e-01, 1.02909e+00, 1.08848e+00, 1.00553e+00,
    9.51021e-01, 8.57520e-01, 9.00765e-01, 1.02909e+00, 1.12103e+00, 1.10114e+00, 1.00293e+00,
    8.73408e-01, 7.96411e-01, 9.28995e-01, 1.08848e+00, 1.10114e+00, 1.02660e+00, 9.82175e-01,
    8.80412e-01, 8.22701e-01, 9.05896e-01, 1.00553e+00, 1.00293e+00, 9.82175e-01, 9.85368e-01 
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

    ovrmat.multiplyLeft(oversample_multiplier_A, mtrxB1.data());
    if (!allClose(truth_oversample_left_multiplication, mtrxB1.data(), mtrxB1.size()))
    {
        fprintf(stderr, "ERROR OversampledMatrix::multiplyLeft.\n");
        printMatrices(truth_oversample_left_multiplication,
            mtrxB1.data(), NrowsOversamp, NcolsOversamp);
        r += 1;
    }

    ovrmat.multiplyRight(mtrxB1.data(), mtrxB2.data());
    if (!allClose(truth_oversample_right_multiplication, mtrxB2.data(), mtrxB2.size()))
    {
        fprintf(stderr, "ERROR OversampledMatrix::multiplyRight.\n");
        printMatrices(truth_oversample_right_multiplication,
            mtrxB2.data(), NrowsOversamp, NrowsOversamp);
        r += 1;
    }

    ovrmat.sandwich(oversample_multiplier_A, mtrxB2.data());
    if (!allClose(truth_oversample_right_multiplication, mtrxB2.data(), mtrxB2.size()))
    {
        fprintf(stderr, "ERROR OversampledMatrix::sandwich.\n");
        printMatrices(truth_oversample_right_multiplication,
            mtrxB2.data(), NrowsOversamp, NrowsOversamp);
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
diamatrix_multiplier_C[] = {
    3, -4, 7, 7, -5, -2, -4, 1, 2,
    -2, -7, 4, -2, 0, -9, -7, 1, 2,
    6, 6, 5, -6, -3, -9, -4, 1, 2,
    3, 2, 4, -2, 8, -8, -4, 1, 2,
    1, 2, -5, 7, -1, 6, 1, 1, 2,
    3, -7, 0, 9, 1, -6, -2, 1, 2,
    -3, 5, 9, -6, 4, 1, 8, 1, 2},
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
    -59,  -64,  -73,    8,   21,  -25,   64},
truth_diamatrix_Nlrect_multiplication[] = {
    25,   -2,   48,  -31,  -20,  -83,  -52,   10,   20,
    2,  -54, 59,  -12,    3, -101,  -76,   13,   26,
    53,  -30,   77,   84, 1,  -77,  -89,   28,   56,
    36,  -40,   11,   63,   26,  -72, -35,   14,   28,
    34,   26,   34,   41,   66,  -43,   -1,   23, 46,
    10,   -7,   30,   21,   31,  -31,    9,   10,   20,
    7, 6,    1,   61,   13,   22,   31,   15,   30},
truth_diamatrix_RTrect_multiplication[] = {
    22,  -5,  32,   0,  -3, -11, -59,  -1,   2,  -6, -27, -31, -18,
    20, -32, -69, -52,  54, 101,  49,  86, -63, -53, -96,  -8, -10,
    10,  27,   8,  50,  32, -18, -14,  -5, -68,  16,   4,  38,  -5,
    36,  11,  69,  15,  18, -13,  -2,  10,  26, -13, -33, -46,   4,
   -22,  27,   7,  22,   5, -24, 124,  24,  79,  20,  68
}
;

const std::unordered_map<std::string, const double*>
    diamatrix_truth_map ({
        {"LN", truth_diamatrix_LN_multiplication},
        {"LT", truth_diamatrix_LT_multiplication},
        {"RN", truth_diamatrix_RN_multiplication},
        {"RT", truth_diamatrix_RT_multiplication},
        {"Nl", truth_diamatrix_LN_multiplication},
        {"Tl", truth_diamatrix_Tl_multiplication},
        {"Nlrect", truth_diamatrix_Nlrect_multiplication},
        {"RTrect", truth_diamatrix_RTrect_multiplication}
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
        const std::string &combo, const double* result,
        int nrows=NrowsDiag, int ncols=NrowsDiag
) {
    int r = 0;
    const double *truth = diamatrix_truth_map.at(combo);
    if (!allClose(truth, result, nrows * ncols))
    {
        fprintf(stderr, "ERROR DiaMatrix::multiply('%c', '%c').\n", combo[0], combo[1]);
        printMatrices(truth, result, nrows, ncols);
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

int test_DiaMatrix_multipyLeft_rectangle()
{
    std::vector<double> result_R(NrowsDiag * (NrowsDiag + 2), 0);
    mxhelp::DiaMatrix diarmat(NrowsDiag, NdiagDiag);
    std::copy(
        &diamatrix_diagonals[0],
        &diamatrix_diagonals[0]+NrowsDiag*NdiagDiag,
        diarmat.matrix()
    );

    diarmat.multiplyLeft(diamatrix_multiplier_C, result_R.data(), NrowsDiag + 2);
    return _compare_one_DiaMatrix_multiplications(
        "Nlrect", result_R.data(), NrowsDiag, NrowsDiag + 2);
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

int test_DiaMatrix_multipyRightT_rectangle()
{
    std::vector<double> result_R(NrowsDiag * (NrowsDiag + 2), 0);
    mxhelp::DiaMatrix diarmat(NrowsDiag, NdiagDiag);
    std::copy(
        &diamatrix_diagonals[0],
        &diamatrix_diagonals[0]+NrowsDiag*NdiagDiag,
        diarmat.matrix()
    );

    diarmat.multiplyRightT(diamatrix_multiplier_C, result_R.data(), NrowsDiag + 2);
    return _compare_one_DiaMatrix_multiplications(
        "RTrect", result_R.data(), NrowsDiag + 2, NrowsDiag);
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
        
        if (!allClose(truth_row, testrow.data(), testrow.size()))
        {
            fprintf(stderr, "ERROR DiaMatrix::getRow(%d).\n", row);
            printMatrices(truth_row, testrow.data(), testrow.size(), 1);
            r += 1;
        }
    }

    return r;
}

int test_DiaMatrix_orderTranspose() {
    int r = 0;
    std::vector<double> testrow(NdiagDiag);

    mxhelp::DiaMatrix diarmat(NrowsDiag, NdiagDiag);
    const double diamatrix_diagonals_oT[] = {
        0, 0, 1, 2, 9,
        0, 4, 8, 4, 1,
        5, 1, 1, 1, 1,
        2, 7, 3, 7, 2,
        7, 2, 7, 1, 7,
        7, 4, 4, 4, 0,
        4, 3, 4, 0, 0};

    std::copy_n(
        diamatrix_diagonals_oT,
        NrowsDiag * NdiagDiag,
        diarmat.matrix()
    );

    diarmat.orderTranspose();
    if (!allClose(diamatrix_diagonals, diarmat.matrix(), diarmat.getSize()))
    {
        fprintf(stderr, "ERROR DiaMatrix::orderTranspose().\n");
        printMatrices(diamatrix_diagonals, diarmat.matrix(), NdiagDiag, NrowsDiag);
        r += 1;
    }
    return r;
    return 0;
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
        if (!allClose(truth_row, this_row, nelemperrow))
        {
            fprintf(stderr, "ERROR Resolution::oversample.\n");
            printMatrices(truth_row, this_row, 1, nelemperrow);
            r += 1;
        }
    }

    return r;
}


int test_cubic_interpolate() {
    double dv = 2.0;
    int narr = 25;
    auto yarr = std::make_unique<double[]>(narr);
    const double a = 21312;
    for (int i = 0; i < narr; ++i) {
        double x = dv * i, x3 = x * x * x;
        yarr[i] = a * x3 + 10;
    }

    auto interp1d_cubic = std::make_unique<DiscreteCubicInterpolation1D>(
            0, dv, narr, yarr.get());

    dv = 1.0;
    narr = 45;
    yarr = std::make_unique<double[]>(narr);
    auto truth = std::make_unique<double[]>(narr);
    for (int i = 0; i < narr; ++i) {
        double x = dv * i, x3 = x * x * x;
        truth[i] = a * x3 + 10;
        yarr[i] = interp1d_cubic->evaluate(x);
    }

    if (!allClose(truth.get(), yarr.get(), narr, 1e-2)) {
        fprintf(stderr, "ERROR test_cubic_interpolate.\n");
        printMatrices(truth.get(), yarr.get(), 1, narr);
        return 1;
    }

    return 0;
}

int main()
{
    int r = 0;
    r += test_cblas_nrm2();
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
    r += test_DiaMatrix_multipyLeft_rectangle();
    r += test_DiaMatrix_multiplyRightT();
    r += test_DiaMatrix_multipyRightT_rectangle();
    r += test_DiaMatrix_getRow();
    r += test_DiaMatrix_orderTranspose();
    r += test_LAPACKE_SVD();
    r += test_Resolution_osamp();
    r += test_cubic_interpolate();

    if (r == 0)
        printf("Matrix operations work!\n");

    return r;
}



  