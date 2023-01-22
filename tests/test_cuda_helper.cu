#include "mathtools/cuda_helper.cu"
#include "mathtools/matrix_helper.hpp"
#include "io/logger.hpp"
#include "tests/test_utils.hpp"
#include <cassert>


CuHelper cuhelper;

int
NA = 4;

double
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
    57., 44., 127., 18.};


int test_cublas_dsymv_1()
{
    MyCuPtr<double>
        dev_res(NA), dev_sym_matrix_A(NA*NA, sym_matrix_A),
        dev_vector_cblas_dsymv_b_1(NA, vector_cblas_dsymv_b_1);
    double cpu_res[NA];

    cuhelper.dsmyv(
        CUBLAS_FILL_MODE_UPPER, NA, 0.5, dev_sym_matrix_A.get(), NA,
        dev_vector_cblas_dsymv_b_1.get(), 1, 0, dev_res.get(), 1);
    dev_res.syncDownload(cpu_res, NA);

    if (allClose(truth_cblas_dsymv_1, cpu_res, NA))
        return 0;

    fprintf(stderr, "ERROR test_cublas_dsymv_1.\n");
    printMatrices(truth_cblas_dsymv_1, cpu_res, NA, 1);
    return 1;
}


int test_cublas_dsymm()
{
    double result[NA*NA];

    MyCuPtr<double>
        dev_res(NA*NA), dev_sym_matrix_A(NA*NA, sym_matrix_A),
        dev_matrix_cblas_dsymm_B(NA*NA, matrix_cblas_dsymm_B);

    cuhelper.dsymm(
        CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
        NA, NA, 1., dev_sym_matrix_A.get(), NA,
        dev_matrix_cblas_dsymm_B.get(), NA,
        0, dev_res.get(), NA);

    dev_res.syncDownload(result, NA*NA);

    if (allClose(truth_out_cblas_dsymm, result, NA*NA))
        return 0;
    fprintf(stderr, "ERROR test_cublas_dsymm.\n");
    printMatrices(truth_out_cblas_dsymm, result, NA, NA);
    return 1;
}


double
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

int
NcolsSVD = 6,
NrowsSVD = 5;

int test_cusolver_SVD()
{
    MyCuPtr<double>
        svals(NcolsSVD), dev_svd_matrix(NcolsSVD*NrowsSVD, matrix_for_SVD_A);
    double svd_matrix[NcolsSVD*NrowsSVD];

    cuhelper.svd(dev_svd_matrix.get(), svals.get(), NcolsSVD, NrowsSVD);
    dev_svd_matrix.syncDownload(svd_matrix, NcolsSVD*NrowsSVD);

    if (allClose(truth_SVD_A, svd_matrix, NcolsSVD*NrowsSVD))
        return 0;
    fprintf(stderr, "ERROR test_cusolver_SVD.\n");
    printMatrices(truth_SVD_A, svd_matrix, NrowsSVD, NcolsSVD);
    return 1;
}


int test_cusolver_invert_cholesky()
{
    const int ndim = 496;
    std::vector<double> A, truth_out_inverse;
    int nrows, ncols;

    const std::string
    fname_matrix = std::string(SRCDIR) + "/tests/input/test_symmatrix_cholesky.txt",
    fname_truth  = std::string(SRCDIR) + "/tests/truth/test_inverse_cholesky.txt";

    A = mxhelp::fscanfMatrix(fname_matrix.c_str(), nrows, ncols);
    assert(nrows == ndim);
    assert(ncols == ndim);

    truth_out_inverse = mxhelp::fscanfMatrix(fname_truth.c_str(), nrows, ncols);
    assert(nrows == ndim);
    assert(ncols == ndim);

    MyCuPtr<double> dev_A(ndim*ndim, A.data());

    cuhelper.invert_cholesky(dev_A.get(), ndim);
    dev_A.syncDownload(A.data(), ndim*ndim);
    mxhelp::copyUpperToLower(A.data(), ndim);

    if (allClose(A.data(), truth_out_inverse.data(), ndim*ndim))
        return 0;

    fprintf(stderr, "ERROR test_cusolver_invert_cholesky.\n");
    // printMatrices(truth_out_inverse.data(), A.data(), ndim, ndim);
    return 1;
}


int main(int argc, char *argv[])
{
    int r=0;
    int NA = 500, NB = 5;
    double IN_ARR[] = {1.1, 2.2, 3.3, 4.4, 5.5};

    LOG::LOGGER.STD("Allocating a single MyCuPtr<double>.\n");
    MyCuPtr<double> covariance_matrix;
    assert(covariance_matrix.get() == nullptr);
    covariance_matrix.realloc(NA*NA);
    LOG::LOGGER.STD("Reset a single MyCuPtr<double>.\n");
    covariance_matrix.reset();
    assert(covariance_matrix.get() == nullptr);

    LOG::LOGGER.STD("Allocating array of two MyCuPtr<double>.\n");
    MyCuPtr<double> temp_matrix[2];
    assert(temp_matrix[0].get() == nullptr);
    assert(temp_matrix[1].get() == nullptr);
    temp_matrix[0].realloc(NA*NA);
    temp_matrix[1].realloc(NA*NA);
    LOG::LOGGER.STD("Reset array of two MyCuPtr<double>.\n");
    temp_matrix[0].reset();
    temp_matrix[1].reset();

    LOG::LOGGER.STD("Asnyc copy a single MyCuPtr<double>.\n");
    MyCuPtr<double> vec(NB, IN_ARR);
    std::vector<double> cpu_vec(5);
    vec.syncDownload(cpu_vec.data(), 5);
    if (!allClose(cpu_vec.data(), IN_ARR, 5)) {
        fprintf(stderr, "ERROR: sync download\n");
        r+=1;
    }

    LOG::LOGGER.STD("Testing test_cublas_dsymv_1.\n");
    r+=test_cublas_dsymv_1();
    r+=test_cublas_dsymm();
    r+=test_cusolver_SVD();
    r+=test_cusolver_invert_cholesky();

    if (r != 0)
        fprintf(stderr, "ERRORs occured!!!!!\n");
    return r;
}
