#include "mathtools/cuda_helper.cu"
#include "mathtools/matrix_helper.hpp"
// #include "io/logger.hpp"
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
    3, 4, 4, 5,
    1, 8, 3, 5,
    9, 8, 2, 9,
    0, 8, 0, 2},
truth_out_cblas_dsymm [] = {
    104, 67, 34, 57,
    113, 89, 28, 44,
    170, 139, 88, 127,
    64, 74, 18, 18};


void test_cublas_dsymv_1() {
    MyCuPtr<double>
        dev_res(NA), dev_sym_matrix_A(NA*NA, sym_matrix_A),
        dev_vector_cblas_dsymv_b_1(NA, vector_cblas_dsymv_b_1);
    double cpu_res[NA];

    cuhelper.dsmyv(
        CUBLAS_FILL_MODE_UPPER, NA, 0.5, dev_sym_matrix_A.get(), NA,
        dev_vector_cblas_dsymv_b_1.get(), 1, 0, dev_res.get(), 1);
    dev_res.syncDownload(cpu_res, NA);

    assert_allclose(truth_cblas_dsymv_1, cpu_res, NA, __FILE__, __LINE__);
}


void test_cublas_dsymm() {
    LOG::LOGGER.STD("test_cublas_dsymm...");
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

    assert_allclose_2d(
        truth_out_cblas_dsymm, result, NA, NA, __FILE__, __LINE__);
}


double
matrix_for_SVD_A[] = {
    8.79, 6.11,  -9.15,  9.57, -3.49,  9.84,
    9.93, 6.91,  -7.93,  1.64,  4.02,  0.15,
    9.83, 5.04,  4.86,  8.83,   9.8,  -8.99,
    5.45, -0.27, 4.85,  0.74,  10.00, -6.02,
    3.16, 7.98,  3.01,  5.8,    4.27, -5.31},
truth_svals[] = {
    27.46873242, 22.64318501,  8.55838823,  5.9857232 ,  2.01489966},
truth_SVD_A[] = {
    -0.59114238, -0.39756679, -0.03347897, -0.4297069 , -0.46974792, 0.29335876, 
    0.26316781,  0.24379903, -0.60027258,  0.23616681, -0.3508914 , 0.57626212, 
    0.35543017, -0.22239   , -0.45083927, -0.68586286,  0.3874446 , -0.02085292, 
    0.31426436, -0.75346615,  0.23344966,  0.33186002,  0.15873556, 0.37907767, 
    0.22993832, -0.36358969, -0.30547573,  0.16492763, -0.51825744, -0.6525516};

int
NcolsSVD = 5,
NrowsSVD = 6;

void test_cusolver_SVD() {
    LOG::LOGGER.STD("Testing test_cusolver_SVD.\n");

    MyCuPtr<double>
        svals(NcolsSVD), dev_svd_matrix(NcolsSVD*NrowsSVD, matrix_for_SVD_A, cuhelper.stream);
    double svd_matrix[NcolsSVD*NrowsSVD], cpu_svals[NcolsSVD];

    cuhelper.svd(dev_svd_matrix.get(), svals.get(), NrowsSVD, NcolsSVD);
    // cuhelper.streamSync();
    dev_svd_matrix.syncDownload(svd_matrix, NcolsSVD*NrowsSVD);
    svals.syncDownload(cpu_svals, NcolsSVD);

    assert_allclose(truth_svals, cpu_svals, NcolsSVD, __FILE__, __LINE__);

    // Truth is degenerate with a minus sign
    // assert_allclose_2d(
    //     truth_SVD_A, svd_matrix, NcolsSVD, NrowsSVD,
    //     __FILE__, __LINE__);
}


void test_cusolver_potrf() {
    LOG::LOGGER.STD("Testing test_cusolver_potrf.\n");
    const int ndim = 496;
    std::vector<double> A, truth_L;
    int nrows, ncols;

    const std::string
    fname_matrix =
        std::string(SRCDIR) + "/tests/input/test_triu_cholesky.txt",
    fname_truth  = std::string(SRCDIR) + "/tests/truth/test_L_cholesky.txt";

    A = mxhelp::fscanfMatrix(fname_matrix.c_str(), nrows, ncols);
    raiser(nrows == ndim, __FILE__, __LINE__);
    raiser(ncols == ndim, __FILE__, __LINE__);

    truth_L = mxhelp::fscanfMatrix(fname_truth.c_str(), nrows, ncols);
    raiser(nrows == ndim, __FILE__, __LINE__);
    raiser(ncols == ndim, __FILE__, __LINE__);

    MyCuPtr<double> dev_A(ndim * ndim, A.data());
    cuhelper.potrf(dev_A.get(), ndim);
    dev_A.syncDownload(A.data(), ndim * ndim);

    assert_allclose_2d(
        truth_L.data(), A.data(), ndim, ndim,
        __FILE__, __LINE__);
}


void test_cusolver_potri() {
    LOG::LOGGER.STD("Testing test_cusolver_potri.\n");
    const int ndim = 496;
    std::vector<double> A, truth_inverse;
    int nrows, ncols;

    const std::string
    fname_matrix = std::string(SRCDIR) + "/tests/truth/test_L_cholesky.txt",
    fname_truth  =
        std::string(SRCDIR) + "/tests/truth/test_inverse_cholesky.txt";

    A = mxhelp::fscanfMatrix(fname_matrix.c_str(), nrows, ncols);
    raiser(nrows == ndim, __FILE__, __LINE__);
    raiser(ncols == ndim, __FILE__, __LINE__);

    truth_inverse = mxhelp::fscanfMatrix(fname_truth.c_str(), nrows, ncols);
    raiser(nrows == ndim, __FILE__, __LINE__);
    raiser(ncols == ndim, __FILE__, __LINE__);

    MyCuPtr<double> dev_A(ndim * ndim, A.data());
    cuhelper.potri(dev_A.get(), ndim);
    dev_A.syncDownload(A.data(), ndim * ndim);
    mxhelp::copyUpperToLower(A.data(), ndim);

    assert_allclose_2d(
        truth_inverse.data(), A.data(), ndim, ndim,
        __FILE__, __LINE__);
}


void test_cusolver_invert_cholesky() {
    const int ndim = 496;
    std::vector<double> A, truth_out_inverse;
    int nrows, ncols;

    const std::string
    fname_matrix =
        std::string(SRCDIR) + "/tests/input/test_symmatrix_cholesky.txt",
    fname_truth  = std::string(SRCDIR) + "/tests/truth/test_inverse_cholesky.txt";

    A = mxhelp::fscanfMatrix(fname_matrix.c_str(), nrows, ncols);
    raiser(nrows == ndim, __FILE__, __LINE__);
    raiser(ncols == ndim, __FILE__, __LINE__);

    truth_out_inverse = mxhelp::fscanfMatrix(fname_truth.c_str(), nrows, ncols);
    raiser(nrows == ndim, __FILE__, __LINE__);
    raiser(ncols == ndim, __FILE__, __LINE__);

    MyCuPtr<double> dev_A(ndim*ndim, A.data());

    cuhelper.invert_cholesky(dev_A.get(), ndim);
    dev_A.syncDownload(A.data(), ndim * ndim);
    mxhelp::copyUpperToLower(A.data(), ndim);

    assert_allclose_2d(
        truth_out_inverse.data(), A.data(), ndim, ndim,
        __FILE__, __LINE__);
}


void test_MyCuPtr_init() {
    int NA = 500;
    // LOG::LOGGER.STD("Allocating a single MyCuPtr<double>.\n");
    MyCuPtr<double> covariance_matrix;
    raiser(covariance_matrix.get() == nullptr, __FILE__, __LINE__);
    covariance_matrix.realloc(NA*NA);
    // LOG::LOGGER.STD("Reset a single MyCuPtr<double>.\n");
    covariance_matrix.reset();
    raiser(covariance_matrix.get() == nullptr, __FILE__, __LINE__);
}


void test_MyCuPtr_array() {
    int NA = 500;
    // LOG::LOGGER.STD("Allocating array of two MyCuPtr<double>.\n");
    MyCuPtr<double> temp_matrix[2];
    raiser(temp_matrix[0].get() == nullptr, __FILE__, __LINE__);
    raiser(temp_matrix[1].get() == nullptr, __FILE__, __LINE__);
    temp_matrix[0].realloc(NA*NA);
    temp_matrix[1].realloc(NA*NA);
    // LOG::LOGGER.STD("Reset array of two MyCuPtr<double>.\n");
    temp_matrix[0].reset();
    temp_matrix[1].reset();
}


void test_MyCuPtr_async() {
    int NB = 5;
    double IN_ARR[] = {1.1, 2.2, 3.3, 4.4, 5.5};

    // LOG::LOGGER.STD("Asnyc copy a single MyCuPtr<double>.\n");
    MyCuPtr<double> vec(NB, IN_ARR);
    std::vector<double> cpu_vec(5);
    vec.syncDownload(cpu_vec.data(), 5);
    assert_allclose(IN_ARR, cpu_vec.data(), 5, __FILE__, __LINE__);
}


int main(int argc, char *argv[]) {
    int r=0;

    cuhelper.streamCreate();
    setbuf(stdout, NULL);

    r += catcher(&test_MyCuPtr_init, "test_MyCuPtr_init");
    r += catcher(&test_MyCuPtr_array, "test_MyCuPtr_array");
    r += catcher(&test_MyCuPtr_async, "test_MyCuPtr_async");

    r += catcher(&test_cublas_dsymv_1, "test_cublas_dsymv_1");
    r += catcher(&test_cublas_dsymm, "test_cublas_dsymm");
    r += catcher(&test_cusolver_SVD, "test_cusolver_SVD");
    r += catcher(&test_cusolver_potrf, "test_cusolver_potrf");
    r += catcher(&test_cusolver_potri, "test_cusolver_potri");
    r += catcher(
        &test_cusolver_invert_cholesky, "test_cusolver_invert_cholesky");

    if (r != 0)
        fprintf(stderr, "ERRORs occured!!!!!\n");
    return r;
}
