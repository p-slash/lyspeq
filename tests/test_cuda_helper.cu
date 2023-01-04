#include "mathtools/cuda_helper.cu"
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
vector_cblas_dsymv_b_1[] = {4, 5, 6, 7};


int test_cublas_dsymv_1()
{
    MyCuDouble
    dev_res(NA), dev_sym_matrix_A(NA*NA, sym_matrix_A),
    dev_vector_cblas_dsymv_b_1(NA, vector_cblas_dsymv_b_1);
    double cpu_res[NA];

    cuhelper.dsmyv(
        CUBLAS_FILL_MODE_UPPER, NA, 0.5, dev_sym_matrix_A.get(), NA,
        dev_vector_cblas_dsymv_b_1.get(), 1, 0, dev_res.get(), 1);
    dev_res.syncDownload(cpu_res);

    if (allClose(truth_cblas_dsymv_1, cpu_res, NA))
        return 0;

    fprintf(stderr, "ERROR test_cublas_dsymv_1.\n");
    printMatrices(truth_cblas_dsymv_1, cpu_res, NA, 1);
    return 1;
}


int main(int argc, char *argv[])
{
    int r=0;
    int NA = 500, NB = 5;
    double IN_ARR[] = {1.1, 2.2, 3.3, 4.4, 5.5};

    LOG::LOGGER.STD("Allocating a single MyCuDouble.\n");
    MyCuDouble covariance_matrix;
    assert(covariance_matrix.get() == nullptr);
    covariance_matrix.realloc(NA*NA);
    LOG::LOGGER.STD("Reset a single MyCuDouble.\n");
    covariance_matrix.reset();
    assert(covariance_matrix.get() == nullptr);

    LOG::LOGGER.STD("Allocating array of two MyCuDouble.\n");
    MyCuDouble temp_matrix[2];
    assert(temp_matrix[0].get() == nullptr);
    assert(temp_matrix[1].get() == nullptr);
    temp_matrix[0].realloc(NA*NA);
    temp_matrix[1].realloc(NA*NA);
    LOG::LOGGER.STD("Reset array of two MyCuDouble.\n");
    temp_matrix[0].reset();
    temp_matrix[1].reset();

    LOG::LOGGER.STD("Asnyc copy a single MyCuDouble.\n");
    MyCuDouble vec(NB, IN_ARR);

    LOG::LOGGER.STD("Testing test_cublas_dsymv_1.\n");
    r+=test_cublas_dsymv_1();
    return r;
}
