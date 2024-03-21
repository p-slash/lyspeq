#include "cross/exposure.hpp"
#include "core/global_numbers.hpp"
#include "core/omp_manager.hpp"
#include "core/fiducial_cosmology.hpp"
#include "core/sq_table.hpp"

#include "mathtools/smoother.hpp"
#include "io/io_helper_functions.hpp"
#include "io/logger.hpp"

#include <cmath>
#include <algorithm> // std::for_each & transform & lower(upper)_bound
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <stdexcept>


namespace glmemory {
    std::unique_ptr<double[]> temp_vector;
}


void Exposure::initMatrices()
{
    _initMatrices();
    covariance_matrix = new double[DATA_SIZE_2];
    weighted_data = std::make_unique<double[]>(size());
}


void Exposure::setCovarianceMatrix() {
    _setVZMatrices();

    for (int i = 0; i < _matrix_n; ++i)
        _matrix_lambda[i] += 1;

    _setFiducialSignalMatrix(covariance_matrix);

    // add noise matrix diagonally
    // but smooth before adding
    double *nvec = qFile->noise();
    if (process::smoother->isSmoothingOn()) {
        process::smoother->smoothNoise(
            qFile->noise(), glmemory::temp_vector.get(), size());
        nvec = glmemory::temp_vector.get();
    }

    cblas_daxpy(size(), 1., nvec, 1, covariance_matrix, size() + 1);
}


void Exposure::weightDataVector() {
    // C-1 . flux
    cblas_dsymv(
        CblasRowMajor, CblasUpper, size(), 1.,
        inverse_covariance_matrix, size(), qFile->delta(), 1,
        0, weighted_data.get(), 1);
}