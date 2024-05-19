#include "cross/exposure.hpp"
#include "core/global_numbers.hpp"

#include "mathtools/smoother.hpp"
#include "io/logger.hpp"

#include <cmath>
#include <algorithm> // std::for_each & transform & lower(upper)_bound
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <stdexcept>


void Exposure::setCovarianceMatrix() {
    _setVZMatrices();

    if (specifics::TURN_OFF_SFID)
        std::fill_n(covariance_matrix, DATA_SIZE_2, 0);
    else
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
