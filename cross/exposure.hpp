#ifndef EXPOSURE_H
#define EXPOSURE_H

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "core/chunk_estimate.hpp"
#include "core/global_numbers.hpp"
#include "mathtools/smoother.hpp"

/*
Class for a single exposure.
Each exposure builts its own covariance matrix. Derivative matrices are built
for cross exposures in OneQsoExposures. Two exposures are cross correlated if
the wavelength overlap is large.
*/
class Exposure: public Chunk
{
    std::unique_ptr<double[]> weighted_data, local_cov_mat;
public:
    Exposure(const qio::QSOFile &qmaster, int i1, int i2) : Chunk() {
        _copyQSOFile(qmaster, i1, i2);
        _setNQandFisherIndex();
        glmemory::setMaxSizes(size(), size(), 2 * N_Q_MATRICES, false);

        // to keep update memories in check as this class does not store
        // its own fisher matrix and dbt vectors.
        N_Q_MATRICES = 0;
    }
    Exposure(Exposure &&rhs) = delete;
    Exposure(const Exposure &rhs) = delete;
    ~Exposure() { if (local_cov_mat || weighted_data) deallocMatrices(); }

    int getExpId() const { return qFile->expid; };
    int getNight() const { return qFile->night; };
    int getFiber() const { return qFile->fiber; };
    int getPetal() const { return qFile->petal; };
    double* getWeightedData() const { return weighted_data.get(); };
    double* getInverseCov() const { return inverse_covariance_matrix; };

    void initMatrices() {
        _initMatrices();
        local_cov_mat = std::make_unique<double[]>(DATA_SIZE_2);
        weighted_data = std::make_unique<double[]>(size());
        covariance_matrix = local_cov_mat.get();

        process::updateMemory(-process::getMemoryMB(DATA_SIZE_2 + size()));
    }

    void deallocMatrices() {
        local_cov_mat.reset();
        weighted_data.reset();
        covariance_matrix = nullptr;
        process::updateMemory(process::getMemoryMB(DATA_SIZE_2 + size()));
    };

    void setInvertCovarianceMatrix() {
        double t = mytime::timer.getTime();
        _setVZMatrices();

        if (specifics::TURN_OFF_SFID)
            std::fill_n(covariance_matrix, DATA_SIZE_2, 0);
        else
            _setFiducialSignalMatrix(covariance_matrix);

        // add noise matrix diagonally
        // but smooth before adding
        double *ivec = qFile->ivar();
        if (process::smoother->isSmoothingOn()) {
            process::smoother->smoothIvar(
                qFile->ivar(), glmemory::temp_vector.get(), size());
            ivec = glmemory::temp_vector.get();
        }

        #pragma omp parallel for
        for (int i = 0; i < size(); ++i) {
            cblas_dscal(size(), ivec[i], covariance_matrix + i * size(), 1);
            covariance_matrix[i * (size() + 1)] += 1.0;
        }

        mxhelp::LAPACKE_InvertMatrixLU(covariance_matrix, size());

        inverse_covariance_matrix = covariance_matrix;

        #pragma omp parallel for
        for (int i = 0; i < size(); ++i)
            for (int j = 0; j < size(); ++j)
                inverse_covariance_matrix[j + i * size()] *= ivec[j];

        if (specifics::CONT_NVECS > 0)
            _addMarginalizations();

        t = mytime::timer.getTime() - t;

        mytime::time_spent_on_c_inv += t;
    };

    void weightDataVector() {
        cblas_dsymv(
            CblasRowMajor, CblasUpper, size(), 1.,
            inverse_covariance_matrix, size(), qFile->delta(), 1,
            0, weighted_data.get(), 1);
    };
};

#endif

