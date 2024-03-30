#ifndef EXPOSURE_H
#define EXPOSURE_H

#include <string>
#include <vector>
#include <memory>

#include "core/chunk_estimate.hpp"
#include "core/global_numbers.hpp"

/*
This is the umbrella class for multiple exposures.
Each exposure builts its own covariance matrix. Derivative matrices are built
for cross exposures. Two exposures are cross correlated if wavelength overlap
is small.
*/
class Exposure: public Chunk
{
    std::unique_ptr<double[]> weighted_data, local_cov_mat;
public:
    Exposure(const qio::QSOFile &qmaster, int i1, int i2) : Chunk() {
        _copyQSOFile(qmaster, i1, i2);
        _setNQandFisherIndex();
        glmemory::setMaxSizes(size(), size(), 2 * N_Q_MATRICES, false);
        N_Q_MATRICES = 0;
    }
    Exposure(Exposure &&rhs) = delete;
    Exposure(const Exposure &rhs) = delete;
    ~Exposure() { if (local_cov_mat || weighted_data) deallocMatrices(); }

    int getExpId() const { return qFile->expid; };
    int getNight() const { return qFile->night; };
    int getFiber() const { return qFile->fiber; };
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
        process::updateMemory(process::getMemoryMB(DATA_SIZE_2 + size()));
    };

    void setCovarianceMatrix();
    void weightDataVector();
};

#endif

