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
        glmemory::setMaxSizes(size(), size(), 0, false);
        process::updateMemory(-getMinMemUsage());
    };

    int getExpId() const { return qFile->expid; };
    int getNight() const { return qFile->night; };
    int getFiber() const { return qFile->fiber; };
    double* getWeightedData() const { return weighted_data.get(); };

    double getMinMemUsage() {
        if (qFile)
            return qFile->getMinMemUsage();
        return 0;
    }

    void initMatrices();
    void deallocMatrices() {
        local_cov_mat.reset();
        weighted_data.reset();
    };

    void setCovarianceMatrix();
    void weightDataVector();
};

#endif

