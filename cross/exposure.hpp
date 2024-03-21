#ifndef EXPOSURE_H
#define EXPOSURE_H

#include <string>
#include <vector>
#include <memory>

#include "core/chunk_estimate.hpp"

/*
This is the umbrella class for multiple exposures.
Each exposure builts its own covariance matrix. Derivative matrices are built
for cross exposures. Two exposures are cross correlated if wavelength overlap
is small.
*/
class Exposure: public Chunk
{
    std::unique_ptr<double[]> weighted_data;
public:
    Exposure(const qio::QSOFile &qmaster, int i1, int i2)
        : Chunk(qmaster, i1, i2) {};
    int getExpId() const { return qFile->expid; };
    int getNight() const { return qFile->night; };
    int getFiber() const { return qFile->fiber; };

    void initMatrices();
    void deallocMatrices() { delete [] covariance_matrix; };

    void setCovarianceMatrix();
    void invertCovarianceMatrix();
    void weightDataVector();
};

#endif

