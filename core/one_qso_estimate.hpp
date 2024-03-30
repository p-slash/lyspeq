#ifndef ONE_QSO_ESTIMATE_H
#define ONE_QSO_ESTIMATE_H

#include <string>
#include <vector>
#include <memory>
#include "core/chunk_estimate.hpp"

/*
This is the umbrella class for multiple chunks
Quadratic estimator is applied to each chunk individually
In terms of CPU time, all chunks are moved together.
Number of chunks decided dynamically:
nchunks = specifics::NUMBER_OF_CHUNKS * size / MAX_PIXELS_IN_FOREST+1;
*/
class OneQSOEstimate
{
protected:
    std::string fname_qso;

    std::unique_ptr<qio::QSOFile> _readQsoFile(const std::string &f_qso);
public:
    int istart, ndim;
    std::unique_ptr<double[]> fisher_matrix, theta_vector;

    std::vector<std::unique_ptr<Chunk>> chunks;

    OneQSOEstimate() {};
    OneQSOEstimate(const std::string &f_qso);
    OneQSOEstimate(OneQSOEstimate &&rhs) = default;
    OneQSOEstimate(const OneQSOEstimate &rhs) = delete;

    static std::vector<int> decideIndices(int size);
    static double getComputeTimeEst(std::string fname_qso, int &zbin);

    // Pass fit values for the power spectrum for numerical stability
    void oneQSOiteration(
        const double *ps_estimate, 
        std::vector<std::unique_ptr<double[]>> &dbt_sum_vector, 
        double *fisher_sum);

    void collapseBootstrap();
    void addBoot(int p, double *temppower, double* tempfisher) {
        double *outfisher = tempfisher + (bins::TOTAL_KZ_BINS + 1) * istart;

        for (int i = 0; i < ndim; ++i) {
            for (int j = i; j < ndim; ++j) {
                outfisher[j + i * bins::TOTAL_KZ_BINS] += p * fisher_matrix[j + i * ndim];
            } 
        }

        cblas_daxpy(ndim, p, theta_vector.get(), 1, temppower + istart, 1);
    }

    void addBootPowerOnly(int nboots, double *pcoeff, double *temppower) {
        // cblas_daxpy(ndim, p, theta_vector.get(), 1, temppower + istart, 1);
        cblas_dger(
            CblasRowMajor, nboots, ndim, 1, pcoeff, 1, theta_vector.get(), 1,
            temppower + istart, bins::TOTAL_KZ_BINS);
    };
};

#endif

