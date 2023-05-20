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
    // Emplace_back with vector<OneQSOEstimate> leaks memory!!
    std::vector<int> indices;

public:
    std::vector<std::unique_ptr<Chunk>> chunks;

    OneQSOEstimate(const std::string &f_qso);
    OneQSOEstimate(OneQSOEstimate &&rhs) = default;
    OneQSOEstimate(const OneQSOEstimate &rhs) = delete;

    static double getComputeTimeEst(std::string fname_qso, int &zbin);

    // Pass fit values for the power spectrum for numerical stability
    void oneQSOiteration(const double *ps_estimate, 
        std::vector<std::unique_ptr<double[]>> &dbt_sum_vector, 
        double *fisher_sum);
};

#endif

