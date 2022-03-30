#ifndef ONE_QSO_ESTIMATE_H
#define ONE_QSO_ESTIMATE_H

#include <string>
#include "core/chunk_estimate.hpp"

class OneQSOEstimate
{
protected:
    std::vector<Chunk> chunks;
    std::vector<int> indices;

public:
    OneQSOEstimate(std::string fname_qso);

    ~OneQSOEstimate();

    // Move constructor 
    OneQSOEstimate(OneQSOEstimate &&rhs) : 
    chunks(std::move(rhs.chunks)), indices(std::move(rhs.indices)) {};
    OneQSOEstimate& operator=(const OneQSOEstimate& rhs) = default;

    static double getComputeTimeEst(std::string fname_qso, int &zbin);

    // Pass fit values for the power spectrum for numerical stability
    void oneQSOiteration(const double *ps_estimate, double *dbt_sum_vector[3], double *fisher_sum);
};

#endif

