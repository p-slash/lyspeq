#ifndef ONE_QSO_EXPOSURES_H
#define ONE_QSO_EXPOSURES_H

#include <string>
#include <vector>
#include <set>
#include <memory>

#include "core/one_qso_estimate.hpp"
#include "cross/exposure.hpp"

/*
This is the umbrella class for multiple exposures.
Each exposure builts its own covariance matrix. Derivative matrices are built
for cross exposures. Two exposures are cross correlated if wavelength overlap
is small.
*/
class OneQsoExposures: public OneQSOEstimate 
{
    std::set<std::pair<int, int>> unique_expid_night_set;
public:
    double z_qso, ra, dec;
    long targetid;
    std::vector<std::unique_ptr<Exposure>> exposures;
    std::vector<std::unique_ptr<double[]>> dbt_estimate_before_fisher_vector;

    OneQsoExposures(const std::string &f_qso);
    OneQsoExposures(OneQsoExposures &&rhs) = default;
    OneQsoExposures(const OneQsoExposures &rhs) = delete;

    void addExposures(OneQsoExposures *other);

    bool hasEnoughUniqueExpidNightPairs() const {
        bool hasit = unique_expid_night_set.size() > 1;

        if (!hasit)
            return false;

        return countExposureCombos() != 0;
    }

    int countExposureCombos() const;
    void setAllocPowerSpMemory();
    void xQmlEstimate();
    // Pass fit values for the power spectrum for numerical stability
    int oneQSOiteration(
        std::vector<std::unique_ptr<double[]>> &dt_sum_vector, 
        double *fisher_sum);

    std::unique_ptr<OneQSOEstimate> move2OneQSOEstimate();
};

#endif

