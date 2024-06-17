#ifndef ONE_QSO_EXPOSURES_H
#define ONE_QSO_EXPOSURES_H

#include <string>
#include <vector>
#include <set>
#include <memory>

#include "core/one_qso_estimate.hpp"
#include "cross/exposure.hpp"

typedef std::pair<const Exposure*, const Exposure*> cExpoCombo;

namespace specifics {
    extern bool X_NIGHT, X_FIBER, X_PETAL;
    extern double X_WAVE_OVERLAP_RATIO;
}

/*
This is the umbrella class for multiple exposures.
Each exposure builts its own covariance matrix. Derivative matrices are built
for cross exposures. Two exposures are cross correlated if the wavelength
overlap is large (skipCombo in cpp file).
*/
class OneQsoExposures: public OneQSOEstimate 
{
    std::set<int> unique_expid_set;
    std::vector<cExpoCombo> exposure_combos;
public:
    double z_qso, ra, dec;
    long targetid;
    std::vector<std::unique_ptr<Exposure>> exposures;
    std::vector<std::unique_ptr<double[]>> dbt_estimate_before_fisher_vector;

    OneQsoExposures(const std::string &f_qso);
    OneQsoExposures(OneQsoExposures &&rhs) = default;
    OneQsoExposures(const OneQsoExposures &rhs) = delete;

    void addExposures(OneQsoExposures *other);

    bool hasEnoughUniqueExpids() {
        bool hasit = unique_expid_set.size() > 1;

        if (!hasit)
            return false;

        return countExposureCombos() != 0;
    }

    int countExposureCombos();
    void setAllocPowerSpMemory();
    void xQmlEstimate();
    bool isAnOutlier();
    // Pass fit values for the power spectrum for numerical stability
    int oneQSOiteration(
        std::vector<std::unique_ptr<double[]>> &dt_sum_vector, 
        double *fisher_sum);
};

#endif

