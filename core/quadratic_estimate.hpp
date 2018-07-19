/* This umbrella class manages the quadratic estimator by 
 *          storing the total Fisher matrix,
 *          computing the power spectrum estimate,
 *          and fitting a smooth function to the power spectrum estimate.
 * It reads a file which should start with number of quasars followed by a list of quasar files.
 */

#ifndef QUADRATIC_ESTIMATE_H
#define QUADRATIC_ESTIMATE_H

#include "one_qso_estimate.hpp"
#include "../gsltools/ln_poly_fit.hpp"

extern int POLYNOMIAL_FIT_DEGREE;

class OneDQuadraticPowerEstimate
{
    int NUMBER_OF_BANDS, \
        NUMBER_OF_QSOS;

    const double *kband_edges;

    OneQSOEstimate **qso_estimators;

    /* NUMBER_OF_BANDS sized vector */ 
    gsl_vector  *ps_before_fisher_estimate_vector_sum, \
                *previous_power_spectrum_estimate_vector, \
                *power_spectrum_estimate_vector;

    /* NUMBER_OF_BANDS x NUMBER_OF_BANDS sized matrices 
       inverse_fisher_matrix points to fisher_matrix
    */
    gsl_matrix  *fisher_matrix_sum, \
                *inverse_fisher_matrix_sum;

    bool isFisherInverted;

    LnPolynomialFit *fit_to_power_spectrum;
    double *weights_ps_bands, *k_centers;

public:
    OneDQuadraticPowerEstimate( const char *fname_list, const char *dir, \
                                int no_bands, \
                                const double *k_edges);

    ~OneDQuadraticPowerEstimate();
    
    void setInitialPSestimateFFT();
    void setInitialScaling();
    
    void invertTotalFisherMatrix();
    void computePowerSpectrumEstimate();
    void iterate(int number_of_iterations);
    bool hasConverged();
    
    void write_spectrum_estimate(const char *fname);
};


#endif

