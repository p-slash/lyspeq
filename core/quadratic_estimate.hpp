/* This umbrella class manages the quadratic estimator by 
 *          storing the total Fisher matrix,
 *          computing the power spectrum estimate,
 *          and fitting a smooth function to the power spectrum estimate (now removed).
 * It reads a file which should start with number of quasars followed by a list of quasar files.
 */

#ifndef QUADRATIC_ESTIMATE_H
#define QUADRATIC_ESTIMATE_H

#include "one_qso_estimate.hpp"
#include "fiducial_cosmology.hpp"

typedef struct
{
    OneQSOEstimate *qso;
    double est_cpu_time;
} qso_computation_time;


class OneDQuadraticPowerEstimate
{
    int NUMBER_OF_QSOS, \
        NUMBER_OF_QSOS_OUT, \
       *Z_BIN_COUNTS;

    struct palanque_fit_params *FIDUCIAL_PS_PARAMS;

    qso_computation_time *qso_estimators;

    /* TOTAL_KZ_BINS sized vector */ 
    gsl_vector  *pmn_before_fisher_estimate_vector_sum, \
                *previous_pmn_estimate_vector, \
                *pmn_estimate_vector;

    /* TOTAL_KZ_BINS x TOTAL_KZ_BINS sized matrix */
    gsl_matrix  *fisher_matrix_sum,\
                *inverse_fisher_matrix_sum;

    bool isFisherInverted;

    void fitPowerSpectra(double *fit_values);

public:
    OneDQuadraticPowerEstimate( const char *fname_list, const char *dir, \
                                struct palanque_fit_params *pfp);

    ~OneDQuadraticPowerEstimate();
    
    double powerSpectrumFiducial(int kn, int zm);

    void initializeIteration();
    void invertTotalFisherMatrix();
    void computePowerSpectrumEstimates();
    void iterate(int number_of_iterations, const char *fname_base);
    bool hasConverged();
    
    void printfSpectra();
    void write_fisher_matrix(const char *fname_base);
    void write_spectrum_estimates(const char *fname_base);
};


#endif

