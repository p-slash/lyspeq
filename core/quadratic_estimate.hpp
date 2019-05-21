#ifndef QUADRATIC_ESTIMATE_H
#define QUADRATIC_ESTIMATE_H

#include "one_qso_estimate.hpp"
#include "fiducial_cosmology.hpp"
#include <vector>

typedef struct
{
    OneQSOEstimate *qso;
    double est_cpu_time;
} qso_computation_time;


// This umbrella class manages the quadratic estimator by 
//      storing the total Fisher matrix and its inverse,
//      computing the power spectrum estimate,
//      and fitting a smooth function to the power spectrum estimate using Python3 script.

// This object performs a load balancing operation based on N^3 estimation
// and skips chunks that do not belong to any redshift bin
//

// It takes + a file path which should start with number of quasars followed by a list of quasar files,
//          + the directory these qso placed as const char *dir
//          + Fiducial fit parameters as defined in fiducial_cosmology.hpp
// Call iterate with max number of iterations and output filename base to const char *fname_base
//      fname_base should have output folder and output filebase in one string

class OneDQuadraticPowerEstimate
{
    int NUMBER_OF_QSOS, \
        NUMBER_OF_QSOS_OUT, \
       *Z_BIN_COUNTS;

    pd13_fit_params *FIDUCIAL_PS_PARAMS;

    qso_computation_time *qso_estimators;

    // TOTAL_KZ_BINS sized vector
    gsl_vector  *pmn_before_fisher_estimate_vector_sum, \
                *previous_pmn_estimate_vector, \
                *pmn_estimate_vector;

    // TOTAL_KZ_BINS x TOTAL_KZ_BINS sized matrix
    gsl_matrix  *fisher_matrix_sum,\
                *inverse_fisher_matrix_sum;

    bool isFisherInverted;

    // Fitting procedure calls Python3 script lorentzian_fit.py.
    // Intermadiate files are saved in TMP_FOLDER (read as TemporaryFolder in config file)
    // make install will copy this script to $HOME/bin and make it executable.
    // Add $HOME/bin to your $PATH
    void fitPowerSpectra(double *fit_values);

    // Performs a load balancing operation based on N^3 estimation
    void loadBalancing(std::vector<qso_computation_time*> *queue_qso, int maxthreads);

public:
    OneDQuadraticPowerEstimate(const char *fname_list, const char *dir, pd13_fit_params *pfp);

    ~OneDQuadraticPowerEstimate();
    
    double powerSpectrumFiducial(int kn, int zm);

    void initializeIteration();
    void invertTotalFisherMatrix();
    void computePowerSpectrumEstimates();
    void iterate(int number_of_iterations, const char *fname_base);
    bool hasConverged();
    
    void printfSpectra();
    void write_fisher_matrix(const char *fname_base);

    // Does not write the last bin since it is ignored
    // You can find that value in logs--printfSpectra prints all
    void write_spectrum_estimates(const char *fname_base);
};


#endif

