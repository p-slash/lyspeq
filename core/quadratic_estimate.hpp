#ifndef QUADRATIC_ESTIMATE_H
#define QUADRATIC_ESTIMATE_H

#include <vector>
#include <string>

#include <gsl/gsl_matrix.h> 
#include <gsl/gsl_vector.h>

#include "core/one_qso_estimate.hpp"

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
    int NUMBER_OF_QSOS, NUMBER_OF_QSOS_OUT, *Z_BIN_COUNTS;

    std::vector< std::pair <double, std::string> > cpu_fname_vector;

    // 3 TOTAL_KZ_BINS sized vectors
    gsl_vector  *dbt_estimate_sum_before_fisher_vector[3],
                *dbt_estimate_fisher_weighted_vector[3], 
                *previous_power_estimate_vector, *current_power_estimate_vector;

    // 2 TOTAL_KZ_BINS x TOTAL_KZ_BINS sized matrices
    gsl_matrix *fisher_matrix_sum, *inverse_fisher_matrix_sum;

    bool isFisherInverted;

    void _readQSOFiles(const char *fname_list, const char *dir);

    // Fitting procedure calls Python3 script lorentzian_fit.py.
    // Intermadiate files are saved in TMP_FOLDER (read as TemporaryFolder in config file)
    // make install will copy this script to $HOME/bin and make it executable.
    // Add $HOME/bin to your $PATH
    void _fitPowerSpectra(double *fit_values);

    // Performs a load balancing operation based on N^3 estimation
    void _loadBalancing(std::vector<OneQSOEstimate*> &local_queue);

public:
    OneDQuadraticPowerEstimate(const char *fname_list, const char *dir);

    ~OneDQuadraticPowerEstimate();
    
    double powerSpectrumFiducial(int kn, int zm);

    void initializeIteration();
    void invertTotalFisherMatrix();
    void computePowerSpectrumEstimates();
    void iterate(int number_of_iterations, const char *fname_base);
    bool hasConverged();
    
    void printfSpectra();
    void writeFisherMatrix(const char *fname);

    // Does not write the last bin since it is ignored when LAST_K_EDGE defined
    // You can find that value in logs--printfSpectra prints all
    void writeSpectrumEstimates(const char *fname);
    void writeDetailedSpectrumEstimates(const char *fname);
};


#endif

