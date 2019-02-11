/* This object stores and computes C, S, Q, Q-slash matrices,
 * as well as a power spectrum estimate and a fisher matrix for individual quasar spectra.
 * This object is called in OneDQuadraticPowerEstimate in quadratic_estimate.hpp
 * It takes the file for quasar spectra, n=NUMBER_OF_BANDS and k points to kband_edges.
 * Quasar spectrum file consists of a header followed by lambda, flux and noise. 
 * Wavelength is then converted into v spacing around the mean lambda.
 */

#ifndef ONE_QSO_ESTIMATE_H
#define ONE_QSO_ESTIMATE_H

#include <gsl/gsl_matrix.h> 
#include "sq_table.hpp"

int getFisherMatrixIndex(int kn, int zm);
void getFisherMatrixBinNoFromIndex(int i, int &kn, int &zm);

class OneQSOEstimate
{
    char qso_sp_fname[250];
    
    int DATA_SIZE, \
        SPECT_RES_FWHM;

    int N_Q_MATRICES, fisher_index_start;

    double  MEDIAN_REDSHIFT, BIN_REDSHIFT, \
            DV_KMS;
    
    /* DATA_SIZE sized vectors */ 
    double *lambda_array, \
           *velocity_array, \
           *flux_array, \
           *noise_array;

    /* DATA_SIZE x DATA_SIZE sized matrices 
       Note that noise matrix is diagonal and stored as pointer to its array 
    */
    gsl_matrix  *covariance_matrix, *inverse_covariance_matrix, \
                *fiducial_signal_matrix, *weighted_fiducial_signal_matrix, \
                *weighted_noise_matrix;

    /* NUMBER_OF_K_BANDS * N_Q_MATRICES (1,2,3) many DATA_SIZE x DATA_SIZE sized matrices */
    gsl_matrix  **derivative_of_signal_matrices, \
                **weighted_derivative_of_signal_matrices;

    bool isCovInverted;

    void allocateMatrices();
    void freeMatrices();

    void dump_all_matrices();

public:
    int ZBIN;

    /* TOTAL_KZ_BINS sized vector */ 
    gsl_vector  *ps_before_fisher_estimate_vector;

    /* TOTAL_KZ_BINS x TOTAL_KZ_BINS sized matrices */
    gsl_matrix  *fisher_matrix;

    OneQSOEstimate(const char *fname_qso);
    ~OneQSOEstimate();

    void setFiducialSignalAndDerivativeSMatrices(const SQLookupTable *sq_lookup_table);
    void computeCSMatrices(const gsl_vector *ps_estimate);
    void invertCovarianceMatrix();
    void computeWeightedMatrices();

    void computePSbeforeFvector();
    void computeFisherMatrix();

    void oneQSOiteration(   const gsl_vector *ps_estimate, \
                            const SQLookupTable *sq_lookup_table, \
                            gsl_vector *pmn_before, gsl_matrix *fisher_sum);
};

#endif

