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

class OneQSOEstimate
{
    int DATA_SIZE, \
        NUMBER_OF_BANDS;

    const double *kband_edges;
    double median_redshift, spect_res, dv_kms;
    
    /* DATA_SIZE sized vectors */ 
    double *lambda_array, \
           *velocity_array, \
           *flux_array, \
           *noise_array;

    /* DATA_SIZE x DATA_SIZE sized matrices 
       Note that noise matrix is diagonal and stored as pointer to its array 
       inverse_covariance_matrix is not allocated; it points to covariance_matrix
    */
    gsl_matrix  *covariance_matrix, \
                *inverse_covariance_matrix, \
                *fiducial_signal_matrix;

    /* NUMBER_OF_BANDS many DATA_SIZE x DATA_SIZE sized matrices */
    gsl_matrix  **derivative_of_signal_matrices, \
                **modified_derivative_of_signal_matrices;

    bool isCovInverted, areSQMatricesSet;

public:
    /* NUMBER_OF_BANDS sized vector */ 
    gsl_vector  *ps_before_fisher_estimate_vector;

    /* NUMBER_OF_BANDS x NUMBER_OF_BANDS sized matrices */
    gsl_matrix  *fisher_matrix;

    OneQSOEstimate(const char *fname_qso, int n, const double *k);
    ~OneQSOEstimate();

    void getFFTEstimate(double *ps, int *bincount);

    void setFiducialSignalMatrix();
    void setDerivativeSMatrices();
    void computeCSMatrices(const double *ps_estimate);
    void invertCovarianceMatrix();
    void computeModifiedDSMatrices();

    void computePSbeforeFvector();
    void computeFisherMatrix();

    void oneQSOiteration(const double *ps_estimate);
};

#endif

