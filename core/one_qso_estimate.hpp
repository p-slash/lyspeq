#ifndef ONE_QSO_ESTIMATE_H
#define ONE_QSO_ESTIMATE_H

#include <gsl/gsl_matrix.h> 

class OneQSOEstimate
{
    int DATA_SIZE, \
        NUMBER_OF_BANDS;

    const double *kband_edges;

    /* DATA_SIZE sized vectors */ 
    double *xspace_array, \
           *data_array, \
           *noise_array;

    /* DATA_SIZE x DATA_SIZE sized matrices 
       Note that noise matrix is diagonal and stored as pointer to its array 
       inverse_covariance_matrix is not allocated; it points to covariance_matrix
    */
    gsl_matrix  *covariance_matrix, \
                *inverse_covariance_matrix, \
                *signal_matrix;

    /* NUMBER_OF_BANDS many DATA_SIZE x DATA_SIZE sized matrices */
    gsl_matrix  **derivative_of_signal_matrices, \
                **modified_derivative_of_signal_matrices;

    bool isCovInverted, isQMatricesSet;

public:
    /* NUMBER_OF_BANDS sized vector */ 
    gsl_vector  *ps_before_fisher_estimate_vector;

    /* NUMBER_OF_BANDS x NUMBER_OF_BANDS sized matrices */
    gsl_matrix  *fisher_matrix;

    OneQSOEstimate(const char *fname_qso, int n, const double *k);
    ~OneQSOEstimate();

    void getFFTEstimate(double *ps, int *bincount);

    void setDerivativeSMatrices();
    void computeCSMatrices(const double *ps_estimate);
    void invertCovarianceMatrix();
    void computeModifiedDSMatrices();

    void computePSbeforeFvector();
    void computeFisherMatrix();

    void oneQSOiteration(const double *ps_estimate);
};

#endif

