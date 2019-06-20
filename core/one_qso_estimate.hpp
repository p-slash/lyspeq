#ifndef ONE_QSO_ESTIMATE_H
#define ONE_QSO_ESTIMATE_H

#include <gsl/gsl_matrix.h> 
#include <gsl/gsl_vector.h>
#include "core/sq_table.hpp"

int  getFisherMatrixIndex(int kn, int zm);
void getFisherMatrixBinNoFromIndex(int i, int &kn, int &zm);

// This object creates and computes C, S, Q, Q-slash matrices,
// as well as a power spectrum estimate and a fisher matrix for individual quasar spectrum.
// Matrices are not stored indefinitely. They are allocated when needed and deleted when done.

// Construct it with file path to qso spectrum. The binary file structes is given in io/qso_file.hpp

// This object is called in OneDQuadraticPowerEstimate in quadratic_estimate.hpp
// It takes the filename for a quasar spectrum in constructor.
// Quasar spectrum file consists of a header followed by lambda, flux and noise. 
// Wavelength is then converted into v spacing around the median lambda.

// The most efficient memory usage is 3 temp matrices
// Saves more derivative matrices according to MEMORY_ALLOC (read as AllocatedMemoryMB in config file)
// Fiducial signal matrix if there is still more space after all derivative matrices.
// This scheme speeds up the algorithm.
class OneQSOEstimate
{
    char qso_sp_fname[250];
    
    int SPECT_RES_FWHM;

    int N_Q_MATRICES, fisher_index_start, r_index;

    double  MEDIAN_REDSHIFT, BIN_REDSHIFT, \
            DV_KMS;
    
    // DATA_SIZE sized vectors
    double *lambda_array, \
           *velocity_array, \
           *flux_array, \
           *noise_array;

    // DATA_SIZE x DATA_SIZE sized matrices 
    // Note that noise matrix is diagonal and stored as pointer to its array 
    gsl_matrix  *covariance_matrix, \
                *inverse_covariance_matrix, \
                *temp_matrix[2];

    gsl_matrix  **stored_qj, *stored_sfid;
    int           nqj_eff;
    bool          isQjSet, isSfidSet, isSfidStored;

    bool isCovInverted;

    void allocateMatrices();
    void freeMatrices();

    void setFiducialSignalMatrix(gsl_matrix *sm);
    void setQiMatrix(gsl_matrix *qi, int i_kz);
    void getWeightedMatrix(gsl_matrix *m);
    void getFisherMatrix(const gsl_matrix *Q_ikz_matrix, int i_kz);

public:
    int ZBIN, DATA_SIZE;

    // TOTAL_KZ_BINS sized vector 
    gsl_vector  *ps_before_fisher_estimate_vector;

    // TOTAL_KZ_BINS x TOTAL_KZ_BINS sized matrices
    gsl_matrix  *fisher_matrix;

    OneQSOEstimate(const char *fname_qso);
    ~OneQSOEstimate();

    void setCovarianceMatrix(const double *ps_estimate);
    void invertCovarianceMatrix();

    void computePSbeforeFvector();
    void computeFisherMatrix();

    void oneQSOiteration(const double *ps_estimate, gsl_vector *pmn_before, gsl_matrix *fisher_sum);
};

#endif

