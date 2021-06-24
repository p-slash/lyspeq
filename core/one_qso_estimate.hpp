#ifndef ONE_QSO_ESTIMATE_H
#define ONE_QSO_ESTIMATE_H

#include <string>

#include "gsltools/discrete_interpolation.hpp"

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
protected:
    std::string qso_sp_fname;
    
    int SPECT_RES_FWHM, RES_INDEX;

    int DATA_SIZE, N_Q_MATRICES, fisher_index_start, r_index;

    double LOWER_REDSHIFT, UPPER_REDSHIFT, MEDIAN_REDSHIFT, BIN_REDSHIFT, DV_KMS;
    
    // DATA_SIZE sized vectors
    double *lambda_array, *velocity_array, *flux_array, *noise_array;

    // DATA_SIZE x DATA_SIZE sized matrices 
    // Note that noise matrix is diagonal and stored as pointer to its array 
    double  *covariance_matrix, *inverse_covariance_matrix, *temp_matrix[2];
    const double *cur_iter_thetas;

    double  **stored_qj, *stored_sfid;
    int       nqj_eff;
    bool      isQjSet, isSfidSet, isSfidStored;

    bool isCovInverted;

    DiscreteInterpolation2D  *interp2d_signal_matrix, *interp2d_smu_matrix;
    DiscreteInterpolation1D **interp_derivative_matrix, **interp_qmu_matrices;

    // 3 TOTAL_KZ_BINS sized vectors
    double  *dbt_estimate_before_fisher_vector[3];

    // TOTAL_KZ_BINS x TOTAL_KZ_BINS sized matrix
    double  *fisher_matrix;
    
    void _readFromFile(std::string fname_qso);
    bool _findRedshiftBin();
    void _setNQandFisherIndex();
    void _setStoredMatrices();

    void _allocateMatrices();
    void _freeMatrices();
    void _saveIndividualResult();

    void _getVandZ(double &v_ij, double &z_ij, int i, int j);

    void _setFiducialSignalMatrix(double *&sm, bool copy=true);
    void _setQiMatrix(double *&qi, int i_kz, bool copy=true);
    void _setQmuMatrix(double *&qi);
    void _getWeightedMatrix(double *m);
    void _getFisherMatrix(const double *Q_ikz_matrix, int i_kz);

public:
    int ZBIN, ZBIN_LOW, ZBIN_UPP;

    OneQSOEstimate(std::string fname_qso);
    
    ~OneQSOEstimate();
    
    // Move constructor 
    // OneQSOEstimate(OneQSOEstimate &&rhs);
    // OneQSOEstimate& operator=(OneQSOEstimate&& rhs);

    double getComputeTimeEst();

    void setCovarianceMatrix();
    void invertCovarianceMatrix();

    void computePSbeforeFvector();
    void computeFisherMatrix();

    // Pass fit values for the power spectrum for numerical stability
    void oneQSOiteration(const double *ps_estimate, double *dbt_sum_vector[3], double *fisher_sum);
};

#endif

