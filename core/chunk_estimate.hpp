#ifndef CHUNK_ESTIMATE_H
#define CHUNK_ESTIMATE_H

#include <string>

#include "io/qso_file.hpp"
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

class Chunk
{
protected:
    qio::QSOFile *qFile;

    int _matrix_n;

    int RES_INDEX;
    int N_Q_MATRICES, fisher_index_start;

    double LOWER_REDSHIFT, UPPER_REDSHIFT, MEDIAN_REDSHIFT, BIN_REDSHIFT;
    // DATA_SIZE sized vectors
    double *highres_lambda;

    // DATA_SIZE x DATA_SIZE sized matrices 
    // Note that noise matrix is diagonal and stored as pointer to its array 
    double  *covariance_matrix, *inverse_covariance_matrix, *temp_matrix[2];

    double  **stored_qj, *stored_sfid;
    int       nqj_eff;
    bool      isQjSet, isSfidSet, isSfidStored;

    bool isCovInverted;

    DiscreteInterpolation2D  *interp2d_signal_matrix;
    DiscreteInterpolation1D **interp_derivative_matrix;

    // 3 TOTAL_KZ_BINS sized vectors
    double  *dbt_estimate_before_fisher_vector[3];

    // TOTAL_KZ_BINS x TOTAL_KZ_BINS sized matrix
    double  *fisher_matrix;

    void _copyQSOFile(const qio::QSOFile *qmaster, double l1, double l2);
    void _findRedshiftBin();
    void _setNQandFisherIndex();
    void _setStoredMatrices();

    void _allocateMatrices();
    void _freeMatrices();
    // void _saveIndividualResult();

    void _setFiducialSignalMatrix(double *&sm, bool copy=true);
    void _setQiMatrix(double *&qi, int i_kz, bool copy=true);
    void _addMarginalizations();
    void _getWeightedMatrix(double *m);
    void _getFisherMatrix(const double *Q_ikz_matrix, int i_kz);

public:
    int ZBIN, ZBIN_LOW, ZBIN_UPP;

    Chunk(const qio::QSOFile *qmaster, double l1, double l2);

    ~Chunk();

    // Move constructor 
    Chunk(Chunk &&rhs);
    Chunk& operator=(const Chunk& rhs) = default;

    static double getComputeTimeEst(const qio::QSOFile &qmaster, double l1, double l2);

    void setCovarianceMatrix(const double *ps_estimate);
    void invertCovarianceMatrix();

    void computePSbeforeFvector();
    void computeFisherMatrix();

    // Pass fit values for the power spectrum for numerical stability
    void oneQSOiteration(const double *ps_estimate, double *dbt_sum_vector[3], double *fisher_sum);

    void fprintfMatrices(const char *fname_base);
    double getMinMemUsage();
};

#endif

