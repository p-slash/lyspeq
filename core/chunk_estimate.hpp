#ifndef CHUNK_ESTIMATE_H
#define CHUNK_ESTIMATE_H

#include <memory>
#include "io/qso_file.hpp"
#include "mathtools/discrete_interpolation.hpp"

const int
MIN_PIXELS_IN_CHUNK = 20;

/*
This object creates and computes C, S, Q, Q-slash matrices,
as well as a power spectrum estimate and a fisher matrix for individual 
quasar spectrum chunk. Matrices are not stored indefinitely. 
They are allocated when needed and deleted when done.

Construct it with file path to qso spectrum. The binary file structes is given 
in io/qso_file.hpp

This object is called in OneQSOEstimate in one_qso_estimate.hpp
It copies the subset of the spectrum from a QSOFile.
Quasar spectrum file consists of a header followed by lambda, flux and noise. 
Wavelength is then converted into v spacing around the median lambda.

The most efficient memory usage is 3 temp matrices
Saves more derivative matrices according to MEMORY_ALLOC 
(read as AllocatedMemoryMB in config file)
Fiducial signal matrix if there is still more space after all derivative 
matrices. This scheme speeds up the algorithm.
*/

class Chunk
{
protected:
    std::unique_ptr<qio::QSOFile> qFile;
    int DATA_SIZE_2;

    int _kncut, _matrix_n, RES_INDEX, N_Q_MATRICES;
    int fisher_index_start;
    bool isQjSet, isCovInverted;
    double LOWER_REDSHIFT, UPPER_REDSHIFT, MEDIAN_REDSHIFT, BIN_REDSHIFT;
    // Will have finer spacing when rmat is oversampled
    double *_matrix_lambda; // Do not delete!

    // Uninitialized arrays
    // Oversampled resomat specifics
    double *_finer_lambda, *_finer_matrix;

    // DATA_SIZE x DATA_SIZE sized matrices 
    // Note that noise matrix is diagonal and stored as pointer to its array 
    __device__
    double  *covariance_matrix, *inverse_covariance_matrix,
            *weighted_data_vector; // DATA_SIZE sized vector
    __host__ __device__
    double  *temp_matrix[2], *stored_qj, *stored_sfid,
            *temp_vector; // DATA_SIZE sized vector.

    // Initialized to 0
    // 3 TOTAL_KZ_BINS sized vectors
    std::vector<std::unique_ptr<double[]>> dbt_estimate_before_fisher_vector;
    // TOTAL_KZ_BINS x TOTAL_KZ_BINS sized matrix
    std::unique_ptr<double[]> fisher_matrix;

    shared_interp_2d interp2d_signal_matrix;
    std::vector<shared_interp_1d> interp_derivative_matrix;

    void _copyQSOFile(const qio::QSOFile &qmaster, int i1, int i2);
    void _findRedshiftBin();
    void _setNQandFisherIndex();
    void _setStoredMatrices();
    bool _isAboveNyquist(int i_kz);
    double* _getStoredQikz(int i_kz) const
    { return stored_qj + (N_Q_MATRICES-i_kz-1) * DATA_SIZE_2; };

    void _allocateMatrices();
    void _freeMatrices();
    // void _saveIndividualResult();

    void _setFiducialSignalMatrix(double *sm);
    void _setQiMatrix(double *qi, int i_kz);
    void _addMarginalizations();
    void _getWeightedMatrix(double *m);
    void _getFisherMatrix(const double *Q_ikz_matrix, int i_kz);

    friend class TestOneQSOEstimate;

public:
    int ZBIN, ZBIN_LOW, ZBIN_UPP;

    Chunk(const qio::QSOFile &qmaster, int i1, int i2);
    Chunk(Chunk &&rhs) = delete;
    Chunk(const Chunk &rhs) = delete;
    ~Chunk();

    static double getComputeTimeEst(const qio::QSOFile &qmaster, int i1, int i2);

    int realSize() const { return qFile->realSize(); };
    int size() const { return qFile->size(); };
    void setCovarianceMatrix(const double *ps_estimate);
    void invertCovarianceMatrix();

    void computePSbeforeFvector();
    void computeFisherMatrix();

    // Pass fit values for the power spectrum for numerical stability
    void oneQSOiteration(const double *ps_estimate,
        std::vector<std::unique_ptr<double[]>> &dbt_sum_vector,
        double *fisher_sum);

    void fprintfMatrices(const char *fname_base);
    double getMinMemUsage();
};

#endif

