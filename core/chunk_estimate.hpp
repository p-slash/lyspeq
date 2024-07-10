#ifndef CHUNK_ESTIMATE_H
#define CHUNK_ESTIMATE_H

#include <memory>
#include "io/qso_file.hpp"
#include "mathtools/discrete_interpolation.hpp"

const int
MIN_PIXELS_IN_CHUNK = 40;

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

namespace glmemory {
    extern int max_size_2;
    extern std::unique_ptr<double[]>
        covariance_matrix, temp_vector, weighted_data_vector,
        stored_sfid, matrix_lambda, finer_matrix, v_matrix, z_matrix;
    extern std::vector<std::unique_ptr<double[]>> stored_ikz_qi, temp_matrices;

    extern shared_interp_2d interp2d_signal_matrix;
    extern std::vector<shared_interp_1d> interp_derivative_matrix;

    extern void setMaxSizes(int size, int matrix_n, int nqdim, bool onsamp);
    extern void allocMemory();
    extern void dealloc();
    extern void updateMemUsed(double mem);
}


class Chunk
{
protected:
    int DATA_SIZE_2;

    int _matrix_n, RES_INDEX;
    double LOWER_REDSHIFT, UPPER_REDSHIFT, MEDIAN_REDSHIFT, BIN_REDSHIFT;
    // Will have finer spacing when rmat is oversampled
    double *_matrix_lambda, *inverse_covariance_matrix; // Do not delete!

    // Uninitialized arrays
    // Oversampled resomat specifics
    bool on_oversampling;
    double *_vmatrix, *_zmatrix;

    // DATA_SIZE x DATA_SIZE sized matrices 
    // Note that noise matrix is diagonal and stored as pointer to its array 
    double *covariance_matrix;
    std::vector<double*> temp_matrix;
    std::vector<std::pair<int, double*>> stored_ikz_qi;

    void _copyQSOFile(const qio::QSOFile &qmaster, int i1, int i2);
    void _findRedshiftBin();
    void _setNQandFisherIndex();
    void _setStoredMatrices();

    void _initMatrices();
    // void _saveIndividualResult();

    void _setFiducialSignalMatrix(double *sm);
    void _setVZMatrices();
    void _setQiMatrix(double *qi, int i_kz);
    void _applyRedshiftInterp();
    void _scaleDerivativesWithRedshiftGrowth();
    void _addMarginalizations();
    void _getWeightedMatrix(double *&m);

    friend class TestOneQSOEstimate;

    Chunk() : DATA_SIZE_2(0), _matrix_n(0), N_Q_MATRICES(0) {};
public:
    std::unique_ptr<qio::QSOFile> qFile;
    int fisher_index_start, N_Q_MATRICES;
    int ZBIN, ZBIN_LOW, ZBIN_UPP, KBIN_UPP;

    // Initialized to 0
    // 3 TOTAL_KZ_BINS sized vectors
    std::vector<std::unique_ptr<double[]>> dbt_estimate_before_fisher_vector;
    // TOTAL_KZ_BINS x TOTAL_KZ_BINS sized matrix
    std::unique_ptr<double[]> fisher_matrix;

    Chunk(const qio::QSOFile &qmaster, int i1, int i2);
    Chunk(Chunk &&rhs) = delete;
    Chunk(const Chunk &rhs) = delete;
    ~Chunk();

    static double getComputeTimeEst(const qio::QSOFile &qmaster, int i1, int i2);

    int realSize() const { return qFile->realSize(); };
    int size() const { return qFile->size(); };
    void setInvertCovarianceMatrix(const double *ps_estimate);

    void computePSbeforeFvector();

    // Pass fit values for the power spectrum for numerical stability
    void oneQSOiteration(
        const double *ps_estimate,
        std::vector<std::unique_ptr<double[]>> &dbt_sum_vector,
        double *fisher_sum);

    void fprintfMatrices(const char *fname_base);
    double getMinMemUsage();
    void releaseFile() { qFile.reset(); };
};

#endif

