#include "core/one_qso_estimate.hpp"
#include "core/fiducial_cosmology.hpp"
#include "core/matrix_helper.hpp"
#include "core/global_numbers.hpp"

#include "io/io_helper_functions.hpp"
#include "io/qso_file.hpp"
#include "io/logger.hpp"

#include <gsl/gsl_cblas.h>
#include <gsl/gsl_errno.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#if defined(_OPENMP)
#include <omp.h>
#endif

void throw_isnan(double t, const char *step)
{
    char err_msg[25];
    sprintf(err_msg, "NaN in %s", step);

    if (std::isnan(t))   throw err_msg;
}

// For a top hat redshift bin, only access 1 redshift bin
// For triangular z bins, access 3 (2 for first and last z bins) redshift bins
void setNQandFisherIndex(int &nq, int &fi, int ZBIN)
{
    #ifdef TOPHAT_Z_BINNING_FN
    nq = 1;
    fi = bins::getFisherMatrixIndex(0, ZBIN);
    #endif

    #ifdef TRIANGLE_Z_BINNING_FN
    nq = 3;
    fi = bins::getFisherMatrixIndex(0, ZBIN - 1);

    if (ZBIN == 0) 
    {
        nq = 2;
        fi = 0;
    }
    else if (ZBIN == bins::NUMBER_OF_Z_BINS - 1) 
    {
        nq = 2;
    }
    #endif

    nq *= bins::NUMBER_OF_K_BANDS;
}

void OneQSOEstimate::_readFromFile(const char *fname_qso)
{
    qso_sp_fname = fname_qso;

    // Construct and read data arrays
    QSOFile qFile(qso_sp_fname.c_str());

    double dummy_qso_z, dummy_s2n;

    qFile.readParameters(DATA_SIZE, dummy_qso_z, SPECT_RES_FWHM, dummy_s2n, DV_KMS);

    LOG::LOGGER.IO("Reading from %s.\n" "Data size is %d\n" "Pixel Width is %.1f\n" "Spectral Resolution is %d.\n", 
        qso_sp_fname.c_str(), DATA_SIZE, DV_KMS, SPECT_RES_FWHM);

    lambda_array    = new double[DATA_SIZE];
    velocity_array  = new double[DATA_SIZE];
    flux_array      = new double[DATA_SIZE];
    noise_array     = new double[DATA_SIZE];

    qFile.readData(lambda_array, flux_array, noise_array);

    // Find the resolution index for the look up table
    r_index = sq_private_table->findSpecResIndex(SPECT_RES_FWHM);
    
    if (r_index == -1)      throw "SPECRES not found in tables!";
}

bool OneQSOEstimate::_findRedshiftBin(double median_z)
{
    // Assign to a redshift bin according to median redshift of this chunk
    ZBIN = bins::findRedshiftBin(median_z);
    
    if (ZBIN >= 0 && ZBIN < bins::NUMBER_OF_Z_BINS)
        BIN_REDSHIFT = bins::ZBIN_CENTERS[ZBIN];
    else
    {
        LOG::LOGGER.IO("This QSO does not belong to any redshift bin!\n"); 
        return false;
    }

    return true;
}

void OneQSOEstimate::_setStoredMatrices()
{
    // Number of Qj matrices to preload.
    double size_m1 = (double)sizeof(double) * DATA_SIZE * DATA_SIZE / 1048576.; // in MB
    
    // Need at least 3 matrices as temp
    nqj_eff      = MEMORY_ALLOC / size_m1 - 3;
    isSfidStored = false;
    
    if (nqj_eff <= 0 )   nqj_eff = 0;
    else 
    {
        if (nqj_eff > N_Q_MATRICES)
        {
            nqj_eff      = N_Q_MATRICES;
            isSfidStored = !TURN_OFF_SFID;
        }

        stored_qj = new gsl_matrix*[nqj_eff];
    }

    LOG::LOGGER.IO("Number of stored Q matrices: %d\n", nqj_eff);
    if (isSfidStored)   LOG::LOGGER.IO("Fiducial signal matrix is stored.\n");

    isQjSet   = false;
    isSfidSet = false;
}

OneQSOEstimate::OneQSOEstimate(const char *fname_qso)
{
    isCovInverted = false;
    _readFromFile(fname_qso);

    // Covert from wavelength to velocity units around median wavelength
    conv::convertLambdaToVelocity(MEDIAN_REDSHIFT, velocity_array, lambda_array, DATA_SIZE);

    if (conv::USE_FID_LEE12_MEAN_FLUX) // Use fiducial mean flux from Lee12
        conv::convertFluxToDeltafLee12(lambda_array, flux_array, noise_array, DATA_SIZE);
    else    // Convert flux to fluctuation around the mean flux
        conv::convertFluxToDeltaf(flux_array, noise_array, DATA_SIZE);
    
    // Keep noise as error squared (variance)
    for (int i = 0; i < DATA_SIZE; ++i)
        noise_array[i] *= noise_array[i];

    LOG::LOGGER.IO("Length of v is %.1f\n" "Median redshift: %.2f\n" "Redshift range: %.2f--%.2f\n", 
                   velocity_array[DATA_SIZE-1] - velocity_array[0], MEDIAN_REDSHIFT,
                   lambda_array[0]/LYA_REST-1, lambda_array[DATA_SIZE-1]/LYA_REST-1);

    if (!_findRedshiftBin(MEDIAN_REDSHIFT))     return;
    
    // Set up number of matrices, index for Fisher matrix
    setNQandFisherIndex(N_Q_MATRICES, fisher_index_start, ZBIN);

    _setStoredMatrices();
}

OneQSOEstimate::~OneQSOEstimate()
{
    delete [] flux_array;
    delete [] lambda_array;
    delete [] velocity_array;
    delete [] noise_array;
}

void OneQSOEstimate::_getVandZ(double &v_ij, double &z_ij, int i, int j)
{
    v_ij = velocity_array[j] - velocity_array[i];
    z_ij = sqrt(lambda_array[j] * lambda_array[i]) / LYA_REST - 1.;
}

void OneQSOEstimate::_setFiducialSignalMatrix(gsl_matrix *sm)
{
    #pragma omp atomic update
    ++mytime::number_of_times_called_setsfid;

    double t = mytime::getTime();
    double v_ij, z_ij, temp;

    if (isSfidSet)
    {
        gsl_matrix_memcpy(sm, stored_sfid);
    }
    else
    {
        for (int i = 0; i < DATA_SIZE; ++i)
        {
            for (int j = i; j < DATA_SIZE; ++j)
            {
                _getVandZ(v_ij, z_ij, i, j);

                temp = sq_private_table->getSignalMatrixValue(v_ij, z_ij, r_index);
                gsl_matrix_set(sm, i, j, temp);
            }
        }

        mxhelp::copyUpperToLower(sm);
    }
    
    t = mytime::getTime() - t;

    #pragma omp atomic update
    mytime::time_spent_on_set_sfid += t;
}

void OneQSOEstimate::_setQiMatrix(gsl_matrix *qi, int i_kz)
{
    #pragma omp atomic update
    ++mytime::number_of_times_called_setq;

    double t = mytime::getTime(), t_interp;
    int kn, zm;
    double v_ij, z_ij, temp;

    if (isQjSet && i_kz >= N_Q_MATRICES - nqj_eff)
    {
        t_interp = 0;
        
        gsl_matrix_memcpy(qi, stored_qj[N_Q_MATRICES - i_kz - 1]);
    }
    else
    {
        bins::getFisherMatrixBinNoFromIndex(i_kz + fisher_index_start, kn, zm);

        for (int i = 0; i < DATA_SIZE; ++i)
        {
            for (int j = i; j < DATA_SIZE; ++j)
            {
                _getVandZ(v_ij, z_ij, i, j);
                
                temp  = sq_private_table->getDerivativeMatrixValue(v_ij, kn, r_index);
                temp *= bins::redshiftBinningFunction(z_ij, zm, ZBIN);
                #ifdef TOPHAT_Z_BINNING_FN
                // When using top hat redshift binning, we need to evolve pixels
                // to central redshift
                temp *= fidcosmo::fiducialPowerGrowthFactor(z_ij, bins::KBAND_CENTERS[kn], BIN_REDSHIFT, &pd13::FIDUCIAL_PD13_PARAMS);
                #endif
                gsl_matrix_set(qi, i, j, temp);
            }
        }

        t_interp = mytime::getTime() - t;

        mxhelp::copyUpperToLower(qi);
    }

    t = mytime::getTime() - t; 

    #pragma omp atomic update
    mytime::time_spent_set_qs += t;

    #pragma omp atomic update
    mytime::time_spent_on_q_interp += t_interp;

    #pragma omp atomic update
    mytime::time_spent_on_q_copy += t - t_interp;
}

void OneQSOEstimate::setCovarianceMatrix(const double *ps_estimate)
{
    // Set fiducial signal matrix
    if (!TURN_OFF_SFID)
        _setFiducialSignalMatrix(covariance_matrix);
    else
        gsl_matrix_set_zero(covariance_matrix);

    for (int i_kz = 0; i_kz < N_Q_MATRICES; ++i_kz)
    {
        SKIP_LAST_K_BIN_WHEN_ENABLED(i_kz)

        _setQiMatrix(temp_matrix[0], i_kz);

        cblas_daxpy(DATA_SIZE*DATA_SIZE, 
                    ps_estimate[i_kz + fisher_index_start], temp_matrix[0]->data, 1, 
                    covariance_matrix->data, 1);
    }

    // add noise matrix diagonally
    cblas_daxpy(DATA_SIZE, 1., noise_array, 1, covariance_matrix->data, DATA_SIZE+1);

    #define ADDED_CONST_TO_COVARIANCE 10.0
    gsl_matrix_add_constant(covariance_matrix, ADDED_CONST_TO_COVARIANCE);
    #undef ADDED_CONST_TO_COVARIANCE

    isCovInverted = false;
}

// Calculate the inverse into temp_matrix[0]
// Then swap the pointer with covariance matrix
void OneQSOEstimate::invertCovarianceMatrix()
{
    double t = mytime::getTime();

    inverse_covariance_matrix  = temp_matrix[0];

    mxhelp::invertMatrixLU(covariance_matrix, inverse_covariance_matrix);
    
    temp_matrix[0]    = covariance_matrix;
    covariance_matrix = inverse_covariance_matrix;

    isCovInverted = true;

    t = mytime::getTime() - t;

    #pragma omp atomic update
    mytime::time_spent_on_c_inv += t;
}

void OneQSOEstimate::_getWeightedMatrix(gsl_matrix *m)
{
    double t = mytime::getTime();

    //C-1 . Q
    cblas_dsymm( CblasRowMajor, CblasLeft, CblasUpper,
                 DATA_SIZE, DATA_SIZE, 1., inverse_covariance_matrix->data, DATA_SIZE,
                 m->data, DATA_SIZE,
                 0, temp_matrix[1]->data, DATA_SIZE);

    //C-1 . Q . C-1
    cblas_dsymm( CblasRowMajor, CblasRight, CblasUpper,
                 DATA_SIZE, DATA_SIZE, 1., inverse_covariance_matrix->data, DATA_SIZE,
                 temp_matrix[1]->data, DATA_SIZE,
                 0, m->data, DATA_SIZE);

    t = mytime::getTime() - t;

    #pragma omp atomic update
    mytime::time_spent_set_modqs += t;
}

void OneQSOEstimate::_getFisherMatrix(const gsl_matrix *Q_ikz_matrix, int i_kz)
{
    double temp;
    gsl_matrix *Q_jkz_matrix = temp_matrix[1];

    double t = mytime::getTime();
    
    // Now compute Fisher Matrix
    for (int j_kz = i_kz; j_kz < N_Q_MATRICES; ++j_kz)
    {
        _setQiMatrix(Q_jkz_matrix, j_kz);

        temp = 0.5 * mxhelp::trace_dsymm(Q_ikz_matrix, Q_jkz_matrix);
        throw_isnan(temp, "F=TrQwQw");

        gsl_matrix_set(fisher_matrix, i_kz + fisher_index_start, j_kz + fisher_index_start, temp);
        gsl_matrix_set(fisher_matrix, j_kz + fisher_index_start, i_kz + fisher_index_start, temp);
    }

    t = mytime::getTime() - t;

    #pragma omp atomic update
    mytime::time_spent_set_fisher += t;
}

void OneQSOEstimate::computePSbeforeFvector()
{
    gsl_vector  *weighted_data_vector = gsl_vector_alloc(DATA_SIZE);

    gsl_matrix  *Q_ikz_matrix = temp_matrix[0],
                *Sfid_matrix  = temp_matrix[1];

    cblas_dsymv(CblasRowMajor, CblasUpper,
                DATA_SIZE, 1., inverse_covariance_matrix->data, DATA_SIZE,
                flux_array, 1,
                0, weighted_data_vector->data, 1);

    double temp_bk, temp_tk = 0, temp_d;

    for (int i_kz = 0; i_kz < N_Q_MATRICES; ++i_kz)
    {
        // Set derivative matrix ikz
        _setQiMatrix(Q_ikz_matrix, i_kz);

        // Find data contribution to ps before F vector
        // (C-1 . flux)T . Q . (C-1 . flux)
        temp_d = mxhelp::my_cblas_dsymvdot(weighted_data_vector, Q_ikz_matrix);

        throw_isnan(temp_d, "d");

        // Get weighted derivative matrix ikz
        _getWeightedMatrix(Q_ikz_matrix);

        // Get Noise contribution
        temp_bk = mxhelp::trace_ddiagmv(Q_ikz_matrix, noise_array);
        
        throw_isnan(temp_bk, "bk");

        // Set Fiducial Signal Matrix
        if (!TURN_OFF_SFID)
        {
            _setFiducialSignalMatrix(Sfid_matrix);

            temp_tk = mxhelp::trace_dsymm(Q_ikz_matrix, Sfid_matrix);

            throw_isnan(temp_tk, "tk");
        }
                
        gsl_vector_set(ps_before_fisher_estimate_vector, i_kz + fisher_index_start, temp_d - temp_bk - temp_tk);
        _getFisherMatrix(Q_ikz_matrix, i_kz);
    }

    // printf("PS before f: %.3e\n", gsl_vector_get(ps_before_fisher_estimate_vector, 0));
    gsl_vector_free(weighted_data_vector);
}

void OneQSOEstimate::oneQSOiteration(const double *ps_estimate, gsl_vector *pmn_before, gsl_matrix *fisher_sum)
{
    _allocateMatrices();

    // Preload last nqj_eff matrices
    // 0 is the last matrix
    for (int j_kz = 0; j_kz < nqj_eff; ++j_kz)
        _setQiMatrix(stored_qj[j_kz], N_Q_MATRICES - j_kz - 1);
    
    if (nqj_eff > 0)    isQjSet = true;

    // Preload fiducial signal matrix if memory allows
    if (isSfidStored)
    {
        _setFiducialSignalMatrix(stored_sfid);
        isSfidSet = true;
    }

    setCovarianceMatrix(ps_estimate);

    try
    {
        invertCovarianceMatrix();

        computePSbeforeFvector();

        gsl_matrix_add(fisher_sum, fisher_matrix);
        gsl_vector_add(pmn_before, ps_before_fisher_estimate_vector);
    }
    catch (const char* msg)
    {
        LOG::LOGGER.ERR("%d/%d - ERROR %s: Covariance matrix is not invertable. %s\n",
                t_rank, numthreads, msg, qso_sp_fname.c_str());

        LOG::LOGGER.ERR("Npixels: %d, Median z: %.2f, dv: %.2f, R=%d\n",
                DATA_SIZE, MEDIAN_REDSHIFT, DV_KMS, SPECT_RES_FWHM);
    }
    
    _freeMatrices();
}

void OneQSOEstimate::_allocateMatrices()
{
    ps_before_fisher_estimate_vector = gsl_vector_calloc(bins::TOTAL_KZ_BINS);
    fisher_matrix                    = gsl_matrix_calloc(bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);

    covariance_matrix = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);

    for (int i = 0; i < 2; ++i)
        temp_matrix[i] = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
    
    for (int i = 0; i < nqj_eff; ++i)
        stored_qj[i] = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
    
    if (isSfidStored)
        stored_sfid = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
    
    isQjSet   = false;
    isSfidSet = false;
}

void OneQSOEstimate::_freeMatrices()
{
    gsl_vector_free(ps_before_fisher_estimate_vector);
    gsl_matrix_free(fisher_matrix);

    gsl_matrix_free(covariance_matrix);

    for (int i = 0; i < 2; ++i)
        gsl_matrix_free(temp_matrix[i]);
    
    for (int i = 0; i < nqj_eff; ++i)
        gsl_matrix_free(stored_qj[i]);

    if (isSfidStored)
        gsl_matrix_free(stored_sfid);
    
    isQjSet   = false;
    isSfidSet = false;
}
















