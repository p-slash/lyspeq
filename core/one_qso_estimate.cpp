#include "one_qso_estimate.hpp"
#include "fiducial_cosmology.hpp"
#include "spectrograph_functions.hpp"
#include "matrix_helper.hpp"
#include "global_numbers.hpp"

#include "../io/io_helper_functions.hpp"
#include "../io/qso_file.hpp"

#include "../gsltools/integrator.hpp"

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

    if (isnan(t))   throw err_msg;
}

int getFisherMatrixIndex(int kn, int zm)
{
    return kn + NUMBER_OF_K_BANDS * zm;
}

void getFisherMatrixBinNoFromIndex(int i, int &kn, int &zm)
{
    kn = i % NUMBER_OF_K_BANDS;
    zm = i / NUMBER_OF_K_BANDS;
}

// For a top hat redshift bin, only access 1 redshift bin
// For triangular z bins, access 3 (2 for first and last z bins) redshift bins
void setNQandFisherIndex(int &nq, int &fi, int ZBIN)
{
    #ifdef TOPHAT_Z_BINNING_FN
    nq = 1;
    fi = getFisherMatrixIndex(0, ZBIN);
    #endif

    #ifdef TRIANGLE_Z_BINNING_FN
    nq = 3;
    fi = getFisherMatrixIndex(0, ZBIN - 1);

    if (ZBIN == 0) 
    {
        nq = 2;
        fi = 0;
    }
    else if (ZBIN == NUMBER_OF_Z_BINS - 1) 
    {
        nq = 2;
    }
    #endif

    nq *= NUMBER_OF_K_BANDS;
}

OneQSOEstimate::OneQSOEstimate(const char *fname_qso)
{
    sprintf(qso_sp_fname, "%s", fname_qso);

    // Construct and read data arrays
    QSOFile qFile(qso_sp_fname);

    double dummy_qso_z, dummy_s2n;

    qFile.readParameters(DATA_SIZE, dummy_qso_z, SPECT_RES_FWHM, dummy_s2n, DV_KMS);
    printf("Data size is %d\n", DATA_SIZE);
    printf("Pixel Width is %.1f\n", DV_KMS);

    lambda_array    = new double[DATA_SIZE];
    velocity_array  = new double[DATA_SIZE];
    flux_array      = new double[DATA_SIZE];
    noise_array     = new double[DATA_SIZE];

    qFile.readData(lambda_array, flux_array, noise_array);

    // Convert flux to fluctuation around the mean flux
    convert_flux2deltaf_mean(flux_array, noise_array, DATA_SIZE);
    
    // Keep noise as error squared (variance)
    for (int i = 0; i < DATA_SIZE; ++i)
        noise_array[i] *= noise_array[i];

    // Covert from wavelength to velocity units around median wavelength
    convert_lambda2v(MEDIAN_REDSHIFT, velocity_array, lambda_array, DATA_SIZE);

    printf("Length of v is %.1f\n", velocity_array[DATA_SIZE-1] - velocity_array[0]);
    printf("Median redshift of spectrum chunk: %.2f\n", MEDIAN_REDSHIFT);

    // Assign to a redshift bin according to median redshift of this chunk
    ZBIN = (MEDIAN_REDSHIFT - ZBIN_CENTERS[0] + Z_BIN_WIDTH/2.) / Z_BIN_WIDTH;
    
    if (MEDIAN_REDSHIFT < ZBIN_CENTERS[0] - Z_BIN_WIDTH/2.)     ZBIN = -1;

    if (ZBIN >= 0 && ZBIN < NUMBER_OF_Z_BINS)   BIN_REDSHIFT = ZBIN_CENTERS[ZBIN];
    else                                        printf("This QSO does not belong to any redshift bin!\n");
    
    // Find the resolution index for the look up table
    r_index = sq_private_table->findSpecResIndex(SPECT_RES_FWHM);
    
    if (r_index == -1)      throw "SPECRES not found in tables!";

    // Set up number of matrices, index for Fisher matrix
    setNQandFisherIndex(N_Q_MATRICES, fisher_index_start, ZBIN);

    isCovInverted    = false;

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

    printf("Number of stored Q matrices: %d\n", nqj_eff);
    if (isSfidStored)   printf("Fiducial signal matrix is stored.\n");
    fflush(stdout);

    isQjSet   = false;
    isSfidSet = false;
}

OneQSOEstimate::~OneQSOEstimate()
{
    delete [] flux_array;
    delete [] lambda_array;
    delete [] velocity_array;
    delete [] noise_array;
}

void OneQSOEstimate::setFiducialSignalMatrix(gsl_matrix *sm)
{
    #pragma omp atomic update
    number_of_times_called_setsfid++;

    double t = get_time();
    double v_ij, z_ij, temp;

    if (isSfidSet)
    {
        gsl_matrix_memcpy(sm, stored_sfid);
    }
    else
    {
        for (int i = 0; i < DATA_SIZE && !TURN_OFF_SFID; i++)
        {
            for (int j = i; j < DATA_SIZE; j++)
            {
                v_ij = velocity_array[j] - velocity_array[i];
                z_ij = sqrt(lambda_array[j] * lambda_array[i]) / LYA_REST - 1.;

                temp = sq_private_table->getSignalMatrixValue(v_ij, z_ij, r_index);
                gsl_matrix_set(sm, i, j, temp);
            }
        }

        copy_upper2lower(sm);
    }
    
    t = get_time() - t;

    #pragma omp atomic update
    time_spent_on_set_sfid += t;
}

void OneQSOEstimate::setQiMatrix(gsl_matrix *qi, int i_kz)
{
    #pragma omp atomic update
    number_of_times_called_setq++;

    double t = get_time(), t_interp;
    int kn, zm;
    double v_ij, z_ij, temp;

    if (isQjSet && i_kz >= N_Q_MATRICES - nqj_eff)
    {
        t_interp = 0;
        
        gsl_matrix_memcpy(qi, stored_qj[N_Q_MATRICES - i_kz - 1]);
    }
    else
    {
        getFisherMatrixBinNoFromIndex(i_kz + fisher_index_start, kn, zm);

        for (int i = 0; i < DATA_SIZE; i++)
        {
            for (int j = i; j < DATA_SIZE; j++)
            {
                v_ij = velocity_array[j] - velocity_array[i];
                z_ij = sqrt(lambda_array[j] * lambda_array[i]) / LYA_REST - 1.;
                
                temp = sq_private_table->getDerivativeMatrixValue(v_ij, z_ij, zm, kn, r_index);
                gsl_matrix_set(qi, i, j, temp);
            }
        }

        t_interp = get_time() - t;

        copy_upper2lower(qi);
    }

    t = get_time() - t; 

    #pragma omp atomic update
    time_spent_set_qs += t;

    #pragma omp atomic update
    time_spent_on_q_interp += t_interp;

    #pragma omp atomic update
    time_spent_on_q_copy += t - t_interp;
}

void OneQSOEstimate::setCovarianceMatrix(const double *ps_estimate)
{
    // Set fiducial signal matrix
    if (!TURN_OFF_SFID)
        setFiducialSignalMatrix(covariance_matrix);
    else
        gsl_matrix_set_zero(covariance_matrix);

    for (int i_kz = 0; i_kz < N_Q_MATRICES; i_kz++)
    {
        // Skip if last k bin
        if (is_last_bin(i_kz))   continue;

        setQiMatrix(temp_matrix[0], i_kz);

        cblas_daxpy(DATA_SIZE*DATA_SIZE, \
                    ps_estimate[i_kz + fisher_index_start], temp_matrix[0]->data, 1, \
                    covariance_matrix->data, 1);
    }

    // add noise matrix diagonally
    cblas_daxpy(DATA_SIZE, 1., noise_array, 1, covariance_matrix->data, DATA_SIZE+1);

    // printf_matrix(covariance_matrix, DATA_SIZE);
    gsl_matrix_add_constant(covariance_matrix, ADDED_CONST_TO_COVARIANCE);

    isCovInverted = false;
}

// Calculate the inverse into temp_matrix[0]
// Then swap the pointer with covariance matrix
void OneQSOEstimate::invertCovarianceMatrix()
{
    double t = get_time();

    inverse_covariance_matrix  = temp_matrix[0];

    invert_matrix_LU(covariance_matrix, inverse_covariance_matrix);
    
    temp_matrix[0]    = covariance_matrix;
    covariance_matrix = inverse_covariance_matrix;

    isCovInverted = true;

    t = get_time() - t;

    #pragma omp atomic update
    time_spent_on_c_inv += t;
}

void OneQSOEstimate::getWeightedMatrix(gsl_matrix *m)
{
    double t = get_time();

    //C-1 . Q
    cblas_dsymm( CblasRowMajor, CblasLeft, CblasUpper, \
                 DATA_SIZE, DATA_SIZE, 1., inverse_covariance_matrix->data, DATA_SIZE, \
                 m->data, DATA_SIZE, \
                 0, temp_matrix[1]->data, DATA_SIZE);

    //C-1 . Q . C-1
    cblas_dsymm( CblasRowMajor, CblasRight, CblasUpper, \
                 DATA_SIZE, DATA_SIZE, 1., inverse_covariance_matrix->data, DATA_SIZE, \
                 temp_matrix[1]->data, DATA_SIZE, \
                 0, m->data, DATA_SIZE);

    t = get_time() - t;

    #pragma omp atomic update
    time_spent_set_modqs += t;
}

void OneQSOEstimate::getFisherMatrix(const gsl_matrix *Q_ikz_matrix, int i_kz)
{
    double temp;
    gsl_matrix *Q_jkz_matrix = temp_matrix[1];

    double t = get_time();
    
    // Now compute Fisher Matrix
    for (int j_kz = i_kz; j_kz < N_Q_MATRICES; j_kz++)
    {
        setQiMatrix(Q_jkz_matrix, j_kz);

        temp = 0.5 * trace_dsymm(Q_ikz_matrix, Q_jkz_matrix);
        throw_isnan(temp, "F=TrQwQw");

        gsl_matrix_set( fisher_matrix, \
                        i_kz + fisher_index_start, \
                        j_kz + fisher_index_start, \
                        temp);
        gsl_matrix_set( fisher_matrix, \
                        j_kz + fisher_index_start, \
                        i_kz + fisher_index_start, \
                        temp);
    }

    t = get_time() - t;

    #pragma omp atomic update
    time_spent_set_fisher += t;
}

void OneQSOEstimate::computePSbeforeFvector()
{
    gsl_vector  *weighted_data_vector = gsl_vector_alloc(DATA_SIZE);

    gsl_matrix  *Q_ikz_matrix = temp_matrix[0], \
                *Sfid_matrix  = temp_matrix[1];

    cblas_dsymv(CblasRowMajor, CblasUpper, \
                DATA_SIZE, 1., inverse_covariance_matrix->data, DATA_SIZE, \
                flux_array, 1, \
                0, weighted_data_vector->data, 1);

    double temp_bk, temp_tk = 0, temp_d;

    for (int i_kz = 0; i_kz < N_Q_MATRICES; i_kz++)
    {
        // Set derivative matrix ikz
        setQiMatrix(Q_ikz_matrix, i_kz);

        // Find data contribution to ps before F vector
        // (C-1 . flux)T . Q . (C-1 . flux)
        temp_d = my_cblas_dsymvdot(weighted_data_vector, Q_ikz_matrix);

        throw_isnan(temp_d, "d");

        // Get weighted derivative matrix ikz
        getWeightedMatrix(Q_ikz_matrix);

        // Get Noise contribution
        temp_bk = trace_ddiagmv(Q_ikz_matrix, noise_array);
        
        throw_isnan(temp_bk, "bk");

        // Set Fiducial Signal Matrix
        if (!TURN_OFF_SFID)
        {
            setFiducialSignalMatrix(Sfid_matrix);

            temp_tk = trace_dsymm(Q_ikz_matrix, Sfid_matrix);

            throw_isnan(temp_tk, "tk");
        }
                
        gsl_vector_set(ps_before_fisher_estimate_vector, i_kz + fisher_index_start, temp_d - temp_bk - temp_tk);
        getFisherMatrix(Q_ikz_matrix, i_kz);
    }

    // printf("PS before f: %.3e\n", gsl_vector_get(ps_before_fisher_estimate_vector, 0));
    gsl_vector_free(weighted_data_vector);
}

void OneQSOEstimate::oneQSOiteration(   const double *ps_estimate, \
                                        gsl_vector *pmn_before, gsl_matrix *fisher_sum)
{
    allocateMatrices();

    // Preload last nqj_eff matrices
    // 0 is the last matrix
    for (int j_kz = 0; j_kz < nqj_eff; j_kz++)
        setQiMatrix(stored_qj[j_kz], N_Q_MATRICES - j_kz - 1);
    
    if (nqj_eff > 0)    isQjSet = true;

    // Preload fiducial signal matrix if memory allows
    if (isSfidStored)
    {
        setFiducialSignalMatrix(stored_sfid);
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
        fprintf(stderr, "%d/%d - ERROR %s: Covariance matrix is not invertable. %s\n", \
                t_rank, numthreads, msg, qso_sp_fname);
        fprintf(stderr, "Npixels: %d, Median z: %.2f, dv: %.2f, R=%d\n", \
                DATA_SIZE, MEDIAN_REDSHIFT, DV_KMS, SPECT_RES_FWHM);
    }
    
    freeMatrices();
}

void OneQSOEstimate::allocateMatrices()
{
    ps_before_fisher_estimate_vector = gsl_vector_calloc(TOTAL_KZ_BINS);
    fisher_matrix                    = gsl_matrix_calloc(TOTAL_KZ_BINS, TOTAL_KZ_BINS);

    covariance_matrix = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);

    for (int i = 0; i < 2; i++)
        temp_matrix[i] = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
    
    for (int i = 0; i < nqj_eff; i++)
        stored_qj[i] = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
    
    if (isSfidStored)
        stored_sfid = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
    
    isQjSet   = false;
    isSfidSet = false;
}

void OneQSOEstimate::freeMatrices()
{
    gsl_vector_free(ps_before_fisher_estimate_vector);
    gsl_matrix_free(fisher_matrix);

    gsl_matrix_free(covariance_matrix);

    for (int i = 0; i < 2; i++)
        gsl_matrix_free(temp_matrix[i]);
    
    for (int i = 0; i < nqj_eff; i++)
        gsl_matrix_free(stored_qj[i]);

    if (isSfidStored)
        gsl_matrix_free(stored_sfid);
    
    isQjSet   = false;
    isSfidSet = false;
}
















