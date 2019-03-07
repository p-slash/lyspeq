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

void compareTableTrue(double true_value, double table_value, const char *which_matrix)
{
    double rel_error = (table_value / true_value) - 1.;

    if (fabs(rel_error) > 0.1)
    {
        printf("True value and table value does not match in %s!\n", which_matrix);
        printf("True value: %.2e\n", true_value);
        printf("Table value: %.2e\n", table_value);
        // throw "Wrong table";
    }
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

    convert_flux2deltaf_mean(flux_array, noise_array, DATA_SIZE);

    convert_lambda2v(MEDIAN_REDSHIFT, velocity_array, lambda_array, DATA_SIZE);
    printf("Length of v is %.1f\n", velocity_array[DATA_SIZE-1] - velocity_array[0]);
    printf("Median redshift of spectrum chunk: %.2f\n", MEDIAN_REDSHIFT);

    ZBIN = (MEDIAN_REDSHIFT - ZBIN_CENTERS[0] + Z_BIN_WIDTH/2.) / Z_BIN_WIDTH;
    
    if (MEDIAN_REDSHIFT < ZBIN_CENTERS[0] - Z_BIN_WIDTH/2.)     ZBIN = -1;

    if (ZBIN >= 0 && ZBIN < NUMBER_OF_Z_BINS)   BIN_REDSHIFT = ZBIN_CENTERS[ZBIN];
    else                                        printf("This QSO does not belong to any redshift bin!\n");
    
#ifdef TOPHAT_Z_BINNING_FN
{
    N_Q_MATRICES       = 1;
    fisher_index_start = getFisherMatrixIndex(0, ZBIN);
}
#endif

#ifdef TRIANGLE_Z_BINNING_FN
{
    N_Q_MATRICES       = 3;
    fisher_index_start = getFisherMatrixIndex(0, ZBIN - 1);

    if (ZBIN == 0) 
    {
        N_Q_MATRICES       = 2;
        fisher_index_start = 0;
    }
    else if (ZBIN == NUMBER_OF_Z_BINS - 1) 
    {
        N_Q_MATRICES     = 2;
    }
}
#endif

    N_Q_MATRICES *= NUMBER_OF_K_BANDS;

    isCovInverted    = false;
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
    float t = get_time();

    double v_ij, z_ij, temp;

    for (int i = 0; i < DATA_SIZE && !TURN_OFF_SFID; i++)
    {
        for (int j = i; j < DATA_SIZE; j++)
        {
            v_ij = velocity_array[j] - velocity_array[i];
            z_ij = sqrt(lambda_array[j] * lambda_array[i]) / LYA_REST - 1.;

            temp = sq_lookup_table->getSignalMatrixValue(v_ij, z_ij, r_index);
            gsl_matrix_set(sm, i, j, temp);
        }
    }

    copy_upper2lower(sm);

    t = get_time() - t;

    #pragma omp atomic update
    time_spent_on_set_sfid += t;
}

void OneQSOEstimate::setQiMatrix(gsl_matrix *qi, int i_kz)
{
    float t = get_time();

    int kn, zm;
    double v_ij, z_ij, temp;

    getFisherMatrixBinNoFromIndex(i_kz + fisher_index_start, kn, zm);

    for (int i = 0; i < DATA_SIZE; i++)
    {
        for (int j = i; j < DATA_SIZE; j++)
        {
            v_ij = velocity_array[j] - velocity_array[i];
            z_ij = sqrt(lambda_array[j] * lambda_array[i]) / LYA_REST - 1.;
            
            temp = sq_lookup_table->getDerivativeMatrixValue(v_ij, z_ij, zm, kn, r_index);
            gsl_matrix_set(qi, i, j, temp);
        }
    }

    copy_upper2lower(qi);

    t = get_time() - t;

    #pragma omp atomic update
    time_spent_set_qs += t;
}

void OneQSOEstimate::setCovarianceMatrix(const gsl_vector *ps_estimate)
{
    r_index = sq_lookup_table->findSpecResIndex(SPECT_RES_FWHM);
    
    if (r_index == -1)      throw "SPECRES not found in tables!";

    // Set fiducial signal matrix

    if (!TURN_OFF_SFID)
        setFiducialSignalMatrix(covariance_matrix);
    else
        gsl_matrix_set_zero(covariance_matrix);

    for (int i_kz = 0; i_kz < N_Q_MATRICES; i_kz++)
    {
        setQiMatrix(temp_matrix[0], i_kz);
        gsl_matrix_scale(temp_matrix[0], gsl_vector_get(ps_estimate, i_kz + fisher_index_start));
        gsl_matrix_add(covariance_matrix, temp_matrix[0]);
    }

    double temp, nn;

    for (int i = 0; i < DATA_SIZE; i++)
    {
        nn = noise_array[i];
        nn *= nn;

        temp = gsl_matrix_get(covariance_matrix, i, i);

        gsl_matrix_set(covariance_matrix, i, i, temp + nn);
    }

    // printf_matrix(covariance_matrix, DATA_SIZE);
    gsl_matrix_add_constant(covariance_matrix, ADDED_CONST_TO_COVARIANCE);

    isCovInverted = false;
}

void OneQSOEstimate::invertCovarianceMatrix()
{
    float t = get_time();
    invert_matrix_cholesky_2(covariance_matrix);

    isCovInverted = true;

    t = get_time() - t;

    #pragma omp atomic update
    time_spent_on_c_inv += t;
}

void OneQSOEstimate::getWeightedMatrix(gsl_matrix *m)
{
    float t = get_time();

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

void OneQSOEstimate::getFisherMatrix(gsl_matrix *Q_ikz_matrix, int i_kz)
{
    double temp;
    gsl_matrix *Q_jkz_matrix = temp_matrix[1];

    float t = get_time();
    
    // Now compute Fisher Matrix
    for (int j_kz = i_kz; j_kz < N_Q_MATRICES; j_kz++)
    {
        setQiMatrix(Q_jkz_matrix, j_kz);

        temp = 0.5 * trace_of_2_sym_matrices(Q_ikz_matrix, Q_jkz_matrix);
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
    gsl_vector  *temp_vector            = gsl_vector_alloc(DATA_SIZE), \
                *weighted_data_vector   = gsl_vector_alloc(DATA_SIZE);

    gsl_matrix  *Q_ikz_matrix = temp_matrix[0], \
                *Sfid_matrix  = temp_matrix[1];

    /*
    cblas_dsymv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, 
                const int N, const double alpha, const double * A, const int lda,
                const double * x, const int incx, const double beta, double * y, const int incy)
    */

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
        // Q . (C-1 . flux)
        cblas_dsymv(CblasRowMajor, CblasUpper, \
                    DATA_SIZE, 1., Q_ikz_matrix->data, DATA_SIZE, \
                    weighted_data_vector->data, 1, \
                    0, temp_vector->data, 1);
        
        // (C-1 . flux)T . Q . (C-1 . flux)
        temp_d = cblas_ddot(DATA_SIZE, \
                            weighted_data_vector->data, 1, \
                            temp_vector->data, 1);

        // Get weighted derivative matrix ikz
        getWeightedMatrix(Q_ikz_matrix);

        // Get Noise contribution
        temp_bk = trace_of_2matrices(Q_ikz_matrix, noise_array);
        
        throw_isnan(temp_bk, "bk");

        // Set Fiducial Signal Matrix
        if (!TURN_OFF_SFID)
        {
            setFiducialSignalMatrix(Sfid_matrix);

            temp_tk = trace_of_2_sym_matrices(Q_ikz_matrix, Sfid_matrix);

            throw_isnan(temp_tk, "tk");
        }
        
        throw_isnan(temp_d - temp_bk - temp_tk, "d");
        
        gsl_vector_set(ps_before_fisher_estimate_vector, i_kz + fisher_index_start, temp_d - temp_bk - temp_tk);
        
        // Now compute Fisher Matrix
        getFisherMatrix(Q_ikz_matrix, i_kz);
    }

    // printf("PS before f: %.3e\n", gsl_vector_get(ps_before_fisher_estimate_vector, 0));
    gsl_vector_free(temp_vector);
}

void OneQSOEstimate::oneQSOiteration(   const gsl_vector *ps_estimate, \
                                        gsl_vector *pmn_before, gsl_matrix *fisher_sum)
{
    allocateMatrices();
    #ifdef DEBUG_ON
    printf("Allocated\n");
    fflush(stdout);
    #endif

    setCovarianceMatrix(ps_estimate);
    #ifdef DEBUG_ON
    printf("Set\n");
    fprintf_matrix("debugdump_covariance_matrix.dat", covariance_matrix);
    fflush(stdout);
    #endif

    try
    {
        invertCovarianceMatrix();

        computePSbeforeFvector();

        gsl_matrix_add(fisher_sum, fisher_matrix);
        gsl_vector_add(pmn_before, ps_before_fisher_estimate_vector);
    }
    catch (const char* msg)
    {
        fprintf(stderr, "ERROR %s: Covariance matrix is not invertable. %s\n", msg, qso_sp_fname);
        fprintf(stderr, "Npixels: %d, Median z: %.2f, dv: %.2f, R=%d\n", DATA_SIZE, MEDIAN_REDSHIFT, DV_KMS, SPECT_RES_FWHM);
        
        // for (int i = 0; i < DATA_SIZE; i++)     fprintf(stderr, "%.2lf ", flux_array[i]);

        // fprintf(stderr, "\nNoise: ");

        // for (int i = 0; i < DATA_SIZE; i++)     fprintf(stderr, "%.2lf ", noise_array[i]);
        
        // fprintf(stderr, "\n");
    }
    
    freeMatrices();
}

void OneQSOEstimate::allocateMatrices()
{
    ps_before_fisher_estimate_vector = gsl_vector_calloc(TOTAL_KZ_BINS);
    fisher_matrix                    = gsl_matrix_calloc(TOTAL_KZ_BINS, TOTAL_KZ_BINS);

    covariance_matrix = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
    inverse_covariance_matrix = covariance_matrix;

    for (int i = 0; i < 2; i++)
        temp_matrix[i] = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
}

void OneQSOEstimate::freeMatrices()
{
    gsl_vector_free(ps_before_fisher_estimate_vector);
    gsl_matrix_free(fisher_matrix);

    gsl_matrix_free(covariance_matrix);
    // gsl_matrix_free(inverse_covariance_matrix);

    for (int i = 0; i < 2; i++)
        gsl_matrix_free(temp_matrix[i]); 
}
















