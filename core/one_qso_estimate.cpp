#include "one_qso_estimate.hpp"
#include "fiducial_cosmology.hpp"
#include "spectrograph_functions.hpp"
#include "matrix_helper.hpp"
#include "global_numbers.hpp"

#include "../io/io_helper_functions.hpp"
#include "../io/qso_file.hpp"

#include "../gsltools/integrator.hpp"

#include <gsl/gsl_blas.h>
#include <gsl/gsl_errno.h>

#include <cmath>
#include <ctime>    /* clock_t, clock, CLOCKS_PER_SEC */
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

    /* Allocate memory */
    derivative_of_signal_matrices          = new gsl_matrix*[N_Q_MATRICES];
    weighted_derivative_of_signal_matrices = new gsl_matrix*[N_Q_MATRICES];

    isCovInverted    = false;
}

OneQSOEstimate::~OneQSOEstimate()
{
    delete [] derivative_of_signal_matrices;
    delete [] weighted_derivative_of_signal_matrices;

    delete [] flux_array;
    delete [] lambda_array;
    delete [] velocity_array;
    delete [] noise_array;
}

void OneQSOEstimate::setFiducialSignalAndDerivativeSMatrices(const SQLookupTable *sq_lookup_table)
{
    int r_index = sq_lookup_table->findSpecResIndex(SPECT_RES_FWHM);
    
    if (r_index == -1)      throw "SPECRES not found in tables!";

    double v_ij, z_ij, temp;

    clock_t t;
    // Set fiducial signal matrix
    t = clock();

    for (int i = 0; i < DATA_SIZE && !TURN_OFF_SFID; i++)
    {
        for (int j = i; j < DATA_SIZE; j++)
        {
            v_ij = velocity_array[j] - velocity_array[i];
            z_ij = sqrt(lambda_array[j] * lambda_array[i]) / LYA_REST - 1.;

            temp = sq_lookup_table->getSignalMatrixValue(v_ij, z_ij, r_index);
            gsl_matrix_set(fiducial_signal_matrix, i, j, temp);
        }
    }

    copy_upper2lower(fiducial_signal_matrix);

    t = clock() - t;

    #pragma omp atomic update
    time_spent_on_set_sfid += ((float) t) / CLOCKS_PER_SEC;

    // Set Q matrices
    t = clock();
    int kn, zm;

    for (int i_kz = 0; i_kz < N_Q_MATRICES; i_kz++)
    {
        getFisherMatrixBinNoFromIndex(i_kz + fisher_index_start, kn, zm);

        for (int i = 0; i < DATA_SIZE; i++)
        {
            for (int j = i; j < DATA_SIZE; j++)
            {
                v_ij = velocity_array[j] - velocity_array[i];
                z_ij = sqrt(lambda_array[j] * lambda_array[i]) / LYA_REST - 1.;
                
                temp = sq_lookup_table->getDerivativeMatrixValue(v_ij, z_ij, zm, kn, r_index);
                gsl_matrix_set(derivative_of_signal_matrices[i_kz], i, j, temp);
            }
        }

        copy_upper2lower(derivative_of_signal_matrices[i_kz]);
    }

    t = clock() - t;

    #pragma omp atomic update
    time_spent_set_qs += ((float) t) / CLOCKS_PER_SEC;
}

void OneQSOEstimate::computeCSMatrices(const gsl_vector *ps_estimate)
{
    if (!TURN_OFF_SFID)
        gsl_matrix_memcpy(covariance_matrix, fiducial_signal_matrix);
    else
        gsl_matrix_set_zero(covariance_matrix);

    gsl_matrix *temp_matrix = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);

    for (int i_kz = 0; i_kz < N_Q_MATRICES; i_kz++)
    {
        gsl_matrix_memcpy(temp_matrix, derivative_of_signal_matrices[i_kz]);
        gsl_matrix_scale(temp_matrix, gsl_vector_get(ps_estimate, i_kz + fisher_index_start));

        gsl_matrix_add(covariance_matrix, temp_matrix);
    }

    gsl_matrix_free(temp_matrix);

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
    clock_t t = clock();
    invert_matrix_LU(covariance_matrix, inverse_covariance_matrix);

    isCovInverted = true;

    t = clock() - t;

    #pragma omp atomic update
    time_spent_on_c_inv += ((float) t) / CLOCKS_PER_SEC;
}

void OneQSOEstimate::computeWeightedMatrices()
{
    clock_t t = clock();

    assert(isCovInverted);

    for (int i_kz = 0; i_kz < N_Q_MATRICES; i_kz++)
    {
        //C-1 . Q
        gsl_blas_dgemm( CblasNoTrans, CblasNoTrans, \
                        1.0, inverse_covariance_matrix, derivative_of_signal_matrices[i_kz], \
                        0, weighted_derivative_of_signal_matrices[i_kz]);
    }

    gsl_matrix *temp_matrix = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);

    // Set weighted fiducial signal matrix
    if (!TURN_OFF_SFID)
    {
        gsl_blas_dgemm( CblasNoTrans, CblasNoTrans, \
                        1.0, inverse_covariance_matrix, fiducial_signal_matrix, \
                        0, temp_matrix);

        gsl_matrix_memcpy(weighted_fiducial_signal_matrix, temp_matrix);
    }
    
    // Set weighted noise matrix
    for (int i = 0; i < DATA_SIZE; i++)
    {
        double n = noise_array[i];
        n *= n;

        gsl_matrix_set(weighted_noise_matrix, i, i, n);
    }

    gsl_blas_dgemm( CblasNoTrans, CblasNoTrans, \
                    1.0, inverse_covariance_matrix, weighted_noise_matrix, \
                    0, temp_matrix);

    gsl_matrix_memcpy(weighted_noise_matrix, temp_matrix);

    gsl_matrix_free(temp_matrix);

    t = clock() - t;

    #pragma omp atomic update
    time_spent_set_modqs += ((float) t) / CLOCKS_PER_SEC;
}

void OneQSOEstimate::computePSbeforeFvector()
{
    gsl_vector  *temp_vector            = gsl_vector_alloc(DATA_SIZE), \
                *weighted_data_vector   = gsl_vector_alloc(DATA_SIZE);

    gsl_vector_const_view data_view = gsl_vector_const_view_array(flux_array, DATA_SIZE);

    gsl_blas_dgemv( CblasNoTrans, 1.0, \
                    inverse_covariance_matrix, &data_view.vector, \
                    0, weighted_data_vector);

    double temp_bk, temp_tk = 0, temp_d;

    for (int i_kz = 0; i_kz < N_Q_MATRICES; i_kz++)
    {
        temp_bk = trace_of_2matrices(weighted_derivative_of_signal_matrices[i_kz], weighted_noise_matrix);
        
        throw_isnan(temp_bk, "bk");

        if (!TURN_OFF_SFID)
            temp_tk = trace_of_2matrices(weighted_derivative_of_signal_matrices[i_kz], weighted_fiducial_signal_matrix);

        throw_isnan(temp_tk, "tk");

        gsl_blas_dgemv( CblasNoTrans, 1.0, \
                        derivative_of_signal_matrices[i_kz], weighted_data_vector, \
                        0, temp_vector);
        
        gsl_blas_ddot(weighted_data_vector, temp_vector, &temp_d);

        throw_isnan(temp_d - temp_bk - temp_tk, "d");
        
        gsl_vector_set(ps_before_fisher_estimate_vector, i_kz + fisher_index_start, temp_d - temp_bk - temp_tk);
    }

    // printf("PS before f: %.3e\n", gsl_vector_get(ps_before_fisher_estimate_vector, 0));
    gsl_vector_free(temp_vector);
}

void OneQSOEstimate::computeFisherMatrix()
{
    clock_t t = clock();

    #ifdef DEBUG_ON
    printf("Computing fisher matrix.\n");
    fflush(stdout);
    #endif
    
    double temp;

    for (int i_kz = 0; i_kz < N_Q_MATRICES; i_kz++)
    {
        for (int j_kz = i_kz; j_kz < N_Q_MATRICES; j_kz++)
        {
            temp = 0.5 * trace_of_2matrices(weighted_derivative_of_signal_matrices[i_kz], \
                                            weighted_derivative_of_signal_matrices[j_kz]);

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
    }
    
    t = clock() - t;

    #pragma omp atomic update
    time_spent_set_fisher += ((float) t) / CLOCKS_PER_SEC;
}

void OneQSOEstimate::oneQSOiteration(   const gsl_vector *ps_estimate, \
                                        const SQLookupTable *sq_lookup_table, \
                                        gsl_vector *pmn_before, gsl_matrix *fisher_sum)
{
    allocateMatrices();
    #ifdef DEBUG_ON
    printf("Allocated\n");
    fflush(stdout);
    #endif

    setFiducialSignalAndDerivativeSMatrices(sq_lookup_table);
    #ifdef DEBUG_ON
    printf("Set\n");
    fprintf_matrix("debugdump_fiducial_signal_matrix.dat", fiducial_signal_matrix);
    fflush(stdout);
    #endif

    computeCSMatrices(ps_estimate);
    #ifdef DEBUG_ON
    printf("CovSig\n");
    fprintf_matrix("debugdump_covariance_matrix.dat", covariance_matrix);
    fflush(stdout);
    #endif

    try
    {
        invertCovarianceMatrix();
        computeWeightedMatrices();

        computePSbeforeFvector();
        computeFisherMatrix();
        
        #ifdef DEBUG_ON
        dump_all_matrices();
        #endif

        gsl_matrix_add(fisher_sum, fisher_matrix);
        gsl_vector_add(pmn_before, ps_before_fisher_estimate_vector);
    }
    catch (const char* msg)
    {
        fprintf(stderr, "ERROR %s: Covariance matrix is not invertable. %s\n", msg, qso_sp_fname);
        fprintf(stderr, "Npixels: %d\nMedian z: %.2f\nFlux: ", DATA_SIZE, MEDIAN_REDSHIFT);
        
        for (int i = 0; i < DATA_SIZE; i++)     fprintf(stderr, "%.2lf ", flux_array[i]);

        fprintf(stderr, "\nNoise: ");

        for (int i = 0; i < DATA_SIZE; i++)     fprintf(stderr, "%.2lf ", noise_array[i]);
        
        fprintf(stderr, "\n");
    }
    
    freeMatrices();
}

void OneQSOEstimate::allocateMatrices()
{
    ps_before_fisher_estimate_vector = gsl_vector_calloc(TOTAL_KZ_BINS);
    fisher_matrix                    = gsl_matrix_calloc(TOTAL_KZ_BINS, TOTAL_KZ_BINS);

    covariance_matrix               = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
    inverse_covariance_matrix       = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
    fiducial_signal_matrix          = gsl_matrix_calloc(DATA_SIZE, DATA_SIZE);
    weighted_fiducial_signal_matrix = fiducial_signal_matrix;
    weighted_noise_matrix           = gsl_matrix_calloc(DATA_SIZE, DATA_SIZE);

    for (int i = 0; i < N_Q_MATRICES; i++)
    {
        derivative_of_signal_matrices[i]          = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
        weighted_derivative_of_signal_matrices[i] = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
    }
}

void OneQSOEstimate::freeMatrices()
{
    gsl_vector_free(ps_before_fisher_estimate_vector);

    gsl_matrix_free(fisher_matrix);

    gsl_matrix_free(covariance_matrix);
    gsl_matrix_free(inverse_covariance_matrix);
    gsl_matrix_free(fiducial_signal_matrix);
    gsl_matrix_free(weighted_noise_matrix);

    for (int i = 0; i < N_Q_MATRICES; i++)
    {
        gsl_matrix_free(derivative_of_signal_matrices[i]);
        gsl_matrix_free(weighted_derivative_of_signal_matrices[i]);
    }
}

void OneQSOEstimate::dump_all_matrices()
{
    char buf[250];

    for (int i_kz = 0; i_kz < N_Q_MATRICES; i_kz++)
    {
        sprintf(buf, "debugdump_Q%d_matrix.dat", i_kz);
        fprintf_matrix(buf, derivative_of_signal_matrices[i_kz]);

        sprintf(buf, "debugdump_WeightedQ%d_matrix.dat", i_kz);
        fprintf_matrix(buf, weighted_derivative_of_signal_matrices[i_kz]);
    }
    
    fprintf_matrix("debugdump_covariance_matrix.dat", covariance_matrix); 
    fprintf_matrix("debugdump_inversecovariance_matrix.dat", inverse_covariance_matrix);
    fprintf_matrix("debugdump_fisher_matrix.dat", fisher_matrix);
}
















