#include "one_qso_estimate.hpp"
#include "fiducial_cosmology.hpp"
#include "spectrograph_functions.hpp"
#include "matrix_helper.hpp"
#include "real_field_1d.hpp"
#include "global_numbers.hpp"

#include "../io/io_helper_functions.hpp"
#include "../io/qso_file.hpp"

#include "../gsltools/integrator.hpp"

#include <gsl/gsl_blas.h>
#include <gsl/gsl_errno.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>

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

OneQSOEstimate::OneQSOEstimate(const char *fname_qso, int n, const double *k, const double *zc)
{
    NUMBER_OF_BANDS = n;
    kband_edges     = k;

    // Construct and read data arrays
    QSOFile qFile(fname_qso);

    double dummy_qso_z, dummy_s2n;

    qFile.readParameters(DATA_SIZE, dummy_qso_z, SPECT_RES, dummy_s2n, DV_KMS);
    printf("Data size is %d\n", DATA_SIZE);
    printf("Pixel Width is %.1f\n", DV_KMS);

    lambda_array    = new double[DATA_SIZE];
    velocity_array  = new double[DATA_SIZE];
    flux_array      = new double[DATA_SIZE];
    noise_array     = new double[DATA_SIZE];

    qFile.readData(lambda_array, flux_array, noise_array);

    convert_flux2deltaf(flux_array, DATA_SIZE);

    convert_lambda2v(MEAN_REDSHIFT, velocity_array, lambda_array, DATA_SIZE);
    printf("Length of v is %.1f\n", velocity_array[DATA_SIZE-1] - velocity_array[0]);
    double delta_z  = zc[1] - zc[0];
    ZBIN            = (MEAN_REDSHIFT - (zc[0] - delta_z/2.)) / delta_z;
    BIN_REDSHIFT    = zc[ZBIN];

    printf("Mean redshift of spectrum chunk: %.2f\n", MEAN_REDSHIFT);

    /* Allocate memory */
    derivative_of_signal_matrices          = new gsl_matrix*[NUMBER_OF_BANDS];
    modified_derivative_of_signal_matrices = new gsl_matrix*[NUMBER_OF_BANDS];

    isCovInverted    = false;
}

OneQSOEstimate::~OneQSOEstimate()
{
    delete [] derivative_of_signal_matrices;
    delete [] modified_derivative_of_signal_matrices;

    delete [] flux_array;
    delete [] lambda_array;
    delete [] velocity_array;
    delete [] noise_array;
}

// void OneQSOEstimate::setFiducialSignalAndDerivativeSMatrices(const SQLookupTable *sq_lookup_table)
// {
//     int r_index = sq_lookup_table->findSpecResIndex(SPECT_RES);
//     double v_ij, z_ij;

//     struct spectrograph_windowfn_params win_params = {0, 0, DV_KMS, SPEED_OF_LIGHT / SPECT_RES};

//     Integrator s_integrator(GSL_QAG, signal_matrix_integrand, &win_params);
//     Integrator q_integrator(GSL_QAG, q_matrix_integrand, &win_params);

//     int sq_array_size = (velocity_array[DATA_SIZE-1] - velocity_array[0]) / DV_KMS + 1, temp_index;
//     double *s_ij_array = new double[sq_array_size];
//     double *q_ij_array = new double[sq_array_size];

//     double  temp, \
//             kvalue_1 = kband_edges[0], \
//             kvalue_2 = kband_edges[1], \
//             table_value;

//     // Also set kn=0 for Q
//     for (int i = 0; i < sq_array_size; i++)
//     {
//         win_params.delta_v_ij = i * DV_KMS;
//         win_params.z_ij       = lambda_array[i] / LYA_REST - 1.;

//         s_ij_array[i] = s_integrator.evaluateAToInfty(0);
//         q_ij_array[i] = q_integrator.evaluate(kvalue_1, kvalue_2);
//     }

//     for (int i = 0; i < DATA_SIZE; i++)
//     {
//         v_ij = 0;
//         z_ij = MEAN_REDSHIFT; // lambda_array[i] / LYA_REST - 1.;

//         // True value
//         temp = s_ij_array[0];
//         gsl_matrix_set(fiducial_signal_matrix, i, i, temp);

//         // Look up table
//         table_value = sq_lookup_table->getSignalMatrixValue(v_ij, z_ij, r_index);
//         compareTableTrue(temp, table_value, "signal");

//         // True value
//         temp = q_ij_array[0];
//         gsl_matrix_set(derivative_of_signal_matrices[0], i, i, temp);

//         // Look up table
//         table_value = sq_lookup_table->getDerivativeMatrixValue(v_ij, z_ij, ZBIN, 0, r_index);
//         compareTableTrue(temp, table_value, "derivative");

//         for (int j = i + 1; j < DATA_SIZE; j++)
//         {
//             v_ij = velocity_array[j] - velocity_array[i];
//             z_ij = MEAN_REDSHIFT; // sqrt(lambda_array[j] * lambda_array[i]) / LYA_REST - 1.;

//             temp_index = int(v_ij / DV_KMS + 0.5);

//             temp = s_ij_array[temp_index];
//             gsl_matrix_set(fiducial_signal_matrix, i, j, temp);
//             gsl_matrix_set(fiducial_signal_matrix, j, i, temp);
//             // Look up table
//             table_value = sq_lookup_table->getSignalMatrixValue(v_ij, z_ij, r_index);
//             compareTableTrue(temp, table_value, "signal");

//             temp = q_ij_array[temp_index];
//             gsl_matrix_set(derivative_of_signal_matrices[0], i, j, temp);
//             gsl_matrix_set(derivative_of_signal_matrices[0], j, i, temp);

//             // Look up table
//             table_value = sq_lookup_table->getDerivativeMatrixValue(v_ij, z_ij, ZBIN, 0, r_index);
//             compareTableTrue(temp, table_value, "derivative");
//         }
//     }

//     // Now set the rest of Q
//     for (int kn = 1; kn < NUMBER_OF_BANDS; kn++)
//     {
//         kvalue_1 = kband_edges[kn];
//         kvalue_2 = kband_edges[kn + 1];

//         for (int i = 0; i < sq_array_size; i++)
//         {
//             win_params.delta_v_ij = i * DV_KMS;
//             win_params.z_ij       = lambda_array[i] / LYA_REST - 1.;

//             q_ij_array[i] = q_integrator.evaluate(kvalue_1, kvalue_2);
//         }

//         for (int i = 0; i < DATA_SIZE; i++)
//         {
//             v_ij = 0;
//             z_ij = MEAN_REDSHIFT; //lambda_array[i] / LYA_REST - 1.;

//             temp = q_ij_array[0];
//             gsl_matrix_set(derivative_of_signal_matrices[kn], i, i, temp);

//             // Look up table
//             table_value = sq_lookup_table->getDerivativeMatrixValue(v_ij, z_ij, ZBIN, kn, r_index);
//             compareTableTrue(temp, table_value, "derivative");

//             for (int j = i + 1; j < DATA_SIZE; j++)
//             {
//                 v_ij = velocity_array[j] - velocity_array[i];
//                 z_ij = MEAN_REDSHIFT; // sqrt(lambda_array[j] * lambda_array[i]) / LYA_REST - 1.;

//                 temp_index = int(v_ij / DV_KMS + 0.5);
//                 temp = q_ij_array[temp_index];
                
//                 gsl_matrix_set(derivative_of_signal_matrices[kn], i, j, temp);
//                 gsl_matrix_set(derivative_of_signal_matrices[kn], j, i, temp);

//                 // Look up table
//                 table_value = sq_lookup_table->getDerivativeMatrixValue(v_ij, z_ij, ZBIN, kn, r_index);
//                 compareTableTrue(temp, table_value, "derivative");
//             }
//         }
//     }

//     delete [] q_ij_array;
//     delete [] s_ij_array;
// }

void OneQSOEstimate::setFiducialSignalAndDerivativeSMatrices(const SQLookupTable *sq_lookup_table)
{
    int r_index = sq_lookup_table->findSpecResIndex(SPECT_RES);
    double v_ij, z_ij, temp;

    // Set fiducial signal matrix and first band for Q matrix
    for (int i = 0; i < DATA_SIZE; i++)
    {
        v_ij = 0;
        z_ij = lambda_array[i] / LYA_REST - 1.;

        temp = sq_lookup_table->getSignalMatrixValue(v_ij, z_ij, r_index);
        gsl_matrix_set(fiducial_signal_matrix, i, i, temp);

        temp = sq_lookup_table->getDerivativeMatrixValue(v_ij, z_ij, ZBIN, 0, r_index);
        gsl_matrix_set(derivative_of_signal_matrices[0], i, i, temp);

        for (int j = i + 1; j < DATA_SIZE; j++)
        {
            v_ij = velocity_array[j] - velocity_array[i];
            z_ij = sqrt(lambda_array[j] * lambda_array[i]) / LYA_REST - 1.;

            temp = sq_lookup_table->getSignalMatrixValue(v_ij, z_ij, r_index);
            gsl_matrix_set(fiducial_signal_matrix, i, j, temp);
            gsl_matrix_set(fiducial_signal_matrix, j, i, temp);

            temp = sq_lookup_table->getDerivativeMatrixValue(v_ij, z_ij, ZBIN, 0, r_index);
            gsl_matrix_set(derivative_of_signal_matrices[0], i, j, temp);
            gsl_matrix_set(derivative_of_signal_matrices[0], j, i, temp);
        }
    }

    // Set other bands for Q matrix
    for (int kn = 1; kn < NUMBER_OF_BANDS; kn++)
    {
        for (int i = 0; i < DATA_SIZE; i++)
        {
            v_ij = 0;
            z_ij = lambda_array[i] / LYA_REST - 1.;
            
            temp = sq_lookup_table->getDerivativeMatrixValue(v_ij, z_ij, ZBIN, kn, r_index);
            gsl_matrix_set(derivative_of_signal_matrices[kn], i, i, temp);

            for (int j = i + 1; j < DATA_SIZE; j++)
            {
                v_ij = velocity_array[j] - velocity_array[i];
                z_ij = sqrt(lambda_array[j] * lambda_array[i]) / LYA_REST - 1.;
                
                temp = sq_lookup_table->getDerivativeMatrixValue(v_ij, z_ij, ZBIN, kn, r_index);
                gsl_matrix_set(derivative_of_signal_matrices[kn], i, j, temp);
                gsl_matrix_set(derivative_of_signal_matrices[kn], j, i, temp);
            }
        }
    }
}

void OneQSOEstimate::computeCSMatrices(gsl_vector * const *ps_estimate)
{
    // printf("Theta: %.3e\n", ps_estimate[0]);
    gsl_matrix_memcpy(covariance_matrix, fiducial_signal_matrix);

    gsl_matrix *temp_matrix = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);

    for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
    {
        gsl_matrix_memcpy(temp_matrix, derivative_of_signal_matrices[kn]);
        gsl_matrix_scale(temp_matrix, ps_estimate[ZBIN]->data[kn]);

        gsl_matrix_add(covariance_matrix, temp_matrix);
    }

    // printf_matrix(fiducial_signal_matrix, DATA_SIZE);

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

    isCovInverted    = false;
}

void OneQSOEstimate::invertCovarianceMatrix()
{
    int status = invert_matrix_cholesky(covariance_matrix);

    if (status == GSL_EDOM)
    {
        fprintf(stderr, "ERROR: Covariance matrix is not positive definite.\n");
        throw "COV";
    }

    isCovInverted    = true;
}

void OneQSOEstimate::computeModifiedDSMatrices()
{
    // printf("Setting modified derivative of signal matrices Q-slash_ij(k).\n");
    // fflush(stdout);

    assert(isCovInverted);

    gsl_matrix *temp_matrix = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);

    for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
    {
        //gsl_matrix_memcpy(modified_derivative_of_signal_matrices[kn], derivative_of_signal_matrices[kn]);

        /*
        //C-1 . Q
        gsl_blas_dsymm( CblasLeft, CblasUpper, \
                        1.0, inverse_covariance_matrix, derivative_of_signal_matrices[kn], \
                        0, temp_matrix);

        // C-1 . Q . C-1
        gsl_blas_dsymm( CblasRight, CblasUpper, \
                        1.0, inverse_covariance_matrix, temp_matrix, \
                        0, modified_derivative_of_signal_matrices[kn]);
        */

        //C-1 . Q
        gsl_blas_dgemm( CblasNoTrans, CblasNoTrans, \
                        1.0, inverse_covariance_matrix, derivative_of_signal_matrices[kn], \
                        0, temp_matrix);

        // C-1 . Q . C-1
        gsl_blas_dgemm( CblasNoTrans, CblasNoTrans, \
                        1.0, temp_matrix, inverse_covariance_matrix, \
                        0, modified_derivative_of_signal_matrices[kn]);
    }

    gsl_matrix_free(temp_matrix);   
}

void OneQSOEstimate::computePSbeforeFvector()
{
    // printf("Power estimate before inverse Fisher weighting.\n");
    // fflush(stdout);

    gsl_vector *temp_vector = gsl_vector_alloc(DATA_SIZE);

    gsl_vector_const_view data_view = gsl_vector_const_view_array(flux_array, DATA_SIZE);

    double temp_bk, temp_tk, temp_d;

    for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
    {
        temp_bk = trace_of_2matrices(modified_derivative_of_signal_matrices[kn], noise_array, DATA_SIZE);
        temp_tk = trace_of_2matrices(modified_derivative_of_signal_matrices[kn], fiducial_signal_matrix, DATA_SIZE);

        // printf("Noise b: %.3e\n", temp_bk);
        /*
        gsl_blas_dsymv( CblasUpper, 1.0, \
                        modified_derivative_of_signal_matrices[kn], &data_view.vector, \
                        0, temp_vector);
        */
        gsl_blas_dgemv( CblasNoTrans, 1.0, \
                        modified_derivative_of_signal_matrices[kn], &data_view.vector, \
                        0, temp_vector);

        gsl_blas_ddot(&data_view.vector, temp_vector, &temp_d);

        gsl_vector_set(ps_before_fisher_estimate_vector, kn, temp_d - temp_bk - temp_tk);
    }

    // printf("PS before f: %.3e\n", gsl_vector_get(ps_before_fisher_estimate_vector, 0));
    gsl_vector_free(temp_vector);
}

void OneQSOEstimate::computeFisherMatrix()
{
    // printf("Computing fisher matrix.\n");
    // fflush(stdout);

    double temp;

    for (int k = 0; k < NUMBER_OF_BANDS; k++)
    {
        temp = 0.5 * trace_of_2matrices(modified_derivative_of_signal_matrices[k], \
                                        derivative_of_signal_matrices[k], \
                                        DATA_SIZE);

        gsl_matrix_set(fisher_matrix, k, k, temp);

        for (int q = k + 1; q < NUMBER_OF_BANDS; q++)
        {
            temp = 0.5 * trace_of_2matrices(modified_derivative_of_signal_matrices[k], \
                                            derivative_of_signal_matrices[q], \
                                            DATA_SIZE);

            gsl_matrix_set(fisher_matrix, k, q, temp);
            gsl_matrix_set(fisher_matrix, q, k, temp);
        }
    }

    // printf("Fisher: %.3e\n", gsl_matrix_get(fisher_matrix, 0, 0));
}

void OneQSOEstimate::oneQSOiteration(   gsl_vector * const *ps_estimate, \
                                        const SQLookupTable *sq_lookup_table, \
                                        gsl_vector **pmn_before, gsl_matrix **fisher_sum)
{
    allocateMatrices();

    setFiducialSignalAndDerivativeSMatrices(sq_lookup_table);

    computeCSMatrices(ps_estimate);
    invertCovarianceMatrix();
    computeModifiedDSMatrices();

    computePSbeforeFvector();
    computeFisherMatrix();
    // printf_matrix(fisher_matrix, NUMBER_OF_BANDS);

    gsl_matrix_add(fisher_sum[ZBIN], fisher_matrix);
    gsl_vector_add(pmn_before[ZBIN], ps_before_fisher_estimate_vector);

    freeMatrices();
}

// double OneQSOEstimate::getFiducialPowerSpectrumValue(double k)
// {
//     struct spectrograph_windowfn_params win_params = {0, BIN_REDSHIFT, DV_KMS, SPEED_OF_LIGHT / SPECT_RES};

//     return fiducial_power_spectrum(k, BIN_REDSHIFT, &win_params);
// }

void OneQSOEstimate::getFFTEstimate(double *ps, int *bincount)
{
    double length_v = DV_KMS * DATA_SIZE;

    RealField1D rf(flux_array, DATA_SIZE, length_v);
    
    rf.fftX2K();

    struct spectrograph_windowfn_params win_params = {0, DV_KMS, SPEED_OF_LIGHT / SPECT_RES};
    
    // rf.deconvolve(spectral_response_window_fn, &win_params);

    rf.getPowerSpectrum(ps, kband_edges, NUMBER_OF_BANDS, bincount);

    // deconvolve power spectrum with spectrograph window function.
    for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
    {
        double k = (kband_edges[kn + 1] + kband_edges[kn]) / 2.;
        double w = spectral_response_window_fn(k, &win_params);

        ps[kn] /= w*w;
    }
}

void OneQSOEstimate::allocateMatrices()
{
    ps_before_fisher_estimate_vector = gsl_vector_alloc(NUMBER_OF_BANDS);

    fisher_matrix             = gsl_matrix_alloc(NUMBER_OF_BANDS, NUMBER_OF_BANDS);

    covariance_matrix         = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
    fiducial_signal_matrix    = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
    inverse_covariance_matrix = covariance_matrix;

    for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
    {
        derivative_of_signal_matrices[kn]          = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
        modified_derivative_of_signal_matrices[kn] = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
    }
}

void OneQSOEstimate::freeMatrices()
{
    gsl_vector_free(ps_before_fisher_estimate_vector);

    gsl_matrix_free(fisher_matrix);

    gsl_matrix_free(covariance_matrix);
    gsl_matrix_free(fiducial_signal_matrix);

    for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
    {
        gsl_matrix_free(derivative_of_signal_matrices[kn]);
        gsl_matrix_free(modified_derivative_of_signal_matrices[kn]);
    }
}

// Backup
/*
void OneQSOEstimate::setFiducialSignalAndDerivativeSMatrices(const SQLookupTable *sq_lookup_table)
{
    struct spectrograph_windowfn_params win_params = {0, 0, DV_KMS, SPECT_RES};

    Integrator s_integrator(GSL_QAG, signal_matrix_integrand, &win_params);
    Integrator q_integrator(GSL_QAG, q_matrix_integrand, &win_params);

    int sq_array_size = (velocity_array[DATA_SIZE-1] - velocity_array[0]) / DV_KMS + 1, temp_index;
    double *s_ij_array = new double[sq_array_size];
    double *q_ij_array = new double[sq_array_size];

    double  temp, \
            kvalue_1 = kband_edges[0], \
            kvalue_2 = kband_edges[1];

    // Also set kn=0 for Q
    for (int i = 0; i < sq_array_size; i++)
    {
        win_params.delta_v_ij = i * DV_KMS;
        win_params.z_ij       = lambda_array[i] / LYA_REST - 1.;

        s_ij_array[i] = s_integrator.evaluateAToInfty(0);
        q_ij_array[i] = q_integrator.evaluate(kvalue_1, kvalue_2);
    }

    for (int i = 0; i < DATA_SIZE; i++)
    {
        temp = s_ij_array[0];
        gsl_matrix_set(fiducial_signal_matrix, i, i, temp);

        temp = q_ij_array[0];
        gsl_matrix_set(derivative_of_signal_matrices[0], i, i, temp);

        for (int j = i + 1; j < DATA_SIZE; j++)
        {
            temp_index = int((velocity_array[j] - velocity_array[i]) / DV_KMS + 0.5);

            temp = s_ij_array[temp_index];
            gsl_matrix_set(fiducial_signal_matrix, i, j, temp);
            gsl_matrix_set(fiducial_signal_matrix, j, i, temp);

            temp = q_ij_array[temp_index];
            gsl_matrix_set(derivative_of_signal_matrices[0], i, j, temp);
            gsl_matrix_set(derivative_of_signal_matrices[0], j, i, temp);
        }
    }

    // Now set the rest of Q
    for (int kn = 1; kn < NUMBER_OF_BANDS; kn++)
    {
        kvalue_1 = kband_edges[kn];
        kvalue_2 = kband_edges[kn + 1];

        for (int i = 0; i < sq_array_size; i++)
        {
            win_params.delta_v_ij = i * DV_KMS;
            win_params.z_ij       = lambda_array[i] / LYA_REST - 1.;

            q_ij_array[i] = q_integrator.evaluate(kvalue_1, kvalue_2);
        }

        for (int i = 0; i < DATA_SIZE; i++)
        {
            temp = q_ij_array[0];
            gsl_matrix_set(derivative_of_signal_matrices[kn], i, i, temp);

            for (int j = i + 1; j < DATA_SIZE; j++)
            {
                temp_index = int((velocity_array[j] - velocity_array[i]) / DV_KMS + 0.5);
                temp = q_ij_array[temp_index];
                
                gsl_matrix_set(derivative_of_signal_matrices[kn], i, j, temp);
                gsl_matrix_set(derivative_of_signal_matrices[kn], j, i, temp);
            }
        }
    }

    delete [] q_ij_array;
    delete [] s_ij_array;
}
*/
















