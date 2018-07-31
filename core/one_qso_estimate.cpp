#include "one_qso_estimate.hpp"
#include "fiducial_cosmology.hpp"
#include "spectrograph_functions.hpp"
#include "matrix_helper.hpp"
#include "real_field_1d.hpp"

#include "../io/io_helper_functions.hpp"
#include "../io/qso_file.hpp"

#include "../gsltools/integrator.hpp"

#include <gsl/gsl_blas.h>
#include <gsl/gsl_errno.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#define ADDED_CONST_TO_C 10.0
#define PI 3.14159265359

// Internal functions and variables
//------------------------------
double q_matrix_integrand(double k, void *params)
{
    struct spectrograph_windowfn_params *wp = (struct spectrograph_windowfn_params*) params;
    double result = spectral_response_window_fn(k, params);

    result *= result * cos(k * wp->delta_v_ij) / PI;

    return result;
}

double signal_matrix_integrand(double k, void *params)
{
    struct spectrograph_windowfn_params *wp = (struct spectrograph_windowfn_params*) params;
    double result = spectral_response_window_fn(k, params);

    result *= debuggin_power_spectrum(k, wp->pixel_width) * result * cos(k * wp->delta_v_ij) / PI;

    return result;
}
//------------------------------
// End of internal functions and variables

OneQSOEstimate::OneQSOEstimate(const char *fname_qso, int n, const double *k)
{
    NUMBER_OF_BANDS = n;
    kband_edges     = k;

    // Construct and read data arrays
    QSOFile qFile(fname_qso);

    double dummy_qso_z, dummy_s2n;

    qFile.readParameters(DATA_SIZE, dummy_qso_z, spect_res, dummy_s2n);
    
    xspace_array    = new double[DATA_SIZE];
    data_array      = new double[DATA_SIZE];
    noise_array     = new double[DATA_SIZE];

    qFile.readData(xspace_array, data_array, noise_array);

    convert_flux2deltaf(data_array, DATA_SIZE);

    convert_lambda2v(median_redshift, xspace_array, DATA_SIZE);

    /* Allocate memory */

    ps_before_fisher_estimate_vector = gsl_vector_alloc(NUMBER_OF_BANDS);

    fisher_matrix             = gsl_matrix_alloc(NUMBER_OF_BANDS, NUMBER_OF_BANDS);

    covariance_matrix         = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
    fiducial_signal_matrix    = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
    inverse_covariance_matrix = covariance_matrix;

    derivative_of_signal_matrices          = new gsl_matrix*[NUMBER_OF_BANDS];
    modified_derivative_of_signal_matrices = new gsl_matrix*[NUMBER_OF_BANDS];

    for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
    {
        derivative_of_signal_matrices[kn]          = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
        modified_derivative_of_signal_matrices[kn] = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
    }  

    isCovInverted    = false;
    areSQMatricesSet = false;
}

OneQSOEstimate::~OneQSOEstimate()
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

    delete [] derivative_of_signal_matrices;
    delete [] modified_derivative_of_signal_matrices;

    delete [] data_array;
    delete [] xspace_array;
    delete [] noise_array;
}

void OneQSOEstimate::setFiducialSignalMatrix()
{
    double  dv_kms   = fabs(xspace_array[1] - xspace_array[0]), temp;

    struct spectrograph_windowfn_params win_params = {0, dv_kms, spect_res};

    Integrator s_integrator(GSL_QAG, signal_matrix_integrand, &win_params);

    for (int i = 0; i < DATA_SIZE; i++)
    {
        win_params.delta_v_ij = 0;

        // temp = s_integrator.evaluate(kvalue_1, kvalue_2);
        temp = s_integrator.evaluateAToInfty(0);
        gsl_matrix_set(fiducial_signal_matrix, i, i, temp);

        for (int j = i + 1; j < DATA_SIZE; j++)
        {
            win_params.delta_v_ij = xspace_array[i] - xspace_array[j];

            // temp = s_integrator.evaluate(kvalue_1, kvalue_2);
            temp = s_integrator.evaluateAToInfty(0);
            
            gsl_matrix_set(fiducial_signal_matrix, i, j, temp);
            gsl_matrix_set(fiducial_signal_matrix, j, i, temp);
        }

        // printf("Progress: %.3f\n", (i+1.) / DATA_SIZE);
        // fflush(stdout);
    }
    // printf("done!\n");
}

void OneQSOEstimate::setDerivativeSMatrices()
{
    // printf("Setting derivative of signal matrices Q_ij(k).\n");
    // fflush(stdout);

    double dv_kms = fabs(xspace_array[1] - xspace_array[0]);

    double temp, kvalue_1, kvalue_2;

    struct spectrograph_windowfn_params win_params = {0, dv_kms, spect_res};

    Integrator q_integrator(GSL_QAG, q_matrix_integrand, &win_params);

    for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
    {
        kvalue_1 = kband_edges[kn];
        kvalue_2 = kband_edges[kn + 1];

        for (int i = 0; i < DATA_SIZE; i++)
        {
            win_params.delta_v_ij = 0;

            temp = q_integrator.evaluate(kvalue_1, kvalue_2);
            // temp = q_integrator.evaluateAToInfty(0);
            gsl_matrix_set(derivative_of_signal_matrices[kn], i, i, temp);

            for (int j = i + 1; j < DATA_SIZE; j++)
            {
                win_params.delta_v_ij = xspace_array[i] - xspace_array[j];

                temp = q_integrator.evaluate(kvalue_1, kvalue_2);
                // temp = q_integrator.evaluateAToInfty(0);
                
                gsl_matrix_set(derivative_of_signal_matrices[kn], i, j, temp);
                gsl_matrix_set(derivative_of_signal_matrices[kn], j, i, temp);
            }

            // printf("Progress: %.3f\n", (i+1.) / DATA_SIZE);
            // fflush(stdout);
        }
        // printf("done!\n");
    }
}

void OneQSOEstimate::computeCSMatrices(const double *ps_estimate)
{
    // printf("Theta: %.3e\n", ps_estimate[0]);
    gsl_matrix_memcpy(covariance_matrix, fiducial_signal_matrix);

    gsl_matrix *temp_matrix = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);

    for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
    {
        gsl_matrix_memcpy(temp_matrix, derivative_of_signal_matrices[kn]);
        gsl_matrix_scale(temp_matrix, ps_estimate[kn]);

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

    gsl_matrix_add_constant(covariance_matrix, ADDED_CONST_TO_C);

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

    gsl_vector_const_view data_view = gsl_vector_const_view_array(data_array, DATA_SIZE);

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

void OneQSOEstimate::oneQSOiteration(const double *ps_estimate)
{
    if (!areSQMatricesSet)
    {
        setFiducialSignalMatrix();
        setDerivativeSMatrices();
        
        areSQMatricesSet = true;
    }
    
    computeCSMatrices(ps_estimate);
    invertCovarianceMatrix();
    computeModifiedDSMatrices();

    computePSbeforeFvector();
    computeFisherMatrix();
}

void OneQSOEstimate::getFFTEstimate(double *ps, int *bincount)
{
    double  dv_kms = fabs(xspace_array[1] - xspace_array[0]), \
            length_v = dv_kms * DATA_SIZE;

    RealField1D rf(data_array, DATA_SIZE, length_v);
    
    rf.fftX2K();

    struct spectrograph_windowfn_params win_params = {0, dv_kms, spect_res};
    
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












