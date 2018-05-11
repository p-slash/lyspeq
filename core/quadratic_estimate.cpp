// TODO: Check---and improve when possible---gsl functions

#include "quadratic_estimate.hpp"
#include "matrix_helper.hpp"

#include "../io/io_helper_functions.hpp"

#include <gsl/gsl_blas.h>

#include <cmath>
#include <cstdio>
#include <cassert>

#define PI 3.14159265359
#define ADDED_CONST_TO_C 10.0

double sinc(double x)
{
    if (abs(x) < 1E-10)
    {
        return 1.;
    }

    return sin(x) / x;
}

OneDQuadraticPowerEstimate::OneDQuadraticPowerEstimate( int data_size, \
                                                        const double *xspace, \
                                                        const double *delta, \
                                                        const double *noise, \
                                                        int no_bands, \
                                                        const double *k_edges)
{
    DATA_SIZE       = data_size;
    NUMBER_OF_BANDS = no_bands;

    /* Create vector views. Copies of arrays are NOT stored. */
    xspace_array  = xspace;
    data_array    = delta;
    noise_array   = noise;

    /* k bands */
    kband_edges = k_edges;

    /* Allocate memory */
    printf("Allocate memory for C, S, F matrices and P, d vectors.\n");
    fflush(stdout);

    ps_before_fisher_estimate_vector = gsl_vector_alloc(NUMBER_OF_BANDS);
    power_spectrum_estimate_vector   = gsl_vector_calloc(NUMBER_OF_BANDS);

    fisher_matrix             = gsl_matrix_alloc(NUMBER_OF_BANDS, NUMBER_OF_BANDS);
    inverse_fisher_matrix     = fisher_matrix;

    covariance_matrix         = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
    signal_matrix             = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
    inverse_covariance_matrix = covariance_matrix;

    printf("Allocating memory for %d many Q and Q-slash.\n", NUMBER_OF_BANDS);
    fflush(stdout);

    derivative_of_signal_matrices          = new gsl_matrix*[NUMBER_OF_BANDS];
    modified_derivative_of_signal_matrices = new gsl_matrix*[NUMBER_OF_BANDS];

    for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
    {
        derivative_of_signal_matrices[kn]          = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
        modified_derivative_of_signal_matrices[kn] = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
    } 

    isFisherInverted = false;
    isCovInverted    = false;
    isQMatricesSet   = false; 
}

OneDQuadraticPowerEstimate::~OneDQuadraticPowerEstimate()
{
    delete [] kband_edges;

    gsl_vector_free(ps_before_fisher_estimate_vector);
    gsl_vector_free(power_spectrum_estimate_vector);

    gsl_matrix_free(fisher_matrix);

    gsl_matrix_free(covariance_matrix);
    gsl_matrix_free(signal_matrix);

    for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
    {
        gsl_matrix_free(derivative_of_signal_matrices[kn]);
        gsl_matrix_free(modified_derivative_of_signal_matrices[kn]);
    }

    delete [] derivative_of_signal_matrices;
    delete [] modified_derivative_of_signal_matrices;
}

void OneDQuadraticPowerEstimate::setDerivativeSMatrices()
{
    if (isQMatricesSet)
    {
        return;
    }

    printf("Setting derivative of signal matrices Q_ij(k).\n");

    double delta_x_ij, temp, kvalue_1, kvalue_2;

    for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
    {
        kvalue_1 = kband_edges[kn];
        kvalue_2 = kband_edges[kn + 1];

        for (int i = 0; i < DATA_SIZE; i++)
        {
            for (int j = i; j < DATA_SIZE; j++)
            {
                delta_x_ij = xspace_array[i] - xspace_array[j];

                temp  = kvalue_2 * sinc(kvalue_2 * delta_x_ij) - kvalue_1 * sinc(kvalue_1 * delta_x_ij);
                
                temp /= PI;

                gsl_matrix_set(derivative_of_signal_matrices[kn], i, j, temp);
                gsl_matrix_set(derivative_of_signal_matrices[kn], j, i, temp);
            }
        }
    }

    isQMatricesSet = true;
}

void OneDQuadraticPowerEstimate::computeCSMatrices()
{
    printf("Computing signal matrix.\n");
    fflush(stdout);

    gsl_matrix_set_zero(signal_matrix);

    gsl_matrix *temp_matrix = gsl_matrix_calloc(DATA_SIZE, DATA_SIZE);

    for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
    {
        gsl_matrix_memcpy(temp_matrix, derivative_of_signal_matrices[kn]);
        gsl_matrix_scale(temp_matrix, gsl_vector_get(power_spectrum_estimate_vector, kn));

        gsl_matrix_add(signal_matrix, temp_matrix);
    }

    if (gsl_matrix_isnull(signal_matrix))
    {
        printf("Signal matrix is null.\n");
    }

    // free temp matrix if unnecessary
    gsl_matrix_free(temp_matrix);

    printf("Finding covariance matrix.\n");
    double temp, nn;

    gsl_matrix_memcpy(covariance_matrix, signal_matrix);

    for (int i = 0; i < DATA_SIZE; i++)
    {
        nn = noise_array[i];
        nn *= nn;

        temp = gsl_matrix_get(covariance_matrix, i, i);

        gsl_matrix_set(covariance_matrix, i, i, temp + nn);
    }

    gsl_matrix_add_constant(covariance_matrix, ADDED_CONST_TO_C);

    isCovInverted    = false;
    isFisherInverted = false;
}

void OneDQuadraticPowerEstimate::invertCovarianceMatrix()
{
    printf("Inverting covariance matrix.\n");
    fflush(stdout);

    invert_matrix_cholesky(covariance_matrix);

    isCovInverted    = true;
    isFisherInverted = false;
}

void OneDQuadraticPowerEstimate::computeModifiedDSMatrices()
{
    printf("Setting modified derivative of signal matrices Q-slash_ij(k).\n");
    fflush(stdout);

    assert(isCovInverted);

    gsl_matrix *temp_matrix = gsl_matrix_calloc(DATA_SIZE, DATA_SIZE);

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

void OneDQuadraticPowerEstimate::computePSbeforeFvector()
{
    printf("Power estimate before inverse Fisher weighting.\n");
    fflush(stdout);

    gsl_vector *temp_vector = gsl_vector_calloc(DATA_SIZE);

    gsl_vector_const_view data_view = gsl_vector_const_view_array(data_array, DATA_SIZE);

    double temp_bk, temp_d;

    for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
    {
        temp_bk = trace_of_2matrices(modified_derivative_of_signal_matrices[kn], noise_array, DATA_SIZE);

        /*
        gsl_blas_dsymv( CblasUpper, 1.0, \
                        modified_derivative_of_signal_matrices[kn], &data_view.vector, \
                        0, temp_vector);
        */
        gsl_blas_dgemv( CblasNoTrans, 1.0, \
                        modified_derivative_of_signal_matrices[kn], &data_view.vector, \
                        0, temp_vector);

        gsl_blas_ddot(&data_view.vector, temp_vector, &temp_d);

        gsl_vector_set(ps_before_fisher_estimate_vector, kn, temp_d - temp_bk);
    }

    gsl_vector_free(temp_vector);
}

void OneDQuadraticPowerEstimate::computeFisherMatrix()
{
    printf("Computing fisher matrix.\n");
    fflush(stdout);

    double temp;

    for (int k = 0; k < NUMBER_OF_BANDS; k++)
    {
        for (int q = k; q < NUMBER_OF_BANDS; q++)
        {
            temp = trace_of_2matrices(  modified_derivative_of_signal_matrices[k], \
                                        derivative_of_signal_matrices[q], \
                                        DATA_SIZE);

            gsl_matrix_set(fisher_matrix, k, q, temp/2.0);
            gsl_matrix_set(fisher_matrix, q, k, temp/2.0);
        }
    }

    isFisherInverted = false;
}

void OneDQuadraticPowerEstimate::invertFisherMatrix()
{
    printf("Inverting Fisher matrix.\n");
    fflush(stdout);

    invert_matrix_cholesky(fisher_matrix);

    isFisherInverted = true;
}

void OneDQuadraticPowerEstimate::computePowerSpectrumEstimate()
{
    assert(isFisherInverted);

    printf("Estimating power spectrum.\n");
    fflush(stdout);

    //gsl_blas_dsymv( CblasUpper, 0.5, 
    gsl_blas_dgemv( CblasNoTrans, 0.5, \
                    inverse_fisher_matrix, ps_before_fisher_estimate_vector, \
                    0, power_spectrum_estimate_vector);
}

void OneDQuadraticPowerEstimate::iterate(int number_of_iterations)
{
    for (int i = 0; i < number_of_iterations; i++)
    {
        printf("Start iteration number %d of %d.\n", i+1, number_of_iterations);

        setDerivativeSMatrices();

        computeCSMatrices();
        invertCovarianceMatrix();

        computeModifiedDSMatrices();

        computePSbeforeFvector();

        computeFisherMatrix();
        invertFisherMatrix();

        computePowerSpectrumEstimate();
    }
}

void OneDQuadraticPowerEstimate::write_spectrum_estimate(const char *fname)
{
    FILE *toWrite;

    toWrite = open_file(fname, "w");

    fprintf(toWrite, "%d\n", NUMBER_OF_BANDS);
    
    for (int i = 0; i < NUMBER_OF_BANDS; i++)
    {
        double k_center = (kband_edges[i] + kband_edges[i + 1]) / 2.;

        fprintf(toWrite, "%e %e\n", k_center, gsl_vector_get(power_spectrum_estimate_vector, i));
    }

    fclose(toWrite);
    
    printf("Quadratic 1D Power Spectrum estimate saved as %s.\n", fname);
}






















