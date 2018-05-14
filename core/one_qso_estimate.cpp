#include "one_qso_estimate.hpp"
#include "matrix_helper.hpp"
#include "real_field_1d.hpp"

#include "../io/io_helper_functions.hpp"
#include "../gsltools/integrator.hpp"

#include <gsl/gsl_blas.h>
#include <gsl/gsl_errno.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#define ADDED_CONST_TO_C 0.
#define PI 3.14159265359

double sinc(double x)
{
    if (abs(x) < 1E-10)
    {
        return 1.;
    }

    return sin(x) / x;
}

void convert_lambda2v(double *lambda, int size)
{
    #define SPEED_OF_LIGHT 299792.458
    #define LYA_REST 1215.67

    double mean_lambda = 0;

    for (int i = 0; i < size; i++)
    {
        mean_lambda += lambda[i] / size;
    }

    for (int i = 0; i < size; i++)
    {
        lambda[i] = 2. * SPEED_OF_LIGHT * (1 - sqrt(mean_lambda / lambda[i]));
    }
}

struct windowfn_params
{
    double delta_v_ij;
    double dv_kms;
    double R;
};

double q_matrix_integrand(double k, void *params)
{
    struct windowfn_params *wp = (struct windowfn_params*) params;
    double result = cos(k * wp->delta_v_ij) / PI;
    result *= exp(-k*k * wp->R*wp->R);
    result *= sinc(k * wp->dv_kms / 2.) * sinc(k * wp->dv_kms / 2.);

    return result;
}


void printf_matrix(const gsl_matrix *m, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            printf("%le ", gsl_matrix_get(m, i, j));
        }
        printf("\n");
    }
}

OneQSOEstimate::OneQSOEstimate( const char *fname_qso, int n, const double *k)
{
    NUMBER_OF_BANDS = n;
    kband_edges     = k;

    // Construct and read data arrays
    FILE *toRead = open_file(fname_qso, "r");
    fscanf(toRead, "%d\n", &DATA_SIZE);

    xspace_array    = new double[DATA_SIZE];
    data_array      = new double[DATA_SIZE];
    noise_array     = new double[DATA_SIZE];

    double mean_f = 0;
    for (int i = 0; i < DATA_SIZE; i++)
    {
        fscanf(toRead, "%le %le\n", &xspace_array[i], &data_array[i]);
        noise_array[i] = 0.01;

        mean_f += data_array[i] / DATA_SIZE;
    }

    fclose(toRead);

    // Convert to mean flux
    for (int i = 0; i < DATA_SIZE; i++)
    {
        data_array[i] = (data_array[i] / mean_f) - 1.;
    }

    convert_lambda2v(xspace_array, DATA_SIZE);

    /* Allocate memory */

    ps_before_fisher_estimate_vector = gsl_vector_alloc(NUMBER_OF_BANDS);

    fisher_matrix             = gsl_matrix_alloc(NUMBER_OF_BANDS, NUMBER_OF_BANDS);

    covariance_matrix         = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
    signal_matrix             = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
    inverse_covariance_matrix = covariance_matrix;

    derivative_of_signal_matrices          = new gsl_matrix*[NUMBER_OF_BANDS];
    modified_derivative_of_signal_matrices = new gsl_matrix*[NUMBER_OF_BANDS];

    for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
    {
        derivative_of_signal_matrices[kn]          = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
        modified_derivative_of_signal_matrices[kn] = gsl_matrix_alloc(DATA_SIZE, DATA_SIZE);
    }  

    isCovInverted    = false;
    isQMatricesSet   = false;
}

OneQSOEstimate::~OneQSOEstimate()
{
    gsl_vector_free(ps_before_fisher_estimate_vector);

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

    delete [] data_array;
    delete [] xspace_array;
    delete [] noise_array;
}

void OneQSOEstimate::setDerivativeSMatrices()
{
    if (isQMatricesSet)
    {
        return;
    }

    //printf("Setting derivative of signal matrices Q_ij(k).\n");

    double  dv_kms = abs(xspace_array[1] - xspace_array[0]), \
            R_smooth = 20.;

    double temp, kvalue_1, kvalue_2;

    struct windowfn_params win_params = {0, dv_kms, R_smooth};

    Integrator q_integrator(GSL_QAG, q_matrix_integrand, &win_params);

    for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
    {
        kvalue_1 = kband_edges[kn];
        kvalue_2 = kband_edges[kn + 1];

        for (int i = 0; i < DATA_SIZE; i++)
        {
            win_params.delta_v_ij = 0;

            temp = q_integrator.evaluate(kvalue_1, kvalue_2);

            gsl_matrix_set(derivative_of_signal_matrices[kn], i, i, temp);

            for (int j = i + 1; j < DATA_SIZE; j++)
            {
                win_params.delta_v_ij = xspace_array[i] - xspace_array[j];

                temp  = q_integrator.evaluate(kvalue_1, kvalue_2);

                // temp  = kvalue_2 * sinc(kvalue_2 * delta_x_ij) - kvalue_1 * sinc(kvalue_1 * delta_x_ij);
                // temp /= PI;

                gsl_matrix_set(derivative_of_signal_matrices[kn], i, j, temp);
                gsl_matrix_set(derivative_of_signal_matrices[kn], j, i, temp);
            }
        }
    }

    isQMatricesSet = true;
}

void OneQSOEstimate::computeCSMatrices(const gsl_vector *ps_estimate)
{
    gsl_matrix_set_zero(signal_matrix);

    gsl_matrix *temp_matrix = gsl_matrix_calloc(DATA_SIZE, DATA_SIZE);

    for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
    {
        gsl_matrix_memcpy(temp_matrix, derivative_of_signal_matrices[kn]);
        gsl_matrix_scale(temp_matrix, gsl_vector_get(ps_estimate, kn));

        gsl_matrix_add(signal_matrix, temp_matrix);
    }

    //printf_matrix(signal_matrix, DATA_SIZE);

    gsl_matrix_free(temp_matrix);

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

void OneQSOEstimate::computePSbeforeFvector()
{
    // printf("Power estimate before inverse Fisher weighting.\n");
    // fflush(stdout);

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


void OneQSOEstimate::computeFisherMatrix()
{
    // printf("Computing fisher matrix.\n");
    // fflush(stdout);

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
}

void OneQSOEstimate::oneQSOiteration(const gsl_vector *ps_estimate)
{
    setDerivativeSMatrices();
    computeCSMatrices(ps_estimate);
    invertCovarianceMatrix();
    computeModifiedDSMatrices();

    computePSbeforeFvector();
    computeFisherMatrix();
}

void OneQSOEstimate::getFFTEstimate(double *ps)
{
    double  dv_kms = abs(xspace_array[1] - xspace_array[0]), \
            length_v = dv_kms * DATA_SIZE;

    RealField1D rf(data_array, DATA_SIZE, length_v);

    rf.fftX2K();

    rf.getPowerSpectrum(ps, kband_edges, NUMBER_OF_BANDS);
}












