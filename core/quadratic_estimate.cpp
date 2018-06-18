// TODO: Check---and improve when possible---gsl functions

#include "quadratic_estimate.hpp"
#include "matrix_helper.hpp"
#include "real_field_1d.hpp"

#include "../io/io_helper_functions.hpp"

#include <gsl/gsl_blas.h>

#include <cmath>
#include <cstdio>
#include <cassert>

#define PI 3.14159265359
#define CONVERGENCE_EPS 1E-5

OneDQuadraticPowerEstimate::OneDQuadraticPowerEstimate( const char *fname_list, \
                                                        int no_bands, \
                                                        const double *k_edges)
{
    NUMBER_OF_BANDS = no_bands;
    kband_edges = k_edges;

    /* Allocate memory */
    ps_before_fisher_estimate_vector_sum    = gsl_vector_alloc(NUMBER_OF_BANDS);
    previous_power_spectrum_estimate_vector = gsl_vector_alloc(NUMBER_OF_BANDS);
    power_spectrum_estimate_vector          = gsl_vector_calloc(NUMBER_OF_BANDS);

    fisher_matrix_sum             = gsl_matrix_alloc(NUMBER_OF_BANDS, NUMBER_OF_BANDS);
    inverse_fisher_matrix_sum     = fisher_matrix_sum;

    isFisherInverted = false; 

    // Create objects for each QSO
    FILE *toRead = open_file(fname_list, "r");
    fscanf(toRead, "%d\n", &NUMBER_OF_QSOS);

    qso_estimators = new OneQSOEstimate*[NUMBER_OF_QSOS];

    char buf[1024];

    for (int q = 0; q < NUMBER_OF_QSOS; q++)
    {
        fscanf(toRead, "%s\n", buf);
        qso_estimators[q] = new OneQSOEstimate(buf, NUMBER_OF_BANDS, kband_edges);
    }
    
    fclose(toRead);
}

OneDQuadraticPowerEstimate::~OneDQuadraticPowerEstimate()
{
    gsl_vector_free(ps_before_fisher_estimate_vector_sum);
    gsl_vector_free(previous_power_spectrum_estimate_vector);
    gsl_vector_free(power_spectrum_estimate_vector);

    gsl_matrix_free(fisher_matrix_sum);

    for (int q = 0; q < NUMBER_OF_QSOS; q++)
    {
        delete qso_estimators[q];
    }

    delete [] qso_estimators;
}


void OneDQuadraticPowerEstimate::setInitialPSestimateFFT()
{
    double *temp_ps = new double[NUMBER_OF_BANDS];

    gsl_vector_view temp_ps_view = gsl_vector_view_array(temp_ps, NUMBER_OF_BANDS);

    for (int q = 0; q < NUMBER_OF_QSOS; q++)
    {
        qso_estimators[q]->getFFTEstimate(temp_ps);

        gsl_vector_scale(&temp_ps_view.vector, 1./NUMBER_OF_QSOS);
        gsl_vector_add(power_spectrum_estimate_vector, &temp_ps_view.vector);
    }

    printf("Initial guess for the power spectrum from FFT:\n");
    for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
    {
        printf("%e ", gsl_vector_get(power_spectrum_estimate_vector, kn));
    }
    printf("\n");
}

void OneDQuadraticPowerEstimate::invertTotalFisherMatrix()
{
    int status = invert_matrix_cholesky(fisher_matrix_sum);

    if (status == GSL_EDOM)
    {
        fprintf(stderr, "ERROR: Fisher matrix is not positive definite.\n");
        throw "FIS";
    }

    isFisherInverted = true;
}

void OneDQuadraticPowerEstimate::computePowerSpectrumEstimate()
{
    assert(isFisherInverted);

    printf("Estimating power spectrum.\n");
    fflush(stdout);

    gsl_vector_memcpy(previous_power_spectrum_estimate_vector, power_spectrum_estimate_vector);

    //gsl_blas_dsymv( CblasUpper, 0.5, 
    gsl_blas_dgemv( CblasNoTrans, 0.5, \
                    inverse_fisher_matrix_sum, ps_before_fisher_estimate_vector_sum, \
                    0, power_spectrum_estimate_vector);
}

void OneDQuadraticPowerEstimate::iterate(int number_of_iterations)
{
    for (int i = 0; i < number_of_iterations; i++)
    {
        printf("Start iteration number %d of %d.\n", i+1, number_of_iterations);

        gsl_vector_set_zero(ps_before_fisher_estimate_vector_sum);
        gsl_matrix_set_zero(fisher_matrix_sum);

        for (int q = 0; q < NUMBER_OF_QSOS; q++)
        {
            qso_estimators[q]->oneQSOiteration(power_spectrum_estimate_vector);

            gsl_matrix_add(fisher_matrix_sum, qso_estimators[q]->fisher_matrix);
            gsl_vector_add(ps_before_fisher_estimate_vector_sum, qso_estimators[q]->ps_before_fisher_estimate_vector);
        }

        invertTotalFisherMatrix();
        computePowerSpectrumEstimate();

        if (hasConverged())
        {
            printf("Iteration has converged.\n");
            break;
        }
    }
}

bool OneDQuadraticPowerEstimate::hasConverged()
{
    double diff, mx, p1, p2;

    for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
    {
        p1 = gsl_vector_get(power_spectrum_estimate_vector, kn);
        p2 = gsl_vector_get(previous_power_spectrum_estimate_vector, kn);
        
        diff = abs(p1 - p2);
        mx = std::max(p1, p2);

        if (diff > CONVERGENCE_EPS * mx)
        {
            return false;
        }
    }

    return true;
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






















