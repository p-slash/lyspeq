// TODO: Check---and improve when possible---gsl functions

#include "quadratic_estimate.hpp"
#include "matrix_helper.hpp"

#include "../io/io_helper_functions.hpp"

#include <gsl/gsl_blas.h>

#include <cmath>
#include <cstdio>
#include <cassert>

#define PI 3.14159265359
#define CONVERGENCE_EPS 1E-7

int POLYNOMIAL_FIT_DEGREE;

OneDQuadraticPowerEstimate::OneDQuadraticPowerEstimate( const char *fname_list, const char *dir, \
                                                        int no_bands, \
                                                        const double *k_edges)
{
    NUMBER_OF_BANDS = no_bands;
    kband_edges = k_edges;

    /* Allocate memory */
    ps_before_fisher_estimate_vector_sum    = gsl_vector_alloc(NUMBER_OF_BANDS);
    previous_power_spectrum_estimate_vector = gsl_vector_alloc(NUMBER_OF_BANDS);
    power_spectrum_estimate_vector          = gsl_vector_calloc(NUMBER_OF_BANDS);
    fisher_filter                           = gsl_vector_alloc(NUMBER_OF_BANDS);

    fisher_matrix_sum             = gsl_matrix_alloc(NUMBER_OF_BANDS, NUMBER_OF_BANDS);
    inverse_fisher_matrix_sum     = fisher_matrix_sum;

    isFisherInverted = false; 

    // Create objects for each QSO
    FILE *toRead = open_file(fname_list, "r");
    fscanf(toRead, "%d\n", &NUMBER_OF_QSOS);

    printf("Number of QSOs: %d\n", NUMBER_OF_QSOS);
    
    qso_estimators = new OneQSOEstimate*[NUMBER_OF_QSOS];

    char buf[1024], temp_fname[700];

    for (int q = 0; q < NUMBER_OF_QSOS; q++)
    {
        fscanf(toRead, "%s\n", temp_fname);
        sprintf(buf, "%s/%s", dir, temp_fname);
        
        qso_estimators[q] = new OneQSOEstimate(buf, NUMBER_OF_BANDS, kband_edges);
    }
    
    fclose(toRead);

    // weights_ps_bands      = new double[NUMBER_OF_BANDS];
    k_centers             = new double[NUMBER_OF_BANDS];

    for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
    {
        k_centers[kn] = (kband_edges[kn] + kband_edges[kn + 1]) / 2.;
    }

    // fit_to_power_spectrum = new LnPolynomialFit(POLYNOMIAL_FIT_DEGREE, 1, NUMBER_OF_BANDS);
    // fit_to_power_spectrum->initialize(k_centers);
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
    // delete [] weights_ps_bands;
    delete [] k_centers;
    // delete fit_to_power_spectrum;
}

void OneDQuadraticPowerEstimate::setInitialPSestimateFFT()
{
    double *temp_ps = new double[NUMBER_OF_BANDS];

    gsl_vector_view temp_ps_view = gsl_vector_view_array(temp_ps, NUMBER_OF_BANDS);
    
    int *bincount_q     = new int[NUMBER_OF_BANDS], \
        *bincount_total = new int[NUMBER_OF_BANDS];

    for (int q = 0; q < NUMBER_OF_QSOS; q++)
    {
        qso_estimators[q]->getFFTEstimate(temp_ps, bincount_q);

        gsl_vector_scale(&temp_ps_view.vector, 1./NUMBER_OF_QSOS);
        gsl_vector_add(power_spectrum_estimate_vector, &temp_ps_view.vector);

        for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
        {
            if (q == 0) bincount_total[kn]  = bincount_q[kn];
            else        bincount_total[kn] += bincount_q[kn];
        }
    }

    printf("Initial guess for the power spectrum from FFT:\n");
    for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
    {
        double psev = gsl_vector_get(power_spectrum_estimate_vector, kn) + 1E-10;
        printf("%.2le ", psev);

        weights_ps_bands[kn] = bincount_total[kn] / (psev * psev);
        
        fit_to_power_spectrum->mask_array[kn] = (bincount_total[kn] == 0);

        gsl_matrix_set(inverse_fisher_matrix_sum, kn, kn, 1./weights_ps_bands[kn]);

    }
    printf("\n");

    delete [] bincount_q;
    delete [] bincount_total;
    delete [] temp_ps;
    
    fit_to_power_spectrum->fit(power_spectrum_estimate_vector->data, weights_ps_bands);
    fit_to_power_spectrum->printFit();
}

void OneDQuadraticPowerEstimate::setInitialScaling()
{
    gsl_vector_set(power_spectrum_estimate_vector, 0, 1.);
    // fit_to_power_spectrum->fitted_values[0] = 1.;
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

    // printf("Estimating power spectrum.\n");
    // fflush(stdout);

    gsl_vector_memcpy(previous_power_spectrum_estimate_vector, power_spectrum_estimate_vector);

    //gsl_blas_dsymv( CblasUpper, 0.5, 
    gsl_blas_dgemv( CblasNoTrans, 0.5, \
                    inverse_fisher_matrix_sum, ps_before_fisher_estimate_vector_sum, \
                    0, power_spectrum_estimate_vector);
}

void OneDQuadraticPowerEstimate::filteredEstimates()
{
    gsl_vector *ones_vector = gsl_vector_alloc(NUMBER_OF_BANDS);
    gsl_vector_set_all(ones_vector, 1.);

    gsl_blas_dgemv( CblasNoTrans, 2.0, \
                    fisher_matrix_sum, ones_vector, \
                    0, fisher_filter);

    gsl_vector_free(ones_vector);

    gsl_vector_div(ps_before_fisher_estimate_vector_sum, fisher_filter);
    gsl_vector_memcpy(previous_power_spectrum_estimate_vector, power_spectrum_estimate_vector);
    gsl_vector_memcpy(power_spectrum_estimate_vector, ps_before_fisher_estimate_vector_sum);
}

void OneDQuadraticPowerEstimate::iterate(int number_of_iterations)
{
    for (int i = 0; i < number_of_iterations; i++)
    {
        printf("Iteration number %d of %d.\n", i+1, number_of_iterations);
        fflush(stdout);

        gsl_vector_set_zero(ps_before_fisher_estimate_vector_sum);
        gsl_matrix_set_zero(fisher_matrix_sum);

        for (int q = 0; q < NUMBER_OF_QSOS; q++)
        {
            // qso_estimators[q]->oneQSOiteration(fit_to_power_spectrum->fitted_values);
            qso_estimators[q]->oneQSOiteration(power_spectrum_estimate_vector->data);

            gsl_matrix_add(fisher_matrix_sum, qso_estimators[q]->fisher_matrix);
            gsl_vector_add(ps_before_fisher_estimate_vector_sum, qso_estimators[q]->ps_before_fisher_estimate_vector);
        }

        // invertTotalFisherMatrix();
        // computePowerSpectrumEstimate();

        filteredEstimates();

        for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
        {
            printf("%.3e ", power_spectrum_estimate_vector->data[kn]);
            // weights_ps_bands[kn] = 1. / gsl_matrix_get(inverse_fisher_matrix_sum, kn, kn);
        }
        printf("\n");

        // fit_to_power_spectrum->fit(power_spectrum_estimate_vector->data, weights_ps_bands);
        // fit_to_power_spectrum->printFit();
        // fit_to_power_spectrum->fitted_values[0] = gsl_vector_get(power_spectrum_estimate_vector, 0);
        // printf("%.2e\n", fit_to_power_spectrum->fitted_values[0]);

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
    bool ifConverged = true;

    printf("Relative change in ps estimate: ");
    for (int kn = 0; kn < NUMBER_OF_BANDS; kn++)
    {
        p1 = gsl_vector_get(power_spectrum_estimate_vector, kn);
        p2 = gsl_vector_get(previous_power_spectrum_estimate_vector, kn);
        
        diff = fabs(p1 - p2);
        mx = std::max(p1, p2);

        if (diff / p2 > CONVERGENCE_EPS)
        {
            ifConverged = false;
        }

        printf("%.1le ", diff/mx);
    }

    printf("\n");
    fflush(stdout);

    return ifConverged;
}

void OneDQuadraticPowerEstimate::write_spectrum_estimate(const char *fname)
{
    FILE *toWrite;

    toWrite = open_file(fname, "w");

    fprintf(toWrite, "%d\n", NUMBER_OF_BANDS);
    double err;

    for (int i = 0; i < NUMBER_OF_BANDS; i++)
    {
        if (isFisherInverted)
            err = sqrt(gsl_matrix_get(inverse_fisher_matrix_sum, i, i));
        else
            err = sqrt(gsl_matrix_get(fisher_matrix_sum, i, i)) / gsl_vector_get(fisher_filter, i);
        
        fprintf(toWrite, "%e %e %e\n",  k_centers[i], \
                                        gsl_vector_get(power_spectrum_estimate_vector, i), \
                                        err );
    }

    fclose(toWrite);
    
    printf("Quadratic 1D Power Spectrum estimate saved as %s.\n", fname);
    fflush(stdout);
}






















