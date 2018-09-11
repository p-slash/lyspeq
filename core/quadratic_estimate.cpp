// TODO: Check---and improve when possible---gsl functions

#include "quadratic_estimate.hpp"
#include "matrix_helper.hpp"
#include "global_numbers.hpp"

#include "../io/io_helper_functions.hpp"

#include <gsl/gsl_blas.h>

#include <cmath>
#include <ctime>    /* clock_t, clock, CLOCKS_PER_SEC */
#include <cstdio>
#include <cassert>

int POLYNOMIAL_FIT_DEGREE;

OneDQuadraticPowerEstimate::OneDQuadraticPowerEstimate( const char *fname_list, const char *dir, \
                                                        const SQLookupTable *table, \
                                                        struct palanque_fit_params *pfp)
{
    sq_lookup_table     = table;
    FIDUCIAL_PS_PARAMS  = pfp;

    Z_BIN_COUNTS     = new int[NUMBER_OF_Z_BINS]();

    /* Allocate memory */
    pmn_before_fisher_estimate_vector_sum   = gsl_vector_alloc(TOTAL_KZ_BINS); 
    previous_pmn_estimate_vector            = gsl_vector_alloc(TOTAL_KZ_BINS);
    pmn_estimate_vector                     = gsl_vector_calloc(TOTAL_KZ_BINS);
    // fisher_filter                        = gsl_vector_alloc(TOTAL_KZ_BINS);
    fisher_matrix_sum                       = gsl_matrix_alloc(TOTAL_KZ_BINS, TOTAL_KZ_BINS);

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
        
        qso_estimators[q] = new OneQSOEstimate(buf);
        Z_BIN_COUNTS[qso_estimators[q]->ZBIN]++;
    }
    
    fclose(toRead);
}

OneDQuadraticPowerEstimate::~OneDQuadraticPowerEstimate()
{
    gsl_vector_free(pmn_before_fisher_estimate_vector_sum);
    gsl_vector_free(previous_pmn_estimate_vector);
    gsl_vector_free(pmn_estimate_vector);

    gsl_matrix_free(fisher_matrix_sum);

    for (int q = 0; q < NUMBER_OF_QSOS; q++)
    {
        delete qso_estimators[q];
    }

    delete [] qso_estimators;
}

void OneDQuadraticPowerEstimate::invertTotalFisherMatrix()
{
    clock_t t = clock();

    printf("Inverting Fisher matrix.\n");
    fflush(stdout);

    int status = invert_matrix_cholesky(fisher_matrix_sum);

    if (status == GSL_EDOM)
    {
        fprintf(stderr, "ERROR: Fisher matrix is not positive definite!\n");
        write_fisher_matrix("./error_dump");
        throw "FIS";
    }

    isFisherInverted = !isFisherInverted;
    t = clock() - t;
    time_spent_on_f_inv += ((float) t) / CLOCKS_PER_SEC;
}

void OneDQuadraticPowerEstimate::computePowerSpectrumEstimates()
{
    assert(isFisherInverted);

    printf("Estimating power spectrum.\n");
    fflush(stdout);

    gsl_vector_memcpy(previous_pmn_estimate_vector, pmn_estimate_vector);

    //gsl_blas_dsymv( CblasUpper, 0.5, 
    gsl_blas_dgemv( CblasNoTrans, 0.5, \
                    fisher_matrix_sum, pmn_before_fisher_estimate_vector_sum, \
                    0, pmn_estimate_vector);
}

void OneDQuadraticPowerEstimate::iterate(int number_of_iterations)
{
    float total_time = 0, total_time_1it = 0;

    clock_t t;

    for (int i = 0; i < number_of_iterations; i++)
    {
        printf("Iteration number %d of %d.\n", i+1, number_of_iterations);
        fflush(stdout);
        
        t = clock();

        // Set to zero for all z bins
        initializeIteration();

        // OneQSOEstimate object decides which redshift it belongs to.
        for (int q = 0; q < NUMBER_OF_QSOS; q++)
        {   
            qso_estimators[q]->oneQSOiteration( pmn_estimate_vector, \
                                                sq_lookup_table, \
                                                pmn_before_fisher_estimate_vector_sum, fisher_matrix_sum);
        }

        // Invert for all z bins
        invertTotalFisherMatrix();
        computePowerSpectrumEstimates();
        
        printfSpectra();

        // filteredEstimates();

        if (hasConverged())
        {
            printf("Iteration has converged.\n");
            
            t = clock() - t;
            total_time_1it = ((float) t) / CLOCKS_PER_SEC;
            total_time_1it /= 60.0; //mins
            total_time += total_time_1it;
            printf("This iteration took %.1f minutes in total. Elapsed time so far is %.1f minutes.\n", total_time_1it, total_time);
            break;
        }

        t = clock() - t;
        total_time_1it = ((float) t) / CLOCKS_PER_SEC;
        total_time_1it /= 60.0; //mins
        total_time += total_time_1it;
        printf("This iteration took %.1f minutes in total. Elapsed time so far is %.1f minutes.\n", total_time_1it, total_time);
        printf_time_spent_details();
    }
    
    printf("Iteration has ended. Total time elapsed is %.1f minutes.\n", total_time);
    printf_time_spent_details();
}

bool OneDQuadraticPowerEstimate::hasConverged()
{
    double diff, mx, p1, p2;
    bool ifConverged = true;
    int i_kz;

    for (int zm = 0; zm < NUMBER_OF_Z_BINS; ++zm)
    {
        if (Z_BIN_COUNTS[zm] == 0)  continue;

        printf("Relative change in ps estimate for redshift range %.2f: ", ZBIN_CENTERS[zm]);
        
        for (int kn = 0; kn < NUMBER_OF_K_BANDS; kn++)
        {
            i_kz = getFisherMatrixIndex(kn, zm);
            p1 = gsl_vector_get(pmn_estimate_vector, i_kz);
            p2 = gsl_vector_get(previous_pmn_estimate_vector, i_kz);
            
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
    }
    
    return ifConverged;
}

void OneDQuadraticPowerEstimate::write_fisher_matrix(const char *fname_base)
{
    if (isFisherInverted)
    {
        invertTotalFisherMatrix();
    }

    FILE *toWrite;
    char buf[500];
    
    sprintf(buf, "%s_fisher_matrix.dat", fname_base);
    toWrite = open_file(buf, "w");

    fprintf_matrix(toWrite, fisher_matrix_sum);

    fclose(toWrite);
}

void OneDQuadraticPowerEstimate::write_spectrum_estimates(const char *fname_base)
{
    FILE *toWrite;
    char buf[500];
    int i_kz;

    for (int zm = 0; zm < NUMBER_OF_Z_BINS; ++zm)
    {
        if (Z_BIN_COUNTS[zm] == 0)  continue;

        sprintf(buf, "%s_%dspectra_z%.2f_qso_power_estimate.dat", fname_base, Z_BIN_COUNTS[zm], ZBIN_CENTERS[zm]);

        toWrite = open_file(buf, "w");

        fprintf(toWrite, "%d\n", NUMBER_OF_K_BANDS);
        double err;

        for (int kn = 0; kn < NUMBER_OF_K_BANDS; kn++)
        {
            i_kz = getFisherMatrixIndex(kn, zm);

            err = sqrt(gsl_matrix_get(fisher_matrix_sum, i_kz, i_kz));

            // if (isFisherInverted)
            //     err = sqrt(gsl_matrix_get(fisher_matrix_sum[zm], i, i));
            // else
            //     err = sqrt(gsl_matrix_get(fisher_matrix_sum[zm], i, i)) / gsl_vector_get(fisher_filter, i);
            
            fprintf(toWrite, "%e %e %e\n",  KBAND_CENTERS[kn], \
                                            gsl_vector_get(pmn_estimate_vector, i_kz) + powerSpectrumValue(kn, zm), \
                                            err );
        }

        fclose(toWrite);
        
        printf("Quadratic 1D Power Spectrum estimate saved as %s.\n", buf);
        fflush(stdout);
    }
}

void OneDQuadraticPowerEstimate::initializeIteration()
{
    gsl_vector_set_zero(pmn_before_fisher_estimate_vector_sum);
    gsl_matrix_set_zero(fisher_matrix_sum);
    isFisherInverted = false;
}

void OneDQuadraticPowerEstimate::printfSpectra()
{
    int i_kz;

    for (int zm = 0; zm < NUMBER_OF_Z_BINS; ++zm)
    {
        if (Z_BIN_COUNTS[zm] == 0)  continue;

        printf("P_m(zm, kn) at z=%.2f: ", ZBIN_CENTERS[zm]);
        for (int kn = 0; kn < NUMBER_OF_K_BANDS; kn++)
        {
            i_kz = getFisherMatrixIndex(kn, zm);

            printf("%.3e ", pmn_estimate_vector->data[i_kz] + powerSpectrumValue(kn, zm));
            // weights_pmn_bands[kn] = 1. / gsl_matrix_get(fisher_matrix_sum, kn, kn);
        }
        printf("\n");
    }
}

double OneDQuadraticPowerEstimate::powerSpectrumValue(int kn, int zm)
{
    return fiducial_power_spectrum(KBAND_CENTERS[kn], ZBIN_CENTERS[zm], FIDUCIAL_PS_PARAMS);
}

void OneDQuadraticPowerEstimate::setInitialPSestimateFFT()
{
    printf("WARNING: Setting initial estimate with FFT DOES NOT work. It is kept for archival purposes.\n");
    /*
    double *temp_ps = new double[NUMBER_OF_K_BANDS];

    gsl_vector_view temp_ps_view = gsl_vector_view_array(temp_ps, NUMBER_OF_K_BANDS);
    
    int *bincount_q     = new int[NUMBER_OF_K_BANDS], \
        *bincount_total = new int[NUMBER_OF_K_BANDS];

    for (int q = 0; q < NUMBER_OF_QSOS; q++)
    {
        qso_estimators[q]->getFFTEstimate(temp_ps, bincount_q);

        gsl_vector_scale(&temp_ps_view.vector, 1./NUMBER_OF_QSOS);
        gsl_vector_add(pmn_estimate_vector, &temp_ps_view.vector);

        for (int kn = 0; kn < NUMBER_OF_K_BANDS; kn++)
        {
            if (q == 0) bincount_total[kn]  = bincount_q[kn];
            else        bincount_total[kn] += bincount_q[kn];
        }
    }

    printf("Initial guess for the power spectrum from FFT:\n");
    for (int kn = 0; kn < NUMBER_OF_K_BANDS; kn++)
    {
        double psev = gsl_vector_get(pmn_estimate_vector, kn) + 1E-10;
        printf("%.2le ", psev);

        weights_pmn_bands[kn] = bincount_total[kn] / (psev * psev);
        
        fit_to_pmn->mask_array[kn] = (bincount_total[kn] == 0);

        gsl_matrix_set(fisher_matrix_sum, kn, kn, 1./weights_pmn_bands[kn]);

    }
    printf("\n");

    delete [] bincount_q;
    delete [] bincount_total;
    delete [] temp_ps;
    
    fit_to_pmn->fit(pmn_estimate_vector->data, weights_pmn_bands);
    fit_to_pmn->printFit(); 
    */
}

void OneDQuadraticPowerEstimate::setInitialScaling()
{
    printf("WARNING: Setting initial scaling to 1 DOES NOT work. It is kept for archival purposes.\n");

    // gsl_vector_set(pmn_estimate_vector, 0, 1.);
    // fit_to_pmn->fitted_values[0] = 1.;
}

void OneDQuadraticPowerEstimate::filteredEstimates()
{
    printf("WARNING: Filtered estimates DOES NOT work. It is kept for archival purposes.\n");

    /*
    gsl_vector *ones_vector = gsl_vector_alloc(NUMBER_OF_K_BANDS);
    gsl_vector_set_all(ones_vector, 1.);

    gsl_blas_dgemv( CblasNoTrans, 2.0, \
                    fisher_matrix_sum, ones_vector, \
                    0, fisher_filter);

    gsl_vector_free(ones_vector);

    gsl_vector_div(pmn_before_fisher_estimate_vector_sum, fisher_filter);
    gsl_vector_memcpy(previous_pmn_estimate_vector, pmn_estimate_vector);
    gsl_vector_memcpy(pmn_estimate_vector, pmn_before_fisher_estimate_vector_sum);
    */
}



















