// TODO: Check---and improve when possible---gsl functions

#include "quadratic_estimate.hpp"
#include "matrix_helper.hpp"
#include "global_numbers.hpp"

#include "../io/io_helper_functions.hpp"

#include <gsl/gsl_blas.h>

#include <cmath>
#include <ctime>    /* clock_t, clock, CLOCKS_PER_SEC */
#include <cstdio>
#include <cstdlib> // system
#include <cassert>

int POLYNOMIAL_FIT_DEGREE;

OneDQuadraticPowerEstimate::OneDQuadraticPowerEstimate( const char *fname_list, const char *dir, \
                                                        const SQLookupTable *table, \
                                                        struct palanque_fit_params *pfp)
{
    sq_lookup_table     = table;
    FIDUCIAL_PS_PARAMS  = pfp;

    Z_BIN_COUNTS     = new int[NUMBER_OF_Z_BINS+2]();

    /* Allocate memory */
    pmn_before_fisher_estimate_vector_sum   = gsl_vector_alloc(TOTAL_KZ_BINS); 
    previous_pmn_estimate_vector            = gsl_vector_alloc(TOTAL_KZ_BINS);
    pmn_estimate_vector                     = gsl_vector_calloc(TOTAL_KZ_BINS);
    // fisher_filter                        = gsl_vector_alloc(TOTAL_KZ_BINS);
    fisher_matrix_sum                       = gsl_matrix_alloc(TOTAL_KZ_BINS, TOTAL_KZ_BINS);
    inverse_fisher_matrix_sum               = fisher_matrix_sum;

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
        
        int temp_qso_zbin = qso_estimators[q]->ZBIN;

        // See if qso does not belong to any redshift
        if (temp_qso_zbin < 0)
        {
            temp_qso_zbin = -1;
        }
        else if (temp_qso_zbin >= NUMBER_OF_Z_BINS)
        {
            temp_qso_zbin = NUMBER_OF_Z_BINS;
        }
        
        Z_BIN_COUNTS[temp_qso_zbin + 1]++;
        
    }
    fclose(toRead);

    int tot_z = 0;
    printf("Z bin counts: ");
    for (int zm = 0; zm < NUMBER_OF_Z_BINS+2; zm++)
    {
        printf("%d ", Z_BIN_COUNTS[zm]);
        tot_z += Z_BIN_COUNTS[zm];
    }
    printf("\n");
    printf("Total qso in z bins: %d\nTotal qsos: %d\n", tot_z, NUMBER_OF_QSOS);
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

    invert_matrix_LU(fisher_matrix_sum);

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
                    inverse_fisher_matrix_sum, pmn_before_fisher_estimate_vector_sum, \
                    0, pmn_estimate_vector);
}

void OneDQuadraticPowerEstimate::fitPowerSpectra(double *fit_values)
{
    char tmp_ps_fname[]  = "tmppsfileXXXXXX", \
         tmp_fit_fname[] = "tmpfitfileXXXXXX", \
         buf[200];
    FILE *tmp_fit_file;
    int s1, s2, kn, zm;
    static float fit_params[6]{0};

    s1 = mkstemp(tmp_ps_fname);
    s2 = mkstemp(tmp_fit_fname);

    if (s1 == -1 || s2 == -1)
    {
        fprintf(stderr, "ERROR: Temp filename cannot be generated!\n");
        throw "tmp";
    }

    write_spectrum_estimates(tmp_ps_fname);

    if (fit_params[0] < 1E-8 && fit_params[1] < 1E-8)
    {
        // Do not pass anything as fit parameters 
        sprintf(buf, "python py/lorentzian_fit.py %s %s", tmp_ps_fname, tmp_fit_fname);
    }
    else
    {
        sprintf(buf, "python py/lorentzian_fit.py %s %s %.2e %.2e %.2e %.2e %.2e %.2e", \
                tmp_ps_fname, tmp_fit_fname, \
                fit_params[0], fit_params[1], fit_params[2], \
                fit_params[3], fit_params[4], fit_params[5]);
    }

    printf("%s\n", buf);
    s1 = system(buf);

    if (s1 != 0)
    {
        fprintf(stderr, "Error in fitting.\n");
        throw "fit";
    }

    remove(tmp_ps_fname);

    tmp_fit_file = open_file(tmp_fit_fname, "r");

    fscanf( tmp_fit_file, "%e %e %e %e %e %e\n", \
            &fit_params[0], &fit_params[1], &fit_params[2], \
            &fit_params[3], &fit_params[4], &fit_params[5]);

    for (int i_kz = 0; i_kz < TOTAL_KZ_BINS; i_kz++)
    {
        fscanf(tmp_fit_file, "%le\n", &fit_values[i_kz]);

        getFisherMatrixBinNoFromIndex(i_kz, kn, zm);

        fit_values[i_kz] -= powerSpectrumFiducial(kn, zm);
    }
    
    fclose(tmp_fit_file);
    remove(tmp_fit_fname);
}

void OneDQuadraticPowerEstimate::iterate(int number_of_iterations, const char *fname_base)
{
    char buf[500];
    float total_time = 0, total_time_1it = 0;

    clock_t t;

    double *powerspectra_fits = new double[TOTAL_KZ_BINS];
    gsl_vector_view fit_view  = gsl_vector_view_array(powerspectra_fits, TOTAL_KZ_BINS);

    for (int i = 0; i < number_of_iterations; i++)
    {
        printf("Iteration number %d of %d.\n", i+1, number_of_iterations);
        fflush(stdout);
        
        t = clock();

        // Set total Fisher matrix and omn before F to zero for all k, z bins
        initializeIteration();

        // OneQSOEstimate object decides which redshift it belongs to.
        for (int q = 0; q < NUMBER_OF_QSOS; q++)
        {
            if (qso_estimators[q]->ZBIN < 0 || qso_estimators[q]->ZBIN >= NUMBER_OF_Z_BINS)     continue;

            qso_estimators[q]->oneQSOiteration( &fit_view.vector, \
                                                sq_lookup_table, \
                                                pmn_before_fisher_estimate_vector_sum, fisher_matrix_sum);
            #ifdef DEBUG_ON
            break;
            #endif
        }
        
        #ifdef DEBUG_ON
        break;
        #endif

        try
        {
            invertTotalFisherMatrix();
            computePowerSpectrumEstimates();

            fitPowerSpectra(powerspectra_fits);
        }
        catch (const char* msg)
        {
            fprintf(stderr, "ERROR %s: Fisher matrix is not invertable.\n", msg);
            throw msg;
        }
        
        t               = clock() - t;
        total_time_1it  = ((float) t) / CLOCKS_PER_SEC;
        total_time_1it /= 60.0; //mins
        total_time     += total_time_1it;
        printf("This iteration took %.1f minutes. ", total_time_1it);
        printf("Elapsed time so far is %.1f minutes.\n", total_time);
        printf_time_spent_details();
        
        sprintf(buf, "%s_it%d_quadratic_power_estimate.dat", fname_base, i+1);
        write_spectrum_estimates(buf);

        sprintf(buf, "%s_it%d_fisher_matrix.dat", fname_base, i+1);
        write_fisher_matrix(buf);

        printfSpectra();

        if (hasConverged())
        {
            printf("Iteration has converged.\n");
            break;
        }
    }
}

bool OneDQuadraticPowerEstimate::hasConverged()
{
    double  diff, pMax, p1, p2, r, \
            abs_mean = 0., abs_max = 0.;
    bool ifConverged = true;
    int i_kz;

    for (int zm = 1; zm <= NUMBER_OF_Z_BINS; zm++)
    {
        if (Z_BIN_COUNTS[zm] == 0)  continue;
        
        for (int kn = 0; kn < NUMBER_OF_K_BANDS; kn++)
        {
            i_kz = getFisherMatrixIndex(kn, zm - 1);
            
            p1 = fabs(gsl_vector_get(pmn_estimate_vector, i_kz));
            p2 = fabs(gsl_vector_get(previous_pmn_estimate_vector, i_kz));
            
            diff = fabs(p1 - p2);
            pMax = std::max(p1, p2);
            r    = diff / pMax;

            if (r > CONVERGENCE_EPS)    ifConverged = false;

            abs_mean += r / TOTAL_KZ_BINS;
            abs_max   = std::max(r, abs_max);
        } 
    }

    printf("Mean relative change is %.1e.\n", abs_mean);
    printf("Maximum relative change is %.1e. ", abs_max);
    printf("Old test: Iteration converges when this is less than %.1e\n", CONVERGENCE_EPS);
    
    // Perform a chi-square test as well
    gsl_vector *temp_vector = gsl_vector_alloc(TOTAL_KZ_BINS);

    gsl_vector_sub(previous_pmn_estimate_vector, pmn_estimate_vector);

    if (isFisherInverted)
    {
        invertTotalFisherMatrix();
    }

    gsl_blas_dgemv( CblasNoTrans, 1.0, \
                    fisher_matrix_sum, previous_pmn_estimate_vector, \
                    0, temp_vector);
        
    gsl_blas_ddot(previous_pmn_estimate_vector, temp_vector, &r);

    printf("Chi square convergence test: %.2f. ", r);
    printf("Iteration converges when this is less than %.2f\n", CHISQ_CONVERGENCE_EPS);

    ifConverged = r < CHISQ_CONVERGENCE_EPS;

    return ifConverged;
}

void OneDQuadraticPowerEstimate::write_fisher_matrix(const char *fname)
{
    if (isFisherInverted)
    {
        invertTotalFisherMatrix();
    }

    fprintf_matrix(fname, fisher_matrix_sum);

    printf("Fisher matrix saved as %s.\n", fname);
    fflush(stdout);
}

void OneDQuadraticPowerEstimate::write_spectrum_estimates(const char *fname)
{
    FILE *toWrite;
    int i_kz;
    double z, k, p, e;

    if (!isFisherInverted)
    {
        invertTotalFisherMatrix();
    }

    toWrite = open_file(fname, "w");

    fprintf(toWrite, "%d %d\n", NUMBER_OF_Z_BINS, NUMBER_OF_K_BANDS);

    for (int zm = 0; zm <= NUMBER_OF_Z_BINS+1; zm++)
    {
        fprintf(toWrite, "%d ", Z_BIN_COUNTS[zm]);
    }

    fprintf(toWrite, "\n");

    for (int zm = 1; zm <= NUMBER_OF_Z_BINS; zm++)
    {
        if (Z_BIN_COUNTS[zm] == 0)  continue;

        z = ZBIN_CENTERS[zm-1];

        for (int kn = 0; kn < NUMBER_OF_K_BANDS; kn++)
        {
            i_kz = getFisherMatrixIndex(kn, zm-1);

            k = KBAND_CENTERS[kn];
            p = gsl_vector_get(pmn_estimate_vector, i_kz) + powerSpectrumFiducial(kn, zm-1);
            e = sqrt(gsl_matrix_get(inverse_fisher_matrix_sum, i_kz, i_kz));

            // if (isFisherInverted)
            //     err = sqrt(gsl_matrix_get(fisher_matrix_sum[zm], i, i));
            // else
            //     err = sqrt(gsl_matrix_get(fisher_matrix_sum[zm], i, i)) / gsl_vector_get(fisher_filter, i);
            
            fprintf(toWrite, "%.3lf %e %e %e\n", z, k, p, e);
        }
    }

    fclose(toWrite);
        
    printf("Quadratic 1D Power Spectrum estimate saved as %s.\n", fname);
    fflush(stdout);
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

    for (int zm = 1; zm <= NUMBER_OF_Z_BINS; zm++)
    {
        if (Z_BIN_COUNTS[zm] == 0)  continue;

        printf(" P(%.1f, k) |", ZBIN_CENTERS[zm-1]);
    }
    printf("\n");
    
    for (int kn = 0; kn < NUMBER_OF_K_BANDS; kn++)
    {
        for (int zm = 1; zm <= NUMBER_OF_Z_BINS; zm++)
        {
            if (Z_BIN_COUNTS[zm] == 0)  continue;

            i_kz = getFisherMatrixIndex(kn, zm-1);

            printf(" %.3e |", pmn_estimate_vector->data[i_kz] + powerSpectrumFiducial(kn, zm-1));
        }
        printf("\n");
    }
}

double OneDQuadraticPowerEstimate::powerSpectrumFiducial(int kn, int zm)
{
    return fiducial_power_spectrum(KBAND_CENTERS[kn], ZBIN_CENTERS[zm], FIDUCIAL_PS_PARAMS);
}





















