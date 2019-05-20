#include "quadratic_estimate.hpp"
#include "matrix_helper.hpp"
#include "global_numbers.hpp"

#include "../io/io_helper_functions.hpp"

#include <gsl/gsl_cblas.h>

#include <cmath>
#include <cstdio>
#include <cstdlib> // system
#include <cassert>

#include <string>

//-------------------------------------------------------
int qso_cputime_compare(const void * a, const void * b)
{
    double t_a = ((const qso_computation_time*)a)->est_cpu_time, \
           t_b = ((const qso_computation_time*)b)->est_cpu_time;

    if (t_a <  t_b) return -1;
    if (t_a == t_b) return 0;

    return 1; //if (t_a >  t_b)
}

int index_of_min_element(double *a, int size)
{
    int i = 0;
    for (int j = 1; j < size; j++)
    {
        if (a[j] < a[i])    i = j;
    }

    return i;
}
//-------------------------------------------------------


OneDQuadraticPowerEstimate::OneDQuadraticPowerEstimate( const char *fname_list, const char *dir, \
                                                        pd13_fit_params *pfp)
{
    FIDUCIAL_PS_PARAMS  = pfp;

    Z_BIN_COUNTS     = new int[NUMBER_OF_Z_BINS+2]();

    /* Allocate memory */
    pmn_before_fisher_estimate_vector_sum   = gsl_vector_alloc(TOTAL_KZ_BINS); 
    previous_pmn_estimate_vector            = gsl_vector_alloc(TOTAL_KZ_BINS);
    pmn_estimate_vector                     = gsl_vector_calloc(TOTAL_KZ_BINS);
    // fisher_filter                        = gsl_vector_alloc(TOTAL_KZ_BINS);
    fisher_matrix_sum                       = gsl_matrix_alloc(TOTAL_KZ_BINS, TOTAL_KZ_BINS);
    inverse_fisher_matrix_sum               = gsl_matrix_alloc(TOTAL_KZ_BINS, TOTAL_KZ_BINS);

    isFisherInverted = false; 

    // Read the file
    FILE *toRead = open_file(fname_list, "r");
    fscanf(toRead, "%d\n", &NUMBER_OF_QSOS);

    printf("Number of QSOs: %d\n", NUMBER_OF_QSOS);

    std::vector<std::string> fpaths(NUMBER_OF_QSOS);
    char buf[1024], temp_fname[700];

    for (int q = 0; q < NUMBER_OF_QSOS; q++)
    {
        fscanf(toRead, "%s\n", temp_fname);
        sprintf(buf, "%s/%s", dir, temp_fname);
        fpaths[q] = buf;
    }
    fclose(toRead);

    // Create objects for each QSO
    qso_estimators = new qso_computation_time[NUMBER_OF_QSOS];

    for (int q = 0; q < NUMBER_OF_QSOS; q++)
    {
        qso_estimators[q].qso = new OneQSOEstimate(fpaths[q].c_str());

        int temp_qso_zbin = qso_estimators[q].qso->ZBIN;

        // See if qso does not belong to any redshift
        if (temp_qso_zbin < 0)                      temp_qso_zbin = -1;
        else if (temp_qso_zbin >= NUMBER_OF_Z_BINS) temp_qso_zbin = NUMBER_OF_Z_BINS;
        
        if (temp_qso_zbin == -1 || temp_qso_zbin == NUMBER_OF_Z_BINS)
            qso_estimators[q].est_cpu_time = 0;
        else
            qso_estimators[q].est_cpu_time = pow((double)(qso_estimators[q].qso->DATA_SIZE), 3);

        Z_BIN_COUNTS[temp_qso_zbin + 1]++;
    }
    
    NUMBER_OF_QSOS_OUT = Z_BIN_COUNTS[0] + Z_BIN_COUNTS[NUMBER_OF_Z_BINS+1];

    printf("Z bin counts: ");
    for (int zm = 0; zm < NUMBER_OF_Z_BINS+2; zm++)
    {
        printf("%d ", Z_BIN_COUNTS[zm]);
    }
    printf("\n");
    printf("Number of quasars: %d\n", NUMBER_OF_QSOS);
    printf("QSOs in z bins: %d\n",  NUMBER_OF_QSOS - NUMBER_OF_QSOS_OUT);

    printf("Sorting with respect to estimated cpu time.\n");
    qsort(qso_estimators, NUMBER_OF_QSOS, sizeof(qso_computation_time), qso_cputime_compare);
}

OneDQuadraticPowerEstimate::~OneDQuadraticPowerEstimate()
{
    gsl_vector_free(pmn_before_fisher_estimate_vector_sum);
    gsl_vector_free(previous_pmn_estimate_vector);
    gsl_vector_free(pmn_estimate_vector);

    gsl_matrix_free(fisher_matrix_sum);
    gsl_matrix_free(inverse_fisher_matrix_sum);

    for (int q = 0; q < NUMBER_OF_QSOS; q++)
    {
        delete qso_estimators[q].qso;
    }

    delete [] qso_estimators;
}

void OneDQuadraticPowerEstimate::invertTotalFisherMatrix()
{
    double t = get_time();

    printf("Inverting Fisher matrix.\n");
    fflush(stdout);

    gsl_matrix *fisher_copy = gsl_matrix_alloc(TOTAL_KZ_BINS, TOTAL_KZ_BINS);
    gsl_matrix_memcpy(fisher_copy, fisher_matrix_sum);
    
    invert_matrix_LU(fisher_copy, inverse_fisher_matrix_sum);
    
    gsl_matrix_free(fisher_copy);

    isFisherInverted = true;

    t = get_time() - t;
    time_spent_on_f_inv += t;
}

void OneDQuadraticPowerEstimate::computePowerSpectrumEstimates()
{
    assert(isFisherInverted);

    printf("Estimating power spectrum.\n");
    fflush(stdout);

    gsl_vector_memcpy(previous_pmn_estimate_vector, pmn_estimate_vector);

    cblas_dsymv(CblasRowMajor, CblasUpper, \
                TOTAL_KZ_BINS, 0.5, inverse_fisher_matrix_sum->data, TOTAL_KZ_BINS, \
                pmn_before_fisher_estimate_vector_sum->data, 1, \
                0, pmn_estimate_vector->data, 1);
}

void OneDQuadraticPowerEstimate::fitPowerSpectra(double *fit_values)
{
    char tmp_ps_fname[320], tmp_fit_fname[320], buf[500];

    FILE *tmp_fit_file;
    int s1, s2, kn, zm;
    static double fit_params[6] = { FIDUCIAL_PS_PARAMS->A, \
                                    FIDUCIAL_PS_PARAMS->n, \
                                    FIDUCIAL_PS_PARAMS->alpha, \
                                    FIDUCIAL_PS_PARAMS->B, \
                                    FIDUCIAL_PS_PARAMS->beta, \
                                    FIDUCIAL_PS_PARAMS->lambda};

    sprintf(tmp_ps_fname, "%s/tmppsfileXXXXXX", TMP_FOLDER);
    sprintf(tmp_fit_fname, "%s/tmpfitfileXXXXXX", TMP_FOLDER);

    s1 = mkstemp(tmp_ps_fname);
    s2 = mkstemp(tmp_fit_fname);

    if (s1 == -1 || s2 == -1)
    {
        fprintf(stderr, "ERROR: Temp filename cannot be generated!\n");
        throw "tmp";
    }

    write_spectrum_estimates(tmp_ps_fname);

    if (NUMBER_OF_Z_BINS == 1)
    {
        // Do not pass redshift evolution parameters
       sprintf(buf, "lorentzian_fit.py %s %s %.2le %.2le %.2le %.2le", \
                tmp_ps_fname, tmp_fit_fname, \
                fit_params[0], fit_params[1], \
                fit_params[2], fit_params[5]);
    }
    else
    {
        sprintf(buf, "lorentzian_fit.py %s %s %.2le %.2le %.2le %.2le %.2le %.2le", \
                tmp_ps_fname, tmp_fit_fname, \
                fit_params[0], fit_params[1], fit_params[2], \
                fit_params[3], fit_params[4], fit_params[5]);
    }

    printf("%s\n", buf);
    fflush(stdout);

    s1 = system(buf);

    if (s1 != 0)
    {
        fprintf(stderr, "Error in fitting.\n");
        remove(tmp_ps_fname);
        throw "fit";
    }

    remove(tmp_ps_fname);

    tmp_fit_file = open_file(tmp_fit_fname, "r");

    fscanf( tmp_fit_file, "%le %le %le %le %le %le\n", \
            &fit_params[0], &fit_params[1], &fit_params[2], \
            &fit_params[3], &fit_params[4], &fit_params[5]);

    for (int i_kz = 0; i_kz < TOTAL_KZ_BINS; i_kz++)
    {
        getFisherMatrixBinNoFromIndex(i_kz, kn, zm);
        // Skip last bin
        if (kn == NUMBER_OF_K_BANDS-1)  continue;
        
        fscanf(tmp_fit_file, "%le\n", &fit_values[i_kz]);

        fit_values[i_kz] -= powerSpectrumFiducial(kn, zm);
    }
    
    fclose(tmp_fit_file);
    remove(tmp_fit_fname);
}

void OneDQuadraticPowerEstimate::loadBalancing(std::vector<qso_computation_time*> *queue_qso, int maxthreads)
{
    printf("Load balancing in for %d threads available.\n", maxthreads);
    double load_balance_time = get_time();

    double *bucket_time = new double[maxthreads]();

    for (int q = NUMBER_OF_QSOS-1; q >= NUMBER_OF_QSOS_OUT; q--)
    {
        printf("QSO %d. EST CPU %e.\n", q, qso_estimators[q].est_cpu_time);
        // find min time bucket
        int min_ind = index_of_min_element(bucket_time, maxthreads);

        // add max time consuming to that bucket
        queue_qso[min_ind].push_back(&qso_estimators[q]);
        bucket_time[min_ind] += qso_estimators[q].est_cpu_time;
    }

    printf("Balanced estimated cpu times: ");
    for (int thr = 0; thr < maxthreads; ++thr)  printf("%.1e ", bucket_time[thr]);
    printf("\n");

    delete [] bucket_time;
    load_balance_time = get_time() - load_balance_time;
    
    printf("Load balancing took %.2f min.\n", load_balance_time);
}

void OneDQuadraticPowerEstimate::iterate(int number_of_iterations, const char *fname_base)
{
    char buf[500];
    double total_time = 0, total_time_1it = 0;

    std::vector<qso_computation_time*> *queue_qso = new std::vector<qso_computation_time*>[numthreads];
    loadBalancing(queue_qso, numthreads);

    double *powerspectra_fits = new double[TOTAL_KZ_BINS]();

    for (int i = 0; i < number_of_iterations; i++)
    {
        printf("Iteration number %d of %d.\n", i+1, number_of_iterations);
        fflush(stdout);
        
        total_time_1it = get_time();
    
        // Set total Fisher matrix and omn before F to zero for all k, z bins
        initializeIteration();

#pragma omp parallel
{
        double thread_time = get_time();

        gsl_vector *local_pmn_before_fisher_estimate_vs   = gsl_vector_calloc(TOTAL_KZ_BINS);
        gsl_matrix *local_fisher_ms                       = gsl_matrix_calloc(TOTAL_KZ_BINS, TOTAL_KZ_BINS);

        std::vector<qso_computation_time*> *local_que     = &queue_qso[t_rank];

        printf("Start working in %d/%d thread with %lu qso in queue.\n", t_rank, numthreads, local_que->size());
        fflush(stdout);

        for (std::vector<qso_computation_time*>::iterator it = local_que->begin(); it != local_que->end(); ++it)
            (*it)->qso->oneQSOiteration(powerspectra_fits, local_pmn_before_fisher_estimate_vs, local_fisher_ms);

        thread_time = get_time() - thread_time;

        printf("Done for loop in %d/%d thread in %.2f minutes. Adding F and d critically.\n", \
                t_rank, numthreads, thread_time);
        fflush(stdout);

        #pragma omp critical
        {
            gsl_matrix_add(fisher_matrix_sum, local_fisher_ms);
            gsl_vector_add(pmn_before_fisher_estimate_vector_sum, local_pmn_before_fisher_estimate_vs);
        } 
        
        gsl_vector_free(local_pmn_before_fisher_estimate_vs);
        gsl_matrix_free(local_fisher_ms);
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
        
        total_time_1it  = get_time() - total_time_1it;
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
            printf("Iteration has converged in %d iterations.\n", i+1);
            break;
        }
    }

    delete [] powerspectra_fits;
    delete [] queue_qso;
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
        
        for (int kn = 0; kn < NUMBER_OF_K_BANDS-1; kn++)
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
    // Maybe just do diagonal (dx)^2/F-1_ii
    
    gsl_vector_sub(previous_pmn_estimate_vector, pmn_estimate_vector);

    r = 0;

    for (i_kz = 0; i_kz < TOTAL_KZ_BINS; ++i_kz)
    {
        // Skip last bin
        if ((i_kz+1)%NUMBER_OF_K_BANDS == 0)    continue;

        double  t = gsl_vector_get(previous_pmn_estimate_vector, i_kz), \
                e = gsl_matrix_get(inverse_fisher_matrix_sum, i_kz, i_kz);

        r += (t*t) / e / (TOTAL_KZ_BINS - NUMBER_OF_Z_BINS);
    }

    r = sqrt(r);

    // r = my_cblas_dsymvdot(previous_pmn_estimate_vector, fisher_matrix_sum) / TOTAL_KZ_BINS;

    printf("Chi square convergence test: %.3f per dof. ", r);
    printf("Iteration converges when this is less than %.2f\n", CHISQ_CONVERGENCE_EPS);

    ifConverged = r < CHISQ_CONVERGENCE_EPS;

    return ifConverged;
}

void OneDQuadraticPowerEstimate::write_fisher_matrix(const char *fname)
{
    fprintf_matrix(fname, fisher_matrix_sum);

    printf("Fisher matrix saved as %s.\n", fname);
    fflush(stdout);
}

void OneDQuadraticPowerEstimate::write_spectrum_estimates(const char *fname)
{
    FILE *toWrite;
    int i_kz;
    double z, k, p, e;

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

        for (int kn = 0; kn < NUMBER_OF_K_BANDS-1; kn++)
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
    if (TURN_OFF_SFID)  return 0;
    
    return fiducial_power_spectrum(KBAND_CENTERS[kn], ZBIN_CENTERS[zm], FIDUCIAL_PS_PARAMS);
}





















