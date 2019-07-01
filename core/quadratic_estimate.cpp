#include "core/quadratic_estimate.hpp"
#include "core/matrix_helper.hpp"
#include "core/global_numbers.hpp"
#include "core/fiducial_cosmology.hpp"

#include "io/io_helper_functions.hpp"
#include "io/logger.hpp"

#include <gsl/gsl_cblas.h>

#include <cmath>
#include <cstdio>
#include <cstdlib> // system
#include <cassert>

#include <string>
#include <sstream>      // std::ostringstream

//-------------------------------------------------------
int qso_cputime_compare(const void * a, const void * b)
{
    double t_a = ((const qso_computation_time*)a)->est_cpu_time,
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


OneDQuadraticPowerEstimate::OneDQuadraticPowerEstimate(const char *fname_list, const char *dir)
{
    Z_BIN_COUNTS     = new int[bins::NUMBER_OF_Z_BINS+2]();

    // Allocate memory
    pmn_before_fisher_estimate_vector_sum   = gsl_vector_alloc(bins::TOTAL_KZ_BINS); 
    previous_pmn_estimate_vector            = gsl_vector_alloc(bins::TOTAL_KZ_BINS);
    pmn_estimate_vector                     = gsl_vector_calloc(bins::TOTAL_KZ_BINS);
    fisher_matrix_sum                       = gsl_matrix_alloc(bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);
    inverse_fisher_matrix_sum               = gsl_matrix_alloc(bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);

    isFisherInverted = false; 

    // Read the file
    FILE *toRead = ioh::open_file(fname_list, "r");
    fscanf(toRead, "%d\n", &NUMBER_OF_QSOS);

    LOG::LOGGER.STD("Number of QSOs: %d\n", NUMBER_OF_QSOS);

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

        if (qso_estimators[q].qso->ZBIN == -1 || qso_estimators[q].qso->ZBIN == bins::NUMBER_OF_Z_BINS)
            qso_estimators[q].est_cpu_time = 0;
        else
            qso_estimators[q].est_cpu_time = pow((double)(qso_estimators[q].qso->DATA_SIZE), 3);

        Z_BIN_COUNTS[qso_estimators[q].qso->ZBIN + 1]++;
    }
    
    NUMBER_OF_QSOS_OUT = Z_BIN_COUNTS[0] + Z_BIN_COUNTS[bins::NUMBER_OF_Z_BINS+1];

    LOG::LOGGER.STD("Z bin counts: ");
    for (int zm = 0; zm < bins::NUMBER_OF_Z_BINS+2; zm++)
        LOG::LOGGER.STD("%d ", Z_BIN_COUNTS[zm]);
    LOG::LOGGER.STD("\nNumber of quasars: %d\nQSOs in z bins: %d\n", NUMBER_OF_QSOS, NUMBER_OF_QSOS - NUMBER_OF_QSOS_OUT);

    LOG::LOGGER.STD("Sorting with respect to estimated cpu time.\n");
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
        delete qso_estimators[q].qso;

    delete [] qso_estimators;
}

void OneDQuadraticPowerEstimate::invertTotalFisherMatrix()
{
    double t = mytime::getTime();

    LOG::LOGGER.STD("Inverting Fisher matrix.\n");

    gsl_matrix *fisher_copy = gsl_matrix_alloc(bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);
    gsl_matrix_memcpy(fisher_copy, fisher_matrix_sum);
    
    mxhelp::invertMatrixLU(fisher_copy, inverse_fisher_matrix_sum);
    
    gsl_matrix_free(fisher_copy);

    isFisherInverted = true;

    t = mytime::getTime() - t;
    mytime::time_spent_on_f_inv += t;
}

void OneDQuadraticPowerEstimate::computePowerSpectrumEstimates()
{
    assert(isFisherInverted);

    LOG::LOGGER.STD("Estimating power spectrum.\n");

    gsl_vector_memcpy(previous_pmn_estimate_vector, pmn_estimate_vector);

    cblas_dsymv(CblasRowMajor, CblasUpper,
                bins::TOTAL_KZ_BINS, 0.5, inverse_fisher_matrix_sum->data, bins::TOTAL_KZ_BINS,
                pmn_before_fisher_estimate_vector_sum->data, 1,
                0, pmn_estimate_vector->data, 1);
}

void OneDQuadraticPowerEstimate::_fitPowerSpectra(double *fit_values)
{
    char tmp_ps_fname[320], tmp_fit_fname[320];
    std::ostringstream command;

    FILE *tmp_fit_file;
    int s1, s2, kn, zm;
    static pd13::pd13_fit_params iteration_fits = pd13::FIDUCIAL_PD13_PARAMS;

    sprintf(tmp_ps_fname, "%s/tmppsfileXXXXXX", TMP_FOLDER);
    sprintf(tmp_fit_fname, "%s/tmpfitfileXXXXXX", TMP_FOLDER);

    s1 = mkstemp(tmp_ps_fname);
    s2 = mkstemp(tmp_fit_fname);

    if (s1 == -1 || s2 == -1)
    {
        LOG::LOGGER.ERR("ERROR: Temp filename cannot be generated!\n");
        throw "tmp";
    }

    writeSpectrumEstimates(tmp_ps_fname);

    command << "lorentzian_fit.py " << tmp_ps_fname << " " << tmp_fit_fname << " "
            << iteration_fits.A << " " << iteration_fits.n << " " << iteration_fits.n << " ";

    // Do not pass redshift parameters is there is only one redshift bin
    if (bins::NUMBER_OF_Z_BINS > 1)  command << iteration_fits.B << " " << iteration_fits.beta << " ";
    
    command << iteration_fits.lambda 
            << " >> " << LOG::LOGGER.getFileName(LOG::TYPE::STD);

    LOG::LOGGER.STD("%s\n", command.str().c_str());
    LOG::LOGGER.close();

    // Print from python does not go into LOG::LOGGER
    s1 = system(command.str().c_str());
    
    LOG::LOGGER.reopen();

    if (s1 != 0)
    {
        LOG::LOGGER.ERR("Error in fitting.\n");
        remove(tmp_ps_fname);
        throw "fit";
    }

    remove(tmp_ps_fname);

    tmp_fit_file = ioh::open_file(tmp_fit_fname, "r");

    fscanf( tmp_fit_file, "%le %le %le %le %le %le\n",
            &iteration_fits.A, &iteration_fits.n, &iteration_fits.alpha,
            &iteration_fits.B, &iteration_fits.beta, &iteration_fits.lambda);

    for (int i_kz = 0; i_kz < bins::TOTAL_KZ_BINS; i_kz++)
    {
        SKIP_LAST_K_BIN_WHEN_ENABLED(i_kz)

        bins::getFisherMatrixBinNoFromIndex(i_kz, kn, zm);   
        
        fscanf(tmp_fit_file, "%le\n", &fit_values[i_kz]);

        fit_values[i_kz] -= powerSpectrumFiducial(kn, zm);
    }
    
    fclose(tmp_fit_file);
    remove(tmp_fit_fname);
}

void OneDQuadraticPowerEstimate::_loadBalancing(std::vector<qso_computation_time*> *queue_qso, int maxthreads)
{
    LOG::LOGGER.STD("Load balancing for %d threads available.\n", maxthreads);
    double load_balance_time = mytime::getTime();

    double *bucket_time = new double[maxthreads]();

    for (int q = NUMBER_OF_QSOS-1; q >= NUMBER_OF_QSOS_OUT; q--)
    {
        // printf("QSO %d. EST CPU %e.\n", q, qso_estimators[q].est_cpu_time);
        // find min time bucket
        int min_ind = index_of_min_element(bucket_time, maxthreads);

        // add max time consuming to that bucket
        queue_qso[min_ind].push_back(&qso_estimators[q]);
        bucket_time[min_ind] += qso_estimators[q].est_cpu_time;
    }

    LOG::LOGGER.STD("Balanced estimated cpu times: ");
    for (int thr = 0; thr < maxthreads; ++thr)  LOG::LOGGER.STD("%.1e ", bucket_time[thr]);
    LOG::LOGGER.STD("\n");

    delete [] bucket_time;
    load_balance_time = mytime::getTime() - load_balance_time;
    
    LOG::LOGGER.STD("Load balancing took %.2f min.\n", load_balance_time);
}

void OneDQuadraticPowerEstimate::iterate(int number_of_iterations, const char *fname_base)
{
    char buf[500];
    double total_time = 0, total_time_1it = 0;

    std::vector<qso_computation_time*> *queue_qso = new std::vector<qso_computation_time*>[numthreads];
    _loadBalancing(queue_qso, numthreads);

    double *powerspectra_fits = new double[bins::TOTAL_KZ_BINS]();

    for (int i = 0; i < number_of_iterations; i++)
    {
        LOG::LOGGER.STD("Iteration number %d of %d.\n", i+1, number_of_iterations);
        
        total_time_1it = mytime::getTime();
    
        // Set total Fisher matrix and omn before F to zero for all k, z bins
        initializeIteration();

#pragma omp parallel
{
        double thread_time = mytime::getTime();

        gsl_vector *local_pmn_before_fisher_estimate_vs   = gsl_vector_calloc(bins::TOTAL_KZ_BINS);
        gsl_matrix *local_fisher_ms                       = gsl_matrix_calloc(bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);

        std::vector<qso_computation_time*> *local_que     = &queue_qso[t_rank];

        LOG::LOGGER.STD("Start working in %d/%d thread with %lu qso in queue.\n", t_rank, numthreads, local_que->size());

        for (std::vector<qso_computation_time*>::iterator it = local_que->begin(); it != local_que->end(); ++it)
            (*it)->qso->oneQSOiteration(powerspectra_fits, local_pmn_before_fisher_estimate_vs, local_fisher_ms);

        thread_time = mytime::getTime() - thread_time;

        LOG::LOGGER.STD("Done for loop in %d/%d thread in %.2f minutes. Adding F and d critically.\n",
                t_rank, numthreads, thread_time);

        #pragma omp critical
        {
            gsl_matrix_add(fisher_matrix_sum, local_fisher_ms);
            gsl_vector_add(pmn_before_fisher_estimate_vector_sum, local_pmn_before_fisher_estimate_vs);
        } 
        
        gsl_vector_free(local_pmn_before_fisher_estimate_vs);
        gsl_matrix_free(local_fisher_ms);
}

        try
        {
            invertTotalFisherMatrix();
            computePowerSpectrumEstimates();

            _fitPowerSpectra(powerspectra_fits);
        }
        catch (const char* msg)
        {
            LOG::LOGGER.ERR("ERROR %s: Fisher matrix is not invertable.\n", msg);
            throw msg;
        }
        
        printfSpectra();

        sprintf(buf, "%s_it%d_quadratic_power_estimate.dat", fname_base, i+1);
        writeSpectrumEstimates(buf);

        sprintf(buf, "%s_it%d_fisher_matrix.dat", fname_base, i+1);
        writeFisherMatrix(buf);

        total_time_1it  = mytime::getTime() - total_time_1it;
        total_time     += total_time_1it;
        LOG::LOGGER.STD("This iteration took %.1f minutes. Elapsed time so far is %.1f minutes.\n", 
            total_time_1it, total_time);
        mytime::printfTimeSpentDetails();

        if (hasConverged())
        {
            LOG::LOGGER.STD("Iteration has converged in %d iterations.\n", i+1);
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
    bool bool_converged = true;
    int kn, zm;

    for (int i_kz = 0; i_kz < bins::TOTAL_KZ_BINS; ++i_kz)
    {
        SKIP_LAST_K_BIN_WHEN_ENABLED(i_kz)

        bins::getFisherMatrixBinNoFromIndex(i_kz, kn, zm);   
        
        if (Z_BIN_COUNTS[zm+1] == 0)  continue;

        p1 = fabs(gsl_vector_get(pmn_estimate_vector, i_kz));
        p2 = fabs(gsl_vector_get(previous_pmn_estimate_vector, i_kz));
        
        diff = fabs(p1 - p2);
        pMax = std::max(p1, p2);
        r    = diff / pMax;

        if (r > CONVERGENCE_EPS)    bool_converged = false;

        abs_mean += r / (bins::TOTAL_KZ_BINS - bins::NUMBER_OF_Z_BINS);
        abs_max   = std::max(r, abs_max);
    }

    LOG::LOGGER.STD("Mean relative change is %.1e.\n"
                    "Maximum relative change is %.1e. "
                    "Old test: Iteration converges when this is less than %.1e\n", 
                    abs_mean, abs_max, CONVERGENCE_EPS);
    
    // Perform a chi-square test as well
    // Maybe just do diagonal (dx)^2/F-1_ii
    
    gsl_vector_sub(previous_pmn_estimate_vector, pmn_estimate_vector);

    r = 0;

    for (int i_kz = 0; i_kz < bins::TOTAL_KZ_BINS; ++i_kz)
    {
        SKIP_LAST_K_BIN_WHEN_ENABLED(i_kz)
        
        double  t = gsl_vector_get(previous_pmn_estimate_vector, i_kz),
                e = gsl_matrix_get(inverse_fisher_matrix_sum, i_kz, i_kz);

        r += (t*t) / e / (bins::TOTAL_KZ_BINS - bins::NUMBER_OF_Z_BINS);
    }

    r = sqrt(r);

    // r = my_cblas_dsymvdot(previous_pmn_estimate_vector, fisher_matrix_sum) / bins::TOTAL_KZ_BINS;

    LOG::LOGGER.STD("Chi square convergence test: %.3f per dof. "
                    "Iteration converges when this is less than %.2f\n", 
                    r, CHISQ_CONVERGENCE_EPS);

    bool_converged = r < CHISQ_CONVERGENCE_EPS;

    return bool_converged;
}

void OneDQuadraticPowerEstimate::writeFisherMatrix(const char *fname)
{
    mxhelp::fprintfMatrix(fname, fisher_matrix_sum);

    LOG::LOGGER.IO("Fisher matrix saved as %s.\n", fname);
}

void OneDQuadraticPowerEstimate::writeSpectrumEstimates(const char *fname)
{
    FILE *toWrite;
    int i_kz, kn, zm;
    double z, k, p, e;

    toWrite = ioh::open_file(fname, "w");
    
    #ifdef LAST_K_EDGE
    fprintf(toWrite, "%d %d\n", bins::NUMBER_OF_Z_BINS, bins::NUMBER_OF_K_BANDS-1);
    #else
    fprintf(toWrite, "%d %d\n", bins::NUMBER_OF_Z_BINS, bins::NUMBER_OF_K_BANDS);
    #endif

    for (zm = 0; zm <= bins::NUMBER_OF_Z_BINS+1; ++zm)
        fprintf(toWrite, "%d ", Z_BIN_COUNTS[zm]);

    fprintf(toWrite, "\n");

    for (i_kz = 0; i_kz < bins::TOTAL_KZ_BINS; ++i_kz)
    {
        SKIP_LAST_K_BIN_WHEN_ENABLED(i_kz)

        bins::getFisherMatrixBinNoFromIndex(i_kz, kn, zm);   
        
        if (Z_BIN_COUNTS[zm+1] == 0)  continue;

        z = bins::ZBIN_CENTERS[zm];
        k = bins::KBAND_CENTERS[kn];
        p = gsl_vector_get(pmn_estimate_vector, i_kz) + powerSpectrumFiducial(kn, zm);
        e = sqrt(gsl_matrix_get(inverse_fisher_matrix_sum, i_kz, i_kz));

        fprintf(toWrite, "%.3lf %e %e %e\n", z, k, p, e);
    }

    fclose(toWrite);
        
    LOG::LOGGER.IO("Quadratic 1D Power Spectrum estimate saved as %s.\n", fname);
    LOG::LOGGER.STD("Quadratic 1D Power Spectrum estimate saved as %s.\n", fname);
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

    for (int zm = 1; zm <= bins::NUMBER_OF_Z_BINS; ++zm)
    {
        if (Z_BIN_COUNTS[zm] == 0)  continue;

        LOG::LOGGER.STD(" P(%.1f, k) |", bins::ZBIN_CENTERS[zm-1]);
    }
    LOG::LOGGER.STD("\n");
    
    for (int kn = 0; kn < bins::NUMBER_OF_K_BANDS; ++kn)
    {
        for (int zm = 1; zm <= bins::NUMBER_OF_Z_BINS; ++zm)
        {
            if (Z_BIN_COUNTS[zm] == 0)  continue;

            i_kz = bins::getFisherMatrixIndex(kn, zm-1);

            LOG::LOGGER.STD(" %.3e |", pmn_estimate_vector->data[i_kz] + powerSpectrumFiducial(kn, zm-1));
        }
        LOG::LOGGER.STD("\n");
    }
}

double OneDQuadraticPowerEstimate::powerSpectrumFiducial(int kn, int zm)
{
    if (TURN_OFF_SFID)  return 0;
    
    return fidcosmo::fiducialPowerSpectrum(bins::KBAND_CENTERS[kn], bins::ZBIN_CENTERS[zm], &pd13::FIDUCIAL_PD13_PARAMS);
}





















