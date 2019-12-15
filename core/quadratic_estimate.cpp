#include "core/quadratic_estimate.hpp"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib> // system
#include <cassert>
#include <stdexcept>
#include <string>
#include <sstream>      // std::ostringstream

#include <gsl/gsl_cblas.h>

#if defined(ENABLE_MPI)
#include "mpi.h" 
#endif

#include "core/matrix_helper.hpp"
#include "core/global_numbers.hpp"
#include "core/fiducial_cosmology.hpp"
#include "io/io_helper_functions.hpp"
#include "io/logger.hpp"

//-------------------------------------------------------

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
    Z_BIN_COUNTS = new int[bins::NUMBER_OF_Z_BINS+2]();

    // Allocate memory
    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
    {
        dbt_estimate_sum_before_fisher_vector[dbt_i] = gsl_vector_alloc(bins::TOTAL_KZ_BINS);
        dbt_estimate_fisher_weighted_vector[dbt_i]   = gsl_vector_alloc(bins::TOTAL_KZ_BINS);
    }

    previous_power_estimate_vector          = gsl_vector_alloc(bins::TOTAL_KZ_BINS);
    current_power_estimate_vector           = gsl_vector_calloc(bins::TOTAL_KZ_BINS);
    fisher_matrix_sum                       = gsl_matrix_alloc(bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);
    inverse_fisher_matrix_sum               = gsl_matrix_alloc(bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);

    isFisherInverted = false; 

    _readQSOFiles(fname_list, dir);
}

void OneDQuadraticPowerEstimate::_readQSOFiles(const char *fname_list, const char *dir)
{
    LOG::LOGGER.STD("Initial reading of quasar spectra and estimating CPU time.\n");

    std::vector<std::string> fpaths;
    NUMBER_OF_QSOS = ioh::readList(fname_list, fpaths);
    
    std::rotate(fpaths.begin(), fpaths.begin()+NUMBER_OF_QSOS*process::this_pe/process::total_pes, fpaths.end());

    // Create objects for each QSO
    cpu_fname_vector.reserve(NUMBER_OF_QSOS);
    double cpu_t_temp;

    for (std::vector<std::string>::iterator fq = fpaths.begin(); fq != fpaths.end(); ++fq)
    {
        fq->insert(0, "/");
        fq->insert(0, dir);

        OneQSOEstimate q_temp(*fq);
        cpu_t_temp = q_temp.getComputeTimeEst();
        
        ++Z_BIN_COUNTS[q_temp.ZBIN + 1];

        if (cpu_t_temp != 0)
            cpu_fname_vector.push_back(std::make_pair(cpu_t_temp, *fq));
    }
    
    NUMBER_OF_QSOS_OUT = Z_BIN_COUNTS[0] + Z_BIN_COUNTS[bins::NUMBER_OF_Z_BINS+1];

    LOG::LOGGER.STD("Z bin counts: ");
    for (int zm = 0; zm < bins::NUMBER_OF_Z_BINS+2; zm++)
        LOG::LOGGER.STD("%d ", Z_BIN_COUNTS[zm]);
    LOG::LOGGER.STD("\nNumber of quasars: %d\nQSOs in z bins: %d\n", NUMBER_OF_QSOS, NUMBER_OF_QSOS - NUMBER_OF_QSOS_OUT);

    LOG::LOGGER.STD("Sorting with respect to estimated cpu time.\n");
    std::sort(cpu_fname_vector.begin(), cpu_fname_vector.end()); // Ascending order
}

OneDQuadraticPowerEstimate::~OneDQuadraticPowerEstimate()
{
    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
    {
        gsl_vector_free(dbt_estimate_sum_before_fisher_vector[dbt_i]);
        gsl_vector_free(dbt_estimate_fisher_weighted_vector[dbt_i]);
    }

    gsl_vector_free(previous_power_estimate_vector);
    gsl_vector_free(current_power_estimate_vector);

    gsl_matrix_free(fisher_matrix_sum);
    gsl_matrix_free(inverse_fisher_matrix_sum);

    // std::vector<std::pair <double, OneQSOEstimate*>>::iterator qe = qso_estimators.begin();
    // for (; qe != qso_estimators.end(); ++qe)
    //     delete qe->second;
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

    gsl_vector_memcpy(previous_power_estimate_vector, current_power_estimate_vector);

    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
    {
        cblas_dsymv(CblasRowMajor, CblasUpper,
                    bins::TOTAL_KZ_BINS, 0.5, inverse_fisher_matrix_sum->data, bins::TOTAL_KZ_BINS,
                    dbt_estimate_sum_before_fisher_vector[dbt_i]->data, 1,
                    0, dbt_estimate_fisher_weighted_vector[dbt_i]->data, 1);
    }

    gsl_vector_memcpy(current_power_estimate_vector, dbt_estimate_fisher_weighted_vector[0]);
    gsl_vector_sub(current_power_estimate_vector, dbt_estimate_fisher_weighted_vector[1]);
    gsl_vector_sub(current_power_estimate_vector, dbt_estimate_fisher_weighted_vector[2]);
}

// Note that fitting is done on bin averaged values plus fiducial power
void OneDQuadraticPowerEstimate::_fitPowerSpectra(double *fit_values)
{
    char tmp_ps_fname[320], tmp_fit_fname[320];
    std::ostringstream command;

    FILE *tmp_fit_file;
    int s1, s2, kn, zm, fr;
    static fidpd13::pd13_fit_params iteration_fits = fidpd13::FIDUCIAL_PD13_PARAMS;

    sprintf(tmp_ps_fname, "%s/tmppsfileXXXXXX", TMP_FOLDER);
    sprintf(tmp_fit_fname, "%s/tmpfitfileXXXXXX", TMP_FOLDER);

    s1 = mkstemp(tmp_ps_fname);
    s2 = mkstemp(tmp_fit_fname);

    if (s1 == -1 || s2 == -1)
    {
        LOG::LOGGER.ERR("ERROR: Temp filename cannot be generated!\n");
        throw std::runtime_error("tmp filename");
    }

    writeSpectrumEstimates(tmp_ps_fname);

    command << "lorentzian_fit.py " << tmp_ps_fname << " " << tmp_fit_fname << " "
            << iteration_fits.A << " " << iteration_fits.n << " " << iteration_fits.n << " ";

    // Do not pass redshift parameters is there is only one redshift bin
    if (bins::NUMBER_OF_Z_BINS > 1)  command << iteration_fits.B << " " << iteration_fits.beta << " ";
    
    command << iteration_fits.lambda;
    
    if (process::this_pe == 0) 
    {
        command
            << " >> " << LOG::LOGGER.getFileName(LOG::TYPE::STD);
        LOG::LOGGER.STD("%s\n", command.str().c_str());
    }

    LOG::LOGGER.close();

    // Print from python does not go into LOG::LOGGER
    s1 = system(command.str().c_str());
    
    LOG::LOGGER.reopen();
    remove(tmp_ps_fname);

    if (s1 != 0)
    {
        LOG::LOGGER.ERR("Error in fitting.\n");  
        throw std::runtime_error("fitting error");
    }

    tmp_fit_file = ioh::open_file(tmp_fit_fname, "r");

    fr = fscanf(tmp_fit_file, "%le %le %le %le %le %le\n",
                &iteration_fits.A, &iteration_fits.n, &iteration_fits.alpha,
                &iteration_fits.B, &iteration_fits.beta, &iteration_fits.lambda);

    if (fr != 6)
        throw std::runtime_error("Reading fit parameters from tmp_fit_file!");

    for (int i_kz = 0; i_kz < bins::TOTAL_KZ_BINS; i_kz++)
    {
        SKIP_LAST_K_BIN_WHEN_ENABLED(i_kz)

        bins::getFisherMatrixBinNoFromIndex(i_kz, kn, zm);   
        
        fr = fscanf(tmp_fit_file, "%le\n", &fit_values[i_kz]);

        if (fr != 1)
            throw std::runtime_error("Reading fit power values from tmp_fit_file!");

        fit_values[i_kz] -= powerSpectrumFiducial(kn, zm);
    }
    
    fclose(tmp_fit_file);
    remove(tmp_fit_fname);
}

void OneDQuadraticPowerEstimate::_loadBalancing(std::vector<OneQSOEstimate*> &local_queue)
{
    LOG::LOGGER.STD("Load balancing for %d threads available.\n", process::total_pes);
    
    double load_balance_time = mytime::getTime();

    double *bucket_time = new double[process::total_pes]();

    std::vector<std::pair <double, std::string>>::reverse_iterator qe = cpu_fname_vector.rbegin();
    for (; qe != cpu_fname_vector.rend(); ++qe)
    {
        // find min time bucket
        int min_ind = index_of_min_element(bucket_time, process::total_pes);
        // add max time consuming to that bucket
        bucket_time[min_ind] += qe->first;

        if (min_ind == process::this_pe)
        {
            // Construct and add queue
            OneQSOEstimate *q_temp = new OneQSOEstimate(qe->second);
            local_queue.push_back(q_temp);
        }
    }

    LOG::LOGGER.STD("Balanced estimated cpu times: ");
    for (int thr = 0; thr < process::total_pes; ++thr)
        LOG::LOGGER.STD("%.1e ", bucket_time[thr]);
    LOG::LOGGER.STD("\n");

    delete [] bucket_time;
    load_balance_time = mytime::getTime() - load_balance_time;
    
    LOG::LOGGER.STD("Load balancing took %.2f sec.\n", load_balance_time*60.);
}

void OneDQuadraticPowerEstimate::iterate(int number_of_iterations, const char *fname_base)
{
    char buf[500];
    double total_time = 0, total_time_1it = 0;
    double *powerspectra_fits = new double[bins::TOTAL_KZ_BINS]();

    std::vector<OneQSOEstimate*> local_queue;
    _loadBalancing(local_queue);

    LOG::LOGGER.TIME("| %2s | %9s | %9s | %9s | %9s | %9s | %9s | %9s | %9s | %9s | %9s | %9s | %9s |\n", 
        "i", "T_i", "T_tot", "T_Cinv", "T_Finv", "T_Sfid", "N_Sfid", "T_Q", "N_Q", "T_Qmod", "T_F", "DChi2", "DMean");

    for (int i = 0; i < number_of_iterations; i++)
    {
        LOG::LOGGER.STD("Iteration number %d of %d.\n", i+1, number_of_iterations);
        
        total_time_1it = mytime::getTime();
    
        // Set total Fisher matrix and omn before F to zero for all k, z bins
        initializeIteration();

        double thread_time = mytime::getTime();

        LOG::LOGGER.STD("Start working in %d/%d thread with %lu qso in queue.\n", process::this_pe, process::total_pes, local_queue.size());

        for (std::vector<OneQSOEstimate*>::iterator it = local_queue.begin(); it != local_queue.end(); ++it)
            (*it)->oneQSOiteration(powerspectra_fits, dbt_estimate_sum_before_fisher_vector, fisher_matrix_sum);

        thread_time = mytime::getTime() - thread_time;

        LOG::LOGGER.STD("Done for loop in %d/%d thread in %.2f minutes. Adding F and d critically.\n",
                process::this_pe, process::total_pes, thread_time);

        #if defined(ENABLE_MPI)
        // allreduce
        MPI_Allreduce(MPI_IN_PLACE, fisher_matrix_sum->data, bins::TOTAL_KZ_BINS*bins::TOTAL_KZ_BINS,
            MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
            MPI_Allreduce(MPI_IN_PLACE, dbt_estimate_sum_before_fisher_vector[dbt_i]->data, bins::TOTAL_KZ_BINS,
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        #endif
        
        try
        {
            invertTotalFisherMatrix();
            computePowerSpectrumEstimates();

            _fitPowerSpectra(powerspectra_fits);
        }
        catch (const char* msg)
        {
            LOG::LOGGER.ERR("ERROR %s: Fisher matrix is not invertable.\n", msg);
            for (std::vector<OneQSOEstimate*>::iterator it = local_queue.begin(); it != local_queue.end(); ++it)
                delete *it;
            
            throw std::runtime_error(msg);
        }
        
        if (process::this_pe == 0)
        {
            printfSpectra();

            sprintf(buf, "%s_it%d_quadratic_power_estimate.dat", fname_base, i+1);
            writeSpectrumEstimates(buf);

            sprintf(buf, "%s_it%d_quadratic_power_estimate_detailed.dat", fname_base, i+1);
            writeDetailedSpectrumEstimates(buf);

            sprintf(buf, "%s_it%d_fisher_matrix.dat", fname_base, i+1);
            writeFisherMatrix(buf);

            total_time_1it  = mytime::getTime() - total_time_1it;
            total_time     += total_time_1it;
            LOG::LOGGER.STD("This iteration took %.1f minutes. Elapsed time so far is %.1f minutes.\n", 
                total_time_1it, total_time);
            LOG::LOGGER.TIME("| %2d | %9.3e | %9.3e | ", i, total_time_1it, total_time);

            mytime::printfTimeSpentDetails();
        }

        if (hasConverged())
        {
            LOG::LOGGER.STD("Iteration has converged in %d iterations.\n", i+1);
            break;
        }
    }

    for (std::vector<OneQSOEstimate*>::iterator it = local_queue.begin(); it != local_queue.end(); ++it)
        delete *it;

    delete [] powerspectra_fits;
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
        
        // if (Z_BIN_COUNTS[zm+1] == 0)  continue;

        p1 = fabs(gsl_vector_get(current_power_estimate_vector, i_kz));
        p2 = fabs(gsl_vector_get(previous_power_estimate_vector, i_kz));
        
        diff = fabs(p1 - p2);
        pMax = std::max(p1, p2);
        r    = diff / pMax;

        if (r > CONVERGENCE_EPS)    bool_converged = false;

        abs_mean += r / bins::DEGREE_OF_FREEDOM;
        abs_max   = std::max(r, abs_max);
    }
    
    LOG::LOGGER.STD("Mean relative change is %.1e.\n"
                    "Maximum relative change is %.1e. "
                    "Old test: Iteration converges when this is less than %.1e\n", 
                    abs_mean, abs_max, CONVERGENCE_EPS);
    
    // Perform a chi-square test as well
    // Maybe just do diagonal (dx)^2/F-1_ii
    
    gsl_vector_sub(previous_power_estimate_vector, current_power_estimate_vector);

    r = 0;

    for (int i_kz = 0; i_kz < bins::TOTAL_KZ_BINS; ++i_kz)
    {
        SKIP_LAST_K_BIN_WHEN_ENABLED(i_kz)
        
        double  t = gsl_vector_get(previous_power_estimate_vector, i_kz),
                e = gsl_matrix_get(inverse_fisher_matrix_sum, i_kz, i_kz);

        if (e < 0)  continue;

        r += (t*t) / e;
    }

    r  = sqrt(r / bins::DEGREE_OF_FREEDOM);

    double rfull = mxhelp::my_cblas_dsymvdot(previous_power_estimate_vector, fisher_matrix_sum) / bins::DEGREE_OF_FREEDOM;
    
    LOG::LOGGER.TIME("%9.3e | %9.3e |\n", r, abs_mean);
    LOG::LOGGER.STD("Chi square convergence test: Diagonal Err: %.3f per dof. Full Fisher: %.3f per dof."
                    "Iteration converges when either is less than %.2f\n", 
                    r, rfull, CHISQ_CONVERGENCE_EPS);

    bool_converged = r < CHISQ_CONVERGENCE_EPS || fabs(rfull) < CHISQ_CONVERGENCE_EPS;

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
        
        // if (Z_BIN_COUNTS[zm+1] == 0)  continue;

        z = bins::ZBIN_CENTERS[zm];
        k = bins::KBAND_CENTERS[kn];
        p = gsl_vector_get(current_power_estimate_vector, i_kz) + powerSpectrumFiducial(kn, zm);
        e = sqrt(gsl_matrix_get(inverse_fisher_matrix_sum, i_kz, i_kz));

        fprintf(toWrite, "%.3lf %e %e %e\n", z, k, p, e);
    }

    fclose(toWrite);
    
    LOG::LOGGER.IO("Quadratic 1D Power Spectrum estimate saved as %s.\n", fname);
    LOG::LOGGER.STD("Quadratic 1D Power Spectrum estimate saved as %s.\n", fname);
}

void OneDQuadraticPowerEstimate::writeDetailedSpectrumEstimates(const char *fname)
{
    FILE *toWrite;
    int i_kz, kn, zm;
    double z, k1, k2, kc, Pfid, ThetaP, ErrorP, dk, bk, tk;

    toWrite = ioh::open_file(fname, "w");
    
    fprintf(toWrite, specifics::BUILD_SPECIFICS);
    specifics::printConfigSpecifics(toWrite);
    
    fprintf(toWrite, "# Fiducial Power Spectrum\n"
                     "# Pfid(k, z) = (A*pi/k0) * q^(2+n+alpha*ln(q)+beta*ln(x)) * x^B / (1 + lambda * k^2)\n"
                     "# k0=0.009 s km^-1, z0=3.0 and q=k/k0, x=(1+z)/(1+z0)\n"
                     "# Parameters set by config file:\n");
    fprintf(toWrite, "# A      = %15e\n"
                     "# n      = %15e\n"
                     "# alpha  = %15e\n"
                     "# B      = %15e\n"
                     "# beta   = %15e\n"
                     "# lambda = %15e\n", 
                     fidpd13::FIDUCIAL_PD13_PARAMS.A, fidpd13::FIDUCIAL_PD13_PARAMS.n, fidpd13::FIDUCIAL_PD13_PARAMS.alpha,
                     fidpd13::FIDUCIAL_PD13_PARAMS.B, fidpd13::FIDUCIAL_PD13_PARAMS.beta, fidpd13::FIDUCIAL_PD13_PARAMS.lambda);
    fprintf(toWrite, "# -----------------------------------------------------------------\n"
                     "# File Template\n"
                     "# Nz Nk\n"
                     "# n[0] n[1] ... n[Nz] n[Nz+1]\n"
                     "# z | k1 | k2 | kc | Pfid | ThetaP | Pest | ErrorP | d | b | t\n"
                     "# Nz                Number of redshift bins\n"
                     "# Nk                Number of k bins\n"
                     "# n[i]              Spectral chunk count in redshift bin i. Left-most and right-most are out of range\n"
                     "# z                 Redshift bin center\n"
                     "# k1                Lower edge of the k bin [s km^-1]\n"
                     "# k2                Upper edge of the k bin [s km^-1]\n"
                     "# kc                Center of the k bin [s km^-1]\n"
                     "# Pfid              Fiducial power at kc [km s^-1]\n"
                     "# ThetaP            Deviation from Pfid found by quadratic estimator = d - b - t [km s^-1]\n"
                     "# Pest              Pfid + ThetaP [km s^-1]\n"
                     "# ErrorP            Error estimated from diagonal terms of the inverse Fisher matrix [km s^-1]\n"
                     "# d                 Power estimate before noise (b) and fiducial power (t) subtracted [km s^-1]\n"
                     "# b                 Noise estimate [km s^-1]\n"
                     "# t                 Fiducial power estimate [km s^-1]\n"
                     "# -----------------------------------------------------------------\n");

    #ifdef LAST_K_EDGE
    fprintf(toWrite, "# %d %d\n# ", bins::NUMBER_OF_Z_BINS, bins::NUMBER_OF_K_BANDS-1);
    #else
    fprintf(toWrite, "# %d %d\n# ", bins::NUMBER_OF_Z_BINS, bins::NUMBER_OF_K_BANDS);
    #endif

    for (zm = 0; zm <= bins::NUMBER_OF_Z_BINS+1; ++zm)
        fprintf(toWrite, "%d ", Z_BIN_COUNTS[zm]);

    fprintf(toWrite, "\n");
    fprintf(toWrite, "z %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s\n", 
        "k1", "k2", "kc", "Pfid", "ThetaP", "Pest", "ErrorP", "d", "b", "t");

    for (i_kz = 0; i_kz < bins::TOTAL_KZ_BINS; ++i_kz)
    {
        SKIP_LAST_K_BIN_WHEN_ENABLED(i_kz)

        bins::getFisherMatrixBinNoFromIndex(i_kz, kn, zm);   
        
        // if (Z_BIN_COUNTS[zm+1] == 0)  continue;

        z  = bins::ZBIN_CENTERS[zm];
        
        k1 = bins::KBAND_EDGES[kn];
        k2 = bins::KBAND_EDGES[kn+1];
        kc = bins::KBAND_CENTERS[kn];

        Pfid = powerSpectrumFiducial(kn, zm);
        ThetaP = gsl_vector_get(current_power_estimate_vector, i_kz);
        ErrorP = sqrt(gsl_matrix_get(inverse_fisher_matrix_sum, i_kz, i_kz));

        dk = gsl_vector_get(dbt_estimate_fisher_weighted_vector[0], i_kz);
        bk = gsl_vector_get(dbt_estimate_fisher_weighted_vector[1], i_kz);
        tk = gsl_vector_get(dbt_estimate_fisher_weighted_vector[2], i_kz);

        fprintf(toWrite, "%.3lf %15e %15e %15e %15e %15e %15e %15e %15e %15e %15e\n", 
                            z,  k1,  k2,  kc,  Pfid,ThetaP, Pfid+ThetaP, ErrorP, dk, bk, tk);
    }

    fclose(toWrite);
        
    LOG::LOGGER.IO("Quadratic 1D Power Spectrum estimate saved as %s.\n", fname);
    LOG::LOGGER.STD("Quadratic 1D Power Spectrum estimate saved as %s.\n", fname);
}

void OneDQuadraticPowerEstimate::initializeIteration()
{
    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
        gsl_vector_set_zero(dbt_estimate_sum_before_fisher_vector[dbt_i]);
           
    gsl_matrix_set_zero(fisher_matrix_sum);
    isFisherInverted = false;
}

void OneDQuadraticPowerEstimate::printfSpectra()
{
    int i_kz;

    for (int zm = 1; zm <= bins::NUMBER_OF_Z_BINS; ++zm)
    {
        // if (Z_BIN_COUNTS[zm] == 0)  continue;

        LOG::LOGGER.STD(" P(%.1f, k) |", bins::ZBIN_CENTERS[zm-1]);
    }
    LOG::LOGGER.STD("\n");
    
    for (int kn = 0; kn < bins::NUMBER_OF_K_BANDS; ++kn)
    {
        for (int zm = 1; zm <= bins::NUMBER_OF_Z_BINS; ++zm)
        {
            // if (Z_BIN_COUNTS[zm] == 0)  continue;

            i_kz = bins::getFisherMatrixIndex(kn, zm-1);

            LOG::LOGGER.STD(" %.3e |", current_power_estimate_vector->data[i_kz] + powerSpectrumFiducial(kn, zm-1));
        }
        LOG::LOGGER.STD("\n");
    }
}

double OneDQuadraticPowerEstimate::powerSpectrumFiducial(int kn, int zm)
{
    if (TURN_OFF_SFID)  return 0;
    
    return fidcosmo::fiducialPowerSpectrum(bins::KBAND_CENTERS[kn], bins::ZBIN_CENTERS[zm], &fidpd13::FIDUCIAL_PD13_PARAMS);
}





















