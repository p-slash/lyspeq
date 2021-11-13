#include "core/quadratic_estimate.hpp"

#include <cmath>
#include <algorithm> // std::for_each
#include <numeric> // std::accumulate
#include <memory> // std::default_delete
#include <cstdio>
#include <cstdlib> // system
#include <stdexcept>
#include <string>
#include <sstream>      // std::ostringstream

#include "core/one_qso_estimate.hpp"
#include "core/matrix_helper.hpp"
#include "core/global_numbers.hpp"
#include "core/fiducial_cosmology.hpp"
#include "io/io_helper_functions.hpp"
#include "io/logger.hpp"
#include "io/qso_file.hpp"

#if defined(ENABLE_MPI)
#include "mpi.h" 
#include "core/mpi_merge_sort.cpp"
#endif

double *OneDQuadraticPowerEstimate::precomputed_fisher { NULL };

OneDQuadraticPowerEstimate::OneDQuadraticPowerEstimate(const char *fname_list, const char *dir)
{
    Z_BIN_COUNTS = new int[bins::NUMBER_OF_Z_BINS+2]();

    // Allocate memory
    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
    {
        dbt_estimate_sum_before_fisher_vector[dbt_i] = new double[bins::TOTAL_KZ_BINS];
        dbt_estimate_fisher_weighted_vector[dbt_i]   = new double[bins::TOTAL_KZ_BINS];
    }

    previous_power_estimate_vector = new double[bins::TOTAL_KZ_BINS];
    current_power_estimate_vector  = new double[bins::TOTAL_KZ_BINS]();
    fisher_matrix_sum              = new double[FISHER_SIZE];
    inverse_fisher_matrix_sum      = new double[FISHER_SIZE];

    powerspectra_fits              = new double[bins::TOTAL_KZ_BINS]();

    isFisherInverted = false; 

    _readQSOFiles(fname_list, dir);
}

void OneDQuadraticPowerEstimate::_readQSOFiles(const char *fname_list, const char *dir)
{
    double t1, t2;
    std::vector<std::string> filepaths;
    std::vector< std::pair<double, int> > cpu_fname_vector;

    LOG::LOGGER.STD("Initial reading of quasar spectra and estimating CPU time.\n");

    NUMBER_OF_QSOS = ioh::readList(fname_list, filepaths);
    // Add parent directory to file path
    for (auto fq = filepaths.begin(); fq != filepaths.end(); ++fq)
    {
        fq->insert(0, "/");
        fq->insert(0, dir);
    }

    // Each PE reads a different section of files
    // They sort individually, then merge in pairs
    // Finally, the master PE broadcasts the sorted full array
    int delta_nfiles = NUMBER_OF_QSOS / process::total_pes,
        fstart_this  = delta_nfiles * process::this_pe,
        fend_this    = delta_nfiles * (process::this_pe + 1);
    
    if (process::this_pe == process::total_pes - 1)
        fend_this = NUMBER_OF_QSOS;

    t2 = mytime::timer.getTime();

    // Create vector for QSOs & read
    cpu_fname_vector.reserve(NUMBER_OF_QSOS);

    for (int findex = fstart_this; findex < fend_this; ++findex)
    {
        int zbin;
        double cpu_t_temp = OneQSOEstimate::getComputeTimeEst(filepaths[findex], 
            zbin);

        ++Z_BIN_COUNTS[zbin+1];

        if (cpu_t_temp != 0)
            cpu_fname_vector.push_back(std::make_pair(cpu_t_temp, findex));
    }

    if (specifics::INPUT_QSO_FILE == qio::Picca)
        qio::PiccaFile::clearCache();

    // Print out time it took to read all files into vector
    t1 = mytime::timer.getTime();
    LOG::LOGGER.STD("Reading QSO files took %.2f m.\n", t1-t2);

    // MPI Reduce ZBIN_COUNTS
    #if defined(ENABLE_MPI)
        MPI_Allreduce(MPI_IN_PLACE, Z_BIN_COUNTS, bins::NUMBER_OF_Z_BINS+2, 
            MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    #endif

    NUMBER_OF_QSOS_OUT = Z_BIN_COUNTS[0] + Z_BIN_COUNTS[bins::NUMBER_OF_Z_BINS+1];

    LOG::LOGGER.STD("Z bin counts: ");
    for (int zm = 0; zm < bins::NUMBER_OF_Z_BINS+2; zm++)
        LOG::LOGGER.STD("%d ", Z_BIN_COUNTS[zm]);
    LOG::LOGGER.STD("\nNumber of quasars: %d\nQSOs in z bins: %d\n", 
        NUMBER_OF_QSOS, NUMBER_OF_QSOS - NUMBER_OF_QSOS_OUT);

    LOG::LOGGER.STD("Sorting with respect to estimated cpu time.\n");
    std::sort(cpu_fname_vector.begin(), cpu_fname_vector.end()); // Ascending order

    #if defined(ENABLE_MPI)
    mpisort::mergeSortedArrays(0, process::total_pes, process::this_pe, 
        cpu_fname_vector);
    mpisort::bcastCpuFnameVec(cpu_fname_vector);
    #endif

    // Print out time it took to sort files wrt CPU time
    t2 = mytime::timer.getTime();
    LOG::LOGGER.STD("Sorting took %.2f m.\n", t2-t1);

    if (cpu_fname_vector.empty())
        throw std::runtime_error("No spectrum in queue. Check files & redshift range.");

    _loadBalancing(filepaths, cpu_fname_vector);
}

void OneDQuadraticPowerEstimate::_loadBalancing(std::vector<std::string> &filepaths,
    std::vector< std::pair<double, int> > &cpu_fname_vector)
{
    LOG::LOGGER.STD("Load balancing for %d tasks available.\n", process::total_pes);
    
    double load_balance_time = mytime::timer.getTime();
    
    std::vector<double> bucket_time(process::total_pes, 0);

    local_fpaths.reserve(int(1.1*filepaths.size()/process::total_pes));

    std::vector<std::pair <double, int>>::reverse_iterator qe = cpu_fname_vector.rbegin();
    for (; qe != cpu_fname_vector.rend(); ++qe)
    {
        // find min time bucket
        // add max time consuming to that bucket
        // Construct and add queue
        auto min_bt = std::min_element(bucket_time.begin(), bucket_time.end());
        (*min_bt) += qe->first;

        if (std::distance(bucket_time.begin(), min_bt) == process::this_pe)
            local_fpaths.push_back(filepaths[qe->second]);
    }

    // sort local_fpaths
    if (specifics::INPUT_QSO_FILE == qio::Picca) 
        std::sort(local_fpaths.begin(), local_fpaths.end(), qio::PiccaFile::compareFnames);

    double ave_balance = std::accumulate(bucket_time.begin(), 
        bucket_time.end(), 0.) / process::total_pes;

    LOG::LOGGER.STD("Off-Balance: ");
    for (auto it = bucket_time.begin(); it != bucket_time.end(); ++it)
        LOG::LOGGER.STD("%.1e ", (*it)/ave_balance-1);
    LOG::LOGGER.STD("\n");

    load_balance_time = mytime::timer.getTime() - load_balance_time;

    LOG::LOGGER.STD("Load balancing took %.2f sec.\n", load_balance_time*60.);
}

OneDQuadraticPowerEstimate::~OneDQuadraticPowerEstimate()
{
    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
    {
        delete [] dbt_estimate_sum_before_fisher_vector[dbt_i];
        delete [] dbt_estimate_fisher_weighted_vector[dbt_i];
    }

    delete [] previous_power_estimate_vector;
    delete [] current_power_estimate_vector;
    delete [] powerspectra_fits;

    delete [] fisher_matrix_sum;
    delete [] inverse_fisher_matrix_sum;
}

void OneDQuadraticPowerEstimate::invertTotalFisherMatrix()
{
    double t = mytime::timer.getTime();

    LOG::LOGGER.STD("Inverting Fisher matrix.\n");
    
    std::copy(fisher_matrix_sum, fisher_matrix_sum+FISHER_SIZE, 
        inverse_fisher_matrix_sum);
    mxhelp::LAPACKE_InvertMatrixLU(inverse_fisher_matrix_sum, 
        bins::TOTAL_KZ_BINS);
    
    isFisherInverted = true;

    t = mytime::timer.getTime() - t;
    mytime::time_spent_on_f_inv += t;
}

void OneDQuadraticPowerEstimate::computePowerSpectrumEstimates()
{
    LOG::LOGGER.STD("Estimating power spectrum.\n");

    std::copy(current_power_estimate_vector, 
        current_power_estimate_vector + bins::TOTAL_KZ_BINS, 
        previous_power_estimate_vector);

    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
    {
        cblas_dsymv(CblasRowMajor, CblasUpper,bins::TOTAL_KZ_BINS, 0.5, 
            inverse_fisher_matrix_sum, bins::TOTAL_KZ_BINS,
            dbt_estimate_sum_before_fisher_vector[dbt_i], 1,
            0, dbt_estimate_fisher_weighted_vector[dbt_i], 1);
    }

    std::copy(dbt_estimate_fisher_weighted_vector[0], 
        dbt_estimate_fisher_weighted_vector[0] + bins::TOTAL_KZ_BINS, 
        current_power_estimate_vector);
    mxhelp::vector_sub(current_power_estimate_vector, 
        dbt_estimate_fisher_weighted_vector[1], 
        bins::TOTAL_KZ_BINS);
    mxhelp::vector_sub(current_power_estimate_vector, 
        dbt_estimate_fisher_weighted_vector[2], 
        bins::TOTAL_KZ_BINS);
}

void OneDQuadraticPowerEstimate::_readScriptOutput(double *script_power, 
    const char *fname, void *itsfits)
{
    int fr;
    FILE *tmp_fit_file = ioh::open_file(fname, "r");

    if (itsfits != NULL)
    {
        fidpd13::pd13_fit_params *iteration_fits = (fidpd13::pd13_fit_params *) itsfits;

        fr = fscanf(tmp_fit_file, "%le %le %le %le %le %le\n",
                &iteration_fits->A, &iteration_fits->n, &iteration_fits->alpha,
                &iteration_fits->B, &iteration_fits->beta, &iteration_fits->lambda);

        if (fr != 6)
            throw std::runtime_error("Reading fit parameters from tmp_fit_file!");
    }
    
    int kn, zm;
    for (int i_kz = 0; i_kz < bins::TOTAL_KZ_BINS; i_kz++)
    {
        bins::getFisherMatrixBinNoFromIndex(i_kz, kn, zm);   
        
        fr = fscanf(tmp_fit_file, "%le\n", &script_power[i_kz]);

        if (fr != 1)
            throw std::runtime_error("Reading fit power values from tmp_fit_file!");

        script_power[i_kz] -= powerSpectrumFiducial(kn, zm);
    }

    fclose(tmp_fit_file);
}


// Note that fitting is done deviations plus fiducial power
void OneDQuadraticPowerEstimate::_fitPowerSpectra(double *fitted_power)
{
    static fidpd13::pd13_fit_params iteration_fits = fidpd13::FIDUCIAL_PD13_PARAMS;
    
    char tmp_ps_fname[320], tmp_fit_fname[320];

    ioh::create_tmp_file(tmp_ps_fname, process::TMP_FOLDER);
    ioh::create_tmp_file(tmp_fit_fname, process::TMP_FOLDER);
    
    writeSpectrumEstimates(tmp_ps_fname);

    std::ostringstream command("lorentzian_fit.py ", std::ostringstream::ate);
    command << tmp_ps_fname << " " << tmp_fit_fname << " "
        << iteration_fits.A << " " << iteration_fits.n << " " 
        << iteration_fits.n << " ";

    // Do not pass redshift parameters if there is only one redshift bin
    if (bins::NUMBER_OF_Z_BINS > 1)
        command << iteration_fits.B << " " << iteration_fits.beta << " ";
    
    command << iteration_fits.lambda;
    
    // Sublime text acts weird when `<< " >> " <<` is explicitly typed
    #define GGSTR " >> "
    if (process::this_pe == 0) 
        command << GGSTR << LOG::LOGGER.getFileName(LOG::TYPE::STD);   
    #undef GGSTR

    LOG::LOGGER.STD("%s\n", command.str().c_str());
    LOG::LOGGER.close();

    // Print from python does not go into LOG::LOGGER
    int s1 = system(command.str().c_str());
    
    LOG::LOGGER.reopen();
    remove(tmp_ps_fname);

    if (s1 != 0)
    {
        LOG::LOGGER.ERR("Error in fitting.\n");  
        throw std::runtime_error("fitting error");
    }

    _readScriptOutput(fitted_power, tmp_fit_fname, &iteration_fits);
    remove(tmp_fit_fname);
}

void OneDQuadraticPowerEstimate::_smoothPowerSpectra(double *smoothed_power)
{
    char tmp_ps_fname[320], tmp_smooth_fname[320];
    
    ioh::create_tmp_file(tmp_ps_fname, process::TMP_FOLDER);
    ioh::create_tmp_file(tmp_smooth_fname, process::TMP_FOLDER);

    writeSpectrumEstimates(tmp_ps_fname);

    std::ostringstream command("smbivspline.py ", std::ostringstream::ate);
    command << tmp_ps_fname << " " << tmp_smooth_fname; 
    std::string additional_command = "";

    if (specifics::SMOOTH_LOGK_LOGP)
        additional_command = " --interp_log";
    
    if (process::this_pe == 0) 
        additional_command += " >> " + LOG::LOGGER.getFileName(LOG::TYPE::STD);

    command << additional_command;
      
    LOG::LOGGER.STD("%s\n", command.str().c_str());
    LOG::LOGGER.close();

    // Print from python does not go into LOG::LOGGER
    int s1 = system(command.str().c_str());
    
    LOG::LOGGER.reopen();
    remove(tmp_ps_fname);

    if (s1 != 0)
    {
        LOG::LOGGER.ERR("Error in smoothing.\n");  
        throw std::runtime_error("smoothing error");
    }

    _readScriptOutput(smoothed_power, tmp_smooth_fname);
    remove(tmp_smooth_fname);
}

void OneDQuadraticPowerEstimate::iterate(int number_of_iterations, 
    const char *fname_base)
{
    double total_time = 0, total_time_1it = mytime::timer.getTime();;

    // Construct local queue
    std::vector<OneQSOEstimate> local_queue;
    local_queue.reserve(local_fpaths.size());
    for (auto it = local_fpaths.begin(); it != local_fpaths.end(); ++it)
        local_queue.emplace_back(*it);

    if (specifics::INPUT_QSO_FILE == qio::Picca)
        qio::PiccaFile::clearCache();

    total_time_1it  = mytime::timer.getTime() - total_time_1it;
    LOG::LOGGER.STD("Local files are read in %.1f minutes.", total_time_1it);

    for (int i = 0; i < number_of_iterations; i++)
    {
        LOG::LOGGER.STD("Iteration number %d of %d.\n", i+1, number_of_iterations);
        
        total_time_1it = mytime::timer.getTime();
    
        // Set total Fisher matrix and omn before F to zero for all k, z bins
        initializeIteration();

        // Calculation for each spectrum
        for (std::vector<OneQSOEstimate>::iterator it = local_queue.begin(); 
            it != local_queue.end(); ++it)
        {
            it->oneQSOiteration(powerspectra_fits, 
                dbt_estimate_sum_before_fisher_vector, fisher_matrix_sum);

            // When compiled with debugging feature
            // save matrices to files, break
            #ifdef DEBUG_MATRIX_OUT
            it->fprintfMatrices(fname_base);
            throw std::runtime_error("DEBUGGING QUIT.");
            #endif
        }

        // All reduce if MPI is enabled
        #if defined(ENABLE_MPI)
        MPI_Allreduce(MPI_IN_PLACE, fisher_matrix_sum, (FISHER_SIZE),
            MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
            MPI_Allreduce(MPI_IN_PLACE, dbt_estimate_sum_before_fisher_vector[dbt_i], 
                bins::TOTAL_KZ_BINS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        #endif

        // If fisher is precomputed, copy this into fisher_matrix_sum. 
        // oneQSOiteration iteration will not compute fishers as well.
        if (precomputed_fisher != NULL)
            std::copy(precomputed_fisher, precomputed_fisher + FISHER_SIZE, 
                fisher_matrix_sum);

        try
        {
            invertTotalFisherMatrix();
        }
        catch (std::exception& e)
        {
            LOG::LOGGER.ERR("ERROR while inverting Fisher matrix: %s.\n", e.what());
            throw e;
        }
        
        computePowerSpectrumEstimates();
        total_time_1it  = mytime::timer.getTime() - total_time_1it;
        total_time     += total_time_1it;
        
        if (process::this_pe == 0)
            iterationOutput(fname_base, i, total_time_1it, total_time);

        if (hasConverged())
        {
            LOG::LOGGER.STD("Iteration has converged in %d iterations.\n", i+1);
            break;
        }

        if (i == number_of_iterations-1)
            break;

        try
        {
            _smoothPowerSpectra(powerspectra_fits);
        }
        catch (std::exception& e)
        {
            LOG::LOGGER.ERR("ERROR in Python script: %s\n", e.what());
            throw e;
        }
    }
}

bool OneDQuadraticPowerEstimate::hasConverged()
{
    double  diff, pMax, p1, p2, r, rfull, abs_mean = 0., abs_max = 0.;
    bool    bool_converged = true;
    int     kn, zm;

    for (int i_kz = 0; i_kz < bins::TOTAL_KZ_BINS; ++i_kz)
    {
        bins::getFisherMatrixBinNoFromIndex(i_kz, kn, zm);   
        
        p1 = fabs(current_power_estimate_vector[i_kz]);
        p2 = fabs(previous_power_estimate_vector[i_kz]);
        
        diff = fabs(p1 - p2);
        pMax = std::max(p1, p2);
        r    = diff / pMax;

        if (r > CONVERGENCE_EPS)    bool_converged = false;

        abs_mean += r / bins::DEGREE_OF_FREEDOM;
        abs_max   = std::max(r, abs_max);
    }
    
    LOG::LOGGER.STD("Mean relative change is %.1e.\n"
        "Maximum relative change is %.1e.\n"
        "Old test: Iteration converges when this is less than %.1e\n", 
        abs_mean, abs_max, CONVERGENCE_EPS);
    
    // Perform a chi-square test as well    
    mxhelp::vector_sub(previous_power_estimate_vector, 
        current_power_estimate_vector, bins::TOTAL_KZ_BINS);

    r = 0;

    for (int i_kz = 0; i_kz < bins::TOTAL_KZ_BINS; ++i_kz)
    {        
        double  t = previous_power_estimate_vector[i_kz],
                e = inverse_fisher_matrix_sum[(1+bins::TOTAL_KZ_BINS) * i_kz];

        if (e < 0)  continue;

        r += (t*t) / e;
    }

    r  = sqrt(r / bins::DEGREE_OF_FREEDOM);

    rfull = sqrt(fabs(mxhelp::my_cblas_dsymvdot(previous_power_estimate_vector, 
        fisher_matrix_sum, bins::TOTAL_KZ_BINS)) / bins::DEGREE_OF_FREEDOM);
    
    LOG::LOGGER.TIME("%9.3e | %9.3e |\n", r, abs_mean);
    LOG::LOGGER.STD("Chi^2/dof convergence test:\nDiagonal: %.3f. Full Fisher: %.3f.\n"
                    "Iteration converges when either is less than %.2f.\n", 
                    r, rfull, specifics::CHISQ_CONVERGENCE_EPS);

    bool_converged = r < specifics::CHISQ_CONVERGENCE_EPS || 
        rfull < specifics::CHISQ_CONVERGENCE_EPS;

    return bool_converged;
}

void OneDQuadraticPowerEstimate::writeFisherMatrix(const char *fname)
{
    mxhelp::fprintfMatrix(fname, fisher_matrix_sum, bins::TOTAL_KZ_BINS, 
        bins::TOTAL_KZ_BINS);

    LOG::LOGGER.IO("Fisher matrix saved as %s.\n", fname);
}

void OneDQuadraticPowerEstimate::writeSpectrumEstimates(const char *fname)
{
    FILE *toWrite;
    int i_kz, kn, zm;
    double z, k, p, e;

    toWrite = ioh::open_file(fname, "w");
    
    fprintf(toWrite, "%d %d\n", bins::NUMBER_OF_Z_BINS, bins::NUMBER_OF_K_BANDS);

    auto fprint = [&](const int& zc) { fprintf(toWrite, "%d ", zc); };
    
    std::for_each(Z_BIN_COUNTS, Z_BIN_COUNTS+bins::NUMBER_OF_Z_BINS+2, fprint);

    fprintf(toWrite, "\n");

    for (i_kz = 0; i_kz < bins::TOTAL_KZ_BINS; ++i_kz)
    {
        bins::getFisherMatrixBinNoFromIndex(i_kz, kn, zm);   
        
        z = bins::ZBIN_CENTERS[zm];
        k = bins::KBAND_CENTERS[kn];
        p = current_power_estimate_vector[i_kz] + powerSpectrumFiducial(kn, zm);
        e = sqrt(inverse_fisher_matrix_sum[i_kz+bins::TOTAL_KZ_BINS*i_kz]);

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
    
    specifics::printBuildSpecifics(toWrite);
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
        fidpd13::FIDUCIAL_PD13_PARAMS.A, fidpd13::FIDUCIAL_PD13_PARAMS.n, 
        fidpd13::FIDUCIAL_PD13_PARAMS.alpha, fidpd13::FIDUCIAL_PD13_PARAMS.B, 
        fidpd13::FIDUCIAL_PD13_PARAMS.beta, fidpd13::FIDUCIAL_PD13_PARAMS.lambda);
    fprintf(toWrite, "# -----------------------------------------------------------------\n"
        "# File Template\n"
        "# Nz Nk\n"
        "# n[0] n[1] ... n[Nz] n[Nz+1]\n"
        "# z | k1 | k2 | kc | Pfid | ThetaP | Pest | ErrorP | d | b | t\n"
        "# Nz     : Number of redshift bins\n"
        "# Nk     : Number of k bins\n"
        "# n[i]   : Spectral chunk count in redshift bin i. Left-most and right-most are out of range\n"
        "# z      : Redshift bin center\n"
        "# k1     : Lower edge of the k bin [s km^-1]\n"
        "# k2     : Upper edge of the k bin [s km^-1]\n"
        "# kc     : Center of the k bin [s km^-1]\n"
        "# Pfid   : Fiducial power at kc [km s^-1]\n"
        "# ThetaP : Deviation from Pfid found by quadratic estimator = d - b - t [km s^-1]\n"
        "# Pest   : Pfid + ThetaP [km s^-1]\n"
        "# ErrorP : Error estimated from diagonal terms of the inverse Fisher matrix [km s^-1]\n"
        "# d      : Power estimate before noise (b) and fiducial power (t) subtracted [km s^-1]\n"
        "# b      : Noise estimate [km s^-1]\n"
        "# t      : Fiducial power estimate [km s^-1]\n"
        "# -----------------------------------------------------------------\n");

    fprintf(toWrite, "# %d %d\n# ", bins::NUMBER_OF_Z_BINS, bins::NUMBER_OF_K_BANDS);

    for (zm = 0; zm <= bins::NUMBER_OF_Z_BINS+1; ++zm)
        fprintf(toWrite, "%d ", Z_BIN_COUNTS[zm]);

    fprintf(toWrite, "\n");
    fprintf(toWrite, "%5s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s\n", 
        "z", "k1", "k2", "kc", "Pfid", "ThetaP", "Pest", "ErrorP", "d", "b", "t");

    for (i_kz = 0; i_kz < bins::TOTAL_KZ_BINS; ++i_kz)
    {
        bins::getFisherMatrixBinNoFromIndex(i_kz, kn, zm);   
        
        z  = bins::ZBIN_CENTERS[zm];
        
        k1 = bins::KBAND_EDGES[kn];
        k2 = bins::KBAND_EDGES[kn+1];
        kc = bins::KBAND_CENTERS[kn];

        Pfid = powerSpectrumFiducial(kn, zm);
        ThetaP = current_power_estimate_vector[i_kz];
        ErrorP = sqrt(inverse_fisher_matrix_sum[(1+bins::TOTAL_KZ_BINS)*i_kz]);

        dk = dbt_estimate_fisher_weighted_vector[0][i_kz];
        bk = dbt_estimate_fisher_weighted_vector[1][i_kz];
        tk = dbt_estimate_fisher_weighted_vector[2][i_kz];

        fprintf(toWrite, "%5.3lf %14e %14e %14e %14e %14e %14e %14e %14e %14e %14e\n", 
            z,  k1,  k2,  kc,  Pfid,ThetaP, Pfid+ThetaP, ErrorP, dk, bk, tk);
    }

    fclose(toWrite);
        
    LOG::LOGGER.IO("Quadratic 1D Power Spectrum estimate saved as %s.\n", fname);
    LOG::LOGGER.STD("Quadratic 1D Power Spectrum estimate saved as %s.\n", fname);
}

void OneDQuadraticPowerEstimate::initializeIteration()
{
    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
        std::fill_n(dbt_estimate_sum_before_fisher_vector[dbt_i], bins::TOTAL_KZ_BINS, 0);

    std::fill_n(fisher_matrix_sum, FISHER_SIZE, 0);

    isFisherInverted = false;
}

void OneDQuadraticPowerEstimate::printfSpectra()
{
    int i_kz;

    for (int zm = 0; zm < bins::NUMBER_OF_Z_BINS; ++zm)
        LOG::LOGGER.STD("  P(%.1f, k) |", bins::ZBIN_CENTERS[zm]);

    LOG::LOGGER.STD("\n");
    
    for (int kn = 0; kn < bins::NUMBER_OF_K_BANDS; ++kn)
    {
        for (int zm = 0; zm < bins::NUMBER_OF_Z_BINS; ++zm)
        {
            i_kz = bins::getFisherMatrixIndex(kn, zm);

            LOG::LOGGER.STD(" %10.2e |", current_power_estimate_vector[i_kz] + 
                powerSpectrumFiducial(kn, zm));
        }
        
        LOG::LOGGER.STD("\n");
    }
}

double OneDQuadraticPowerEstimate::powerSpectrumFiducial(int kn, int zm)
{
    if (specifics::TURN_OFF_SFID)  return 0;
    
    double k = bins::KBAND_CENTERS[kn], z = bins::ZBIN_CENTERS[zm];
    
    return fidcosmo::fiducialPowerSpectrum(k, z, &fidpd13::FIDUCIAL_PD13_PARAMS);
}

void OneDQuadraticPowerEstimate::iterationOutput(const char *fname_base, int it, 
    double t1, double tot)
{
    char buf[500];
    printfSpectra();

    sprintf(buf, "%s_it%d_quadratic_power_estimate_detailed.dat", fname_base, it+1);
    writeDetailedSpectrumEstimates(buf);

    sprintf(buf, "%s_it%d_fisher_matrix.dat", fname_base, it+1);
    mxhelp::fprintfMatrix(buf, fisher_matrix_sum, bins::TOTAL_KZ_BINS, 
        bins::TOTAL_KZ_BINS);
    sprintf(buf, "%s_it%d_inversefisher_matrix.dat", fname_base, it+1);
    mxhelp::fprintfMatrix(buf, inverse_fisher_matrix_sum, bins::TOTAL_KZ_BINS, 
        bins::TOTAL_KZ_BINS);
    LOG::LOGGER.IO("Fisher matrix and inverse are saved as %s.\n", buf);

    LOG::LOGGER.STD("This iteration took %.1f minutes."
        " Elapsed time so far is %.1f minutes.\n", t1, tot);
    LOG::LOGGER.TIME("| %2d | %9.3e | %9.3e | ", it, t1, tot);

    mytime::printfTimeSpentDetails();
}

void OneDQuadraticPowerEstimate::readPrecomputedFisher(const char *fname)
{
    int N1, N2;
    mxhelp::fscanfMatrix(fname, precomputed_fisher, N1, N2);

    if (N1 != bins::TOTAL_KZ_BINS || N2 != bins::TOTAL_KZ_BINS)
        throw std::invalid_argument("Precomputed Fisher matrix does not have" 
            " correct number of rows or columns.");
}


















