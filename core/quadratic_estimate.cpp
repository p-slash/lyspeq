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

#include "core/bootstrapper.hpp"
#include "core/one_qso_estimate.hpp"
#include "core/global_numbers.hpp"
#include "core/fiducial_cosmology.hpp"
#include "core/progress.hpp"

#include "mathtools/matrix_helper.hpp"

#include "io/io_helper_functions.hpp"
#include "io/bootstrap_file.hpp"
#include "io/logger.hpp"
#include "io/qso_file.hpp"

#if defined(ENABLE_MPI)
#include "mpi.h" 
#include "core/mpi_merge_sort.cpp"
#endif

// This variable is set in inverse fisher matrix
int _NewDegreesOfFreedom = 0;


void medianStats(
        std::vector<double> &v, double &med_offset, double &max_diff_offset
) {
    // Obtain some statistics
    // convert to off-balance
    double ave_balance = std::accumulate(
        v.begin(), v.end(), 0.) / process::total_pes;

    std::for_each(
        v.begin(), v.end(), [ave_balance](double &t) { t = t / ave_balance - 1; }
    );

    // find min and max offset
    auto minmax_off = std::minmax_element(v.begin(), v.end());
    max_diff_offset = *minmax_off.second - *minmax_off.first;

    // convert to absolute values and find find median
    std::for_each(v.begin(), v.end(), [](double &t) { t = fabs(t); });
    std::sort(v.begin(), v.end());
    med_offset = v[v.size() / 2];
}

void _saveChunkResults(
        std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
) {
    // Create FITS file
    ioh::BootstrapChunksFile bfile(process::FNAME_BASE, process::this_pe);
    // For each chunk to a different extention
    for (auto &one_qso : local_queue) {
        for (auto &one_chunk : one_qso->chunks) {
            double *pk = one_chunk->dbt_estimate_before_fisher_vector[0].get();
            double *nk = one_chunk->dbt_estimate_before_fisher_vector[1].get();
            double *tk = one_chunk->dbt_estimate_before_fisher_vector[2].get();

            int ndim = one_chunk->N_Q_MATRICES;

            bfile.writeChunk(
                pk, nk, tk,
                one_chunk->fisher_matrix.get(), ndim,
                one_chunk->fisher_index_start,
                one_chunk->qFile->id, one_chunk->qFile->z_qso,
                one_chunk->qFile->ra, one_chunk->qFile->dec);
        }
    }
}


/* This function reads following keys from config file:
NumberOfIterations: int
    Number of iterations. Default 1.
PrecomputedFisher: string
    File to precomputed Fisher matrix. If present, Fisher matrix is not
        calculated for spectra. Off by default.
FileNameList: string
    File to spectra to list. Filenames are wrt FileInputDir.
FileInputDir: string
    Directory where files reside.
*/
OneDQuadraticPowerEstimate::OneDQuadraticPowerEstimate(ConfigFile &con)
    : config(con)
{
    Z_BIN_COUNTS.assign(bins::NUMBER_OF_Z_BINS+2, 0);
    NUMBER_OF_ITERATIONS = config.getInteger("NumberOfIterations", 1);
    // Allocate memory
    dbt_estimate_sum_before_fisher_vector.reserve(3);
    dbt_estimate_fisher_weighted_vector.reserve(3);
    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
    {
        dbt_estimate_sum_before_fisher_vector.push_back(
            std::make_unique<double[]>(bins::TOTAL_KZ_BINS));
        dbt_estimate_fisher_weighted_vector.push_back(
            std::make_unique<double[]>(bins::TOTAL_KZ_BINS));
    }

    temp_vector = std::make_unique<double[]>(bins::TOTAL_KZ_BINS);

    previous_power_estimate_vector = std::make_unique<double[]>(bins::TOTAL_KZ_BINS);
    current_power_estimate_vector  = std::make_unique<double[]>(bins::TOTAL_KZ_BINS);
    fisher_matrix_sum              = std::make_unique<double[]>(bins::FISHER_SIZE);
    inverse_fisher_matrix_sum      = std::make_unique<double[]>(bins::FISHER_SIZE);

    powerspectra_fits              = std::make_unique<double[]>(bins::TOTAL_KZ_BINS);

    isFisherInverted = false;
    if (specifics::USE_PRECOMPUTED_FISHER)
        _readPrecomputedFisher(config.get("PrecomputedFisher").c_str());

    std::string flist  = config.get("FileNameList"),
                findir = config.get("FileInputDir");

    if (flist.empty())
        throw std::invalid_argument("Must pass FileNameList.");
    if (findir.empty())
        throw std::invalid_argument("Must pass FileInputDir.");
}

std::vector<std::string>
OneDQuadraticPowerEstimate::_readQSOFiles()
{
    std::string flist  = config.get("FileNameList"),
                findir = config.get("FileInputDir");
    if (findir.back() != '/')
        findir += '/';

    double t1, t2;
    std::vector<std::string> filepaths;
    std::vector< std::pair<double, int> > cpu_fname_vector;

    LOG::LOGGER.STD(
        "Initial reading of quasar spectra and estimating CPU time.\n");

    NUMBER_OF_QSOS = ioh::readList(flist.c_str(), filepaths);
    // Add parent directory to file path
    for (auto &fq : filepaths)
        fq.insert(0, findir);

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
    MPI_Allreduce(
        MPI_IN_PLACE, Z_BIN_COUNTS.data(), bins::NUMBER_OF_Z_BINS+2, 
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

    return _loadBalancing(filepaths, cpu_fname_vector);
}

std::vector<std::string> OneDQuadraticPowerEstimate::_loadBalancing(
        std::vector<std::string> &filepaths,
        std::vector< std::pair<double, int> > &cpu_fname_vector
) {
    LOG::LOGGER.STD(
        "Load balancing for %d tasks available.\n", process::total_pes);
    
    double load_balance_time = mytime::timer.getTime();

    std::vector<double> bucket_time(process::total_pes, 0);
    std::vector<std::string> local_fpaths;
    local_fpaths.reserve(int(1.1*filepaths.size()/process::total_pes));

    std::vector<std::pair <double, int>>::reverse_iterator qe =
        cpu_fname_vector.rbegin();
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

    // sort local_fpaths for caching picca files
    if (specifics::INPUT_QSO_FILE == qio::Picca) 
        std::sort(
            local_fpaths.begin(), local_fpaths.end(),
            qio::PiccaFile::compareFnames
        );

    double med_offset, max_diff_offset;
    medianStats(bucket_time, med_offset, max_diff_offset);

    load_balance_time = mytime::timer.getTime() - load_balance_time;
    LOG::LOGGER.STD(
        "Absolute median offset: %.1e\n"
        "High-Low offset difference: %.1e\n"
        "Load balancing took %.2f sec.\n",
        med_offset, max_diff_offset, load_balance_time * 60.);

    return local_fpaths;
}

void OneDQuadraticPowerEstimate::invertTotalFisherMatrix()
{
    double t = mytime::timer.getTime(), damp = 0;
    int status = 0;

    LOG::LOGGER.STD("Inverting Fisher matrix.\n");

    std::copy(
        fisher_matrix_sum.get(), fisher_matrix_sum.get() + bins::FISHER_SIZE,
        inverse_fisher_matrix_sum.get());

    status = mxhelp::stableInvertSym(
        inverse_fisher_matrix_sum.get(), bins::TOTAL_KZ_BINS,
        _NewDegreesOfFreedom, damp);

    if (status != 0) {
        LOG::LOGGER.STD(
            "* Fisher matrix is damped by adding %.2e to the diagonal.\n",
            damp);
    }

    isFisherInverted = true;

    t = mytime::timer.getTime() - t;
    mytime::time_spent_on_f_inv += t;
}

void OneDQuadraticPowerEstimate::computePowerSpectrumEstimates()
{
    LOG::LOGGER.STD("Estimating power spectrum.\n");

    std::copy(
        current_power_estimate_vector.get(), 
        current_power_estimate_vector.get() + bins::TOTAL_KZ_BINS, 
        previous_power_estimate_vector.get());

    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
    {
        cblas_dsymv(
            CblasRowMajor, CblasUpper,bins::TOTAL_KZ_BINS, 0.5, 
            inverse_fisher_matrix_sum.get(), bins::TOTAL_KZ_BINS,
            dbt_estimate_sum_before_fisher_vector[dbt_i].get(), 1,
            0, dbt_estimate_fisher_weighted_vector[dbt_i].get(), 1);
    }

    std::copy(
        dbt_estimate_fisher_weighted_vector[0].get(), 
        dbt_estimate_fisher_weighted_vector[0].get() + bins::TOTAL_KZ_BINS, 
        current_power_estimate_vector.get());
    mxhelp::vector_sub(
        current_power_estimate_vector.get(), 
        dbt_estimate_fisher_weighted_vector[1].get(), 
        bins::TOTAL_KZ_BINS);
    mxhelp::vector_sub(
        current_power_estimate_vector.get(), 
        dbt_estimate_fisher_weighted_vector[2].get(), 
        bins::TOTAL_KZ_BINS);
}

void OneDQuadraticPowerEstimate::_readScriptOutput(const char *fname, void *itsfits)
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
        
        fr = fscanf(tmp_fit_file, "%le\n", powerspectra_fits.get()+i_kz);

        if (fr != 1)
            throw std::runtime_error("Reading fit power values from tmp_fit_file!");

        powerspectra_fits[i_kz] -= powerSpectrumFiducial(kn, zm);
    }

    fclose(tmp_fit_file);
}

void OneDQuadraticPowerEstimate::_smoothPowerSpectra()
{
    static std::string
        tmp_ps_fname = (
            process::TMP_FOLDER + "/tmp-power-"
            + std::to_string(process::this_pe) + ".txt"),
        tmp_smooth_fname= (
            process::TMP_FOLDER + "/tmp-smooth-"
            + std::to_string(process::this_pe) + ".txt");
    
    // ioh::create_tmp_file(tmp_ps_fname, process::TMP_FOLDER);
    // ioh::create_tmp_file(tmp_smooth_fname, process::TMP_FOLDER);

    writeSpectrumEstimates(tmp_ps_fname.c_str());

    std::ostringstream command("smbivspline.py ", std::ostringstream::ate);
    command << tmp_ps_fname << " " << tmp_smooth_fname; 
    std::string additional_command = "";

    if (specifics::SMOOTH_LOGK_LOGP)
        additional_command = " --interp_log";
 
    additional_command += " >> " + LOG::LOGGER.getFileName(LOG::TYPE::STD);

    command << additional_command;
      
    LOG::LOGGER.STD("%s\n", command.str().c_str());
    LOG::LOGGER.close();

    // Print from python does not go into LOG::LOGGER
    int s1 = system(command.str().c_str());
    
    LOG::LOGGER.reopen();
    remove(tmp_ps_fname.c_str());

    if (s1 != 0)
    {
        LOG::LOGGER.ERR("Error in smoothing.\n");  
        throw std::runtime_error("smoothing error");
    }

    _readScriptOutput(tmp_smooth_fname.c_str());
    remove(tmp_smooth_fname.c_str());
}

void OneDQuadraticPowerEstimate::iterate()
{
    double total_time = 0, total_time_1it = mytime::timer.getTime();;
    std::vector<std::string> local_fpaths = _readQSOFiles();

    // Construct local queue
    // Emplace_back with vector<OneQSOEstimate> leaks memory!!
    std::vector<std::unique_ptr<OneQSOEstimate>> local_queue;
    local_queue.reserve(local_fpaths.size());
    for (const auto &fpath : local_fpaths)
        local_queue.push_back(std::make_unique<OneQSOEstimate>(fpath));

    if (specifics::INPUT_QSO_FILE == qio::Picca)
        qio::PiccaFile::clearCache();

    total_time_1it  = mytime::timer.getTime() - total_time_1it;
    LOG::LOGGER.STD("Local files are read in %.1f minutes.\n", total_time_1it);

    std::vector<double> time_all_pes;
    if (process::this_pe == 0)
        time_all_pes.resize(process::total_pes);

    for (int iteration = 0; iteration < NUMBER_OF_ITERATIONS; iteration++)
    {
        LOG::LOGGER.STD("Iteration number %d of %d.\n", iteration+1, NUMBER_OF_ITERATIONS);
        total_time_1it = mytime::timer.getTime();

        // Set total Fisher matrix and omn before F to zero for all k, z bins
        initializeIteration();

        // Calculation for each spectrum
        DEBUG_LOG("Running on local queue size %zu\n", local_queue.size());
        Progress prog_tracker(local_queue.size());
        for (auto &one_qso : local_queue) {
            one_qso->oneQSOiteration(
                powerspectra_fits.get(), 
                dbt_estimate_sum_before_fisher_vector,
                fisher_matrix_sum.get()
            );

            ++prog_tracker;
            DEBUG_LOG("One done.\n");
        }

        DEBUG_LOG("All done.\n");

        // Scale and copy first before summing across PEs
        cblas_dscal(bins::FISHER_SIZE, 0.5, fisher_matrix_sum.get(), 1);
        mxhelp::copyUpperToLower(fisher_matrix_sum.get(), bins::TOTAL_KZ_BINS);

        // Save bootstrap files only if MPI is enabled.
        #if defined(ENABLE_MPI)
        double timenow =  mytime::timer.getTime() - total_time_1it;
        MPI_Gather(
            &timenow, 1, MPI_DOUBLE, time_all_pes.data(), 1, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        // Save PE estimates to a file
        if (process::SAVE_EACH_PE_RESULT)
            _savePEResult();

        DEBUG_LOG("MPI All reduce.\n");
        if (!specifics::USE_PRECOMPUTED_FISHER)
            MPI_Allreduce(
                MPI_IN_PLACE,
                fisher_matrix_sum.get(), bins::FISHER_SIZE,
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
            MPI_Allreduce(
                MPI_IN_PLACE,
                dbt_estimate_sum_before_fisher_vector[dbt_i].get(), 
                bins::TOTAL_KZ_BINS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        #endif

        // If fisher is precomputed, copy this into fisher_matrix_sum. 
        // oneQSOiteration iteration will not compute fishers as well.
        if (specifics::USE_PRECOMPUTED_FISHER)
            std::copy(
                precomputed_fisher.begin(), precomputed_fisher.end(), 
                fisher_matrix_sum.get());

        try
        {
            invertTotalFisherMatrix();
        }
        catch (std::exception& e)
        {
            LOG::LOGGER.ERR("ERROR while inverting Fisher matrix: %s.\n", e.what());
            total_time_1it = mytime::timer.getTime() - total_time_1it;
            total_time    += total_time_1it;
            iterationOutput(iteration, total_time_1it, total_time, time_all_pes);
            throw e;
        }

        computePowerSpectrumEstimates();
        total_time_1it = mytime::timer.getTime() - total_time_1it;
        total_time    += total_time_1it;

        iterationOutput(iteration, total_time_1it, total_time, time_all_pes);

        #if defined(ENABLE_MPI)
        MPI_Barrier(MPI_COMM_WORLD);
        #endif

        if (hasConverged())
        {
            LOG::LOGGER.STD(
                "Iteration has converged in %d iterations.\n",
                iteration+1);
            break;
        }

        if (iteration == NUMBER_OF_ITERATIONS-1)
            break;

        try
        {
            if (process::this_pe == 0)
                _smoothPowerSpectra();
            #if defined(ENABLE_MPI)
            MPI_Bcast(
                powerspectra_fits.get(), bins::TOTAL_KZ_BINS,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
            #endif
        }
        catch (std::exception& e)
        {
            LOG::LOGGER.ERR("ERROR in Python script: %s\n", e.what());
            throw e;
        }
    }

    // Save chunk estimates to a file
    if (process::SAVE_EACH_CHUNK_RESULT)
        _saveChunkResults(local_queue);

    if (specifics::NUMBER_OF_BOOTS > 0) {
        PoissonBootstrapper pbooter(specifics::NUMBER_OF_BOOTS);
        pbooter.run(local_queue);
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
        
        if (p1 == 0 && p2 == 0) continue;

        diff = fabs(p1 - p2);
        pMax = std::max(p1, p2);
        r    = diff / pMax;

        if (r > specifics::CHISQ_CONVERGENCE_EPS)
            bool_converged = false;

        abs_mean += r / _NewDegreesOfFreedom;
        abs_max   = std::max(r, abs_max);
    }

    LOG::LOGGER.STD("Mean relative change is %.1e.\n"
        "Maximum relative change is %.1e.\n"
        "Old test: Iteration converges when this is less than %.1e\n", 
        abs_mean, abs_max, specifics::CHISQ_CONVERGENCE_EPS);
    
    // Perform a chi-square test as well    
    mxhelp::vector_sub(previous_power_estimate_vector.get(), 
        current_power_estimate_vector.get(), bins::TOTAL_KZ_BINS);

    r = 0;

    for (int i_kz = 0; i_kz < bins::TOTAL_KZ_BINS; ++i_kz)
    {        
        double  t = previous_power_estimate_vector[i_kz],
                e = inverse_fisher_matrix_sum[(1+bins::TOTAL_KZ_BINS) * i_kz];

        if (e <= 0)  continue;

        r += (t*t) / e;
    }

    r  = sqrt(r / _NewDegreesOfFreedom);

    rfull = sqrt(fabs(mxhelp::my_cblas_dsymvdot(
        previous_power_estimate_vector.get(),
        fisher_matrix_sum.get(), temp_vector.get(), bins::TOTAL_KZ_BINS)
    ) / _NewDegreesOfFreedom);
    
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
    mxhelp::fprintfMatrix(fname, fisher_matrix_sum.get(), bins::TOTAL_KZ_BINS, 
        bins::TOTAL_KZ_BINS);

    LOG::LOGGER.STD("Fisher matrix saved as %s.\n", fname);
}

void OneDQuadraticPowerEstimate::writeSpectrumEstimates(const char *fname)
{
    FILE *toWrite;
    int i_kz, kn, zm;
    double z, k, p, e;

    toWrite = ioh::open_file(fname, "w");
    
    fprintf(toWrite, "%d %d\n", bins::NUMBER_OF_Z_BINS, bins::NUMBER_OF_K_BANDS);

    auto fprint = [toWrite](const int& zc) { fprintf(toWrite, "%d ", zc); };

    std::for_each(Z_BIN_COUNTS.begin(), Z_BIN_COUNTS.end(), fprint);

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
    
    // LOG::LOGGER.IO("Quadratic 1D Power Spectrum estimate saved as %s.\n", fname);
    LOG::LOGGER.STD("Quadratic 1D Power Spectrum estimate saved as %s.\n", fname);
}

void OneDQuadraticPowerEstimate::writeDetailedSpectrumEstimates(const char *fname)
{
    FILE *toWrite = ioh::open_file(fname, "w");

    specifics::printBuildSpecifics(toWrite);
    config.writeConfig(toWrite);

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
        "# Fd     : d before Fisher\n"
        "# Fb     : b before Fisher\n"
        "# Ft     : t before Fisher\n"
        "# -----------------------------------------------------------------\n");

    fprintf(toWrite, "# %d %d\n# ", bins::NUMBER_OF_Z_BINS, bins::NUMBER_OF_K_BANDS);

    for (int i = 0; i <= bins::NUMBER_OF_Z_BINS + 1; ++i)
        fprintf(toWrite, "%d ", Z_BIN_COUNTS[i]);

    fprintf(toWrite, "\n");
    fprintf(
        toWrite,
        "%5s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s\n", 
        "z", "k1", "k2", "kc", "Pfid", "ThetaP", "Pest", "ErrorP",
        "d", "b", "t", "Fd", "Fb", "Ft");

    int kn = 0, zm = 0;
    for (int i_kz = 0; i_kz < bins::TOTAL_KZ_BINS; ++i_kz)
    {
        bins::getFisherMatrixBinNoFromIndex(i_kz, kn, zm);   
        double
            z = bins::ZBIN_CENTERS[zm],
            k1 = bins::KBAND_EDGES[kn],
            k2 = bins::KBAND_EDGES[kn + 1],
            kc = bins::KBAND_CENTERS[kn],

            Pfid = powerSpectrumFiducial(kn, zm),
            ThetaP = current_power_estimate_vector[i_kz],
            ErrorP = sqrt(inverse_fisher_matrix_sum[(1 + bins::TOTAL_KZ_BINS) * i_kz]),

            dk = dbt_estimate_fisher_weighted_vector[0][i_kz],
            bk = dbt_estimate_fisher_weighted_vector[1][i_kz],
            tk = dbt_estimate_fisher_weighted_vector[2][i_kz],

            Fdk = dbt_estimate_sum_before_fisher_vector[0][i_kz],
            Fbk = dbt_estimate_sum_before_fisher_vector[1][i_kz],
            Ftk = dbt_estimate_sum_before_fisher_vector[2][i_kz];

        fprintf(
            toWrite,
            "%5.3lf %14e %14e %14e %14e %14e %14e %14e %14e %14e %14e %14e %14e %14e\n", 
            z, k1, k2, kc, Pfid, ThetaP, Pfid + ThetaP, ErrorP,
            dk, bk, tk, Fdk, Fbk, Ftk);
    }

    fclose(toWrite);
    LOG::LOGGER.STD("Quadratic 1D Power Spectrum estimate saved as %s.\n", fname);
}

void OneDQuadraticPowerEstimate::initializeIteration()
{
    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
        std::fill_n(
            dbt_estimate_sum_before_fisher_vector[dbt_i].get(),
            bins::TOTAL_KZ_BINS, 0);

    std::fill_n(fisher_matrix_sum.get(), bins::FISHER_SIZE, 0);

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

void OneDQuadraticPowerEstimate::iterationOutput(
        int it, double t1, double tot, std::vector<double> &times_all_pes
) {
    if (process::this_pe != 0)
        return;

    std::ostringstream buffer(process::FNAME_BASE, std::ostringstream::ate);
    printfSpectra();

    buffer << "_it" << it+1 << "_quadratic_power_estimate_detailed.txt";
    writeDetailedSpectrumEstimates(buffer.str().c_str());

    buffer.str(process::FNAME_BASE);
    buffer << "_it" << it+1 << "_fisher_matrix.txt";
    mxhelp::fprintfMatrix(
        buffer.str().c_str(), fisher_matrix_sum.get(),
        bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);

    buffer.str(process::FNAME_BASE);
    buffer << "_it" << it+1 << "_inversefisher_matrix.txt";
    mxhelp::fprintfMatrix(
        buffer.str().c_str(), inverse_fisher_matrix_sum.get(),
        bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);
    LOG::LOGGER.STD(
        "Fisher matrix and inverse are saved as %s.\n",
        buffer.str().c_str());

    LOG::LOGGER.STD(
        "----------------------------------\n"
        "This iteration took %.1f minutes."
        " Elapsed time so far is %.1f minutes.\n", t1, tot);
    LOG::LOGGER.TIME("| %2d | %9.3e | %9.3e | ", it, t1, tot);

    mytime::printfTimeSpentDetails();

    if (process::total_pes == 1) {
        LOG::LOGGER.STD("----------------------------------\n");
        return;
    }

    double med_offset, max_diff_offset;
    medianStats(times_all_pes, med_offset, max_diff_offset);

    LOG::LOGGER.STD(
        "Measured load balancing offsets:\n"
        "Absolute median offset: %.1e\n"
        "High-Low offset difference: %.1e\n"
        "----------------------------------\n",
        med_offset, max_diff_offset);
}

void OneDQuadraticPowerEstimate::_readPrecomputedFisher(const std::string &fname)
{
    int N1, N2;
    precomputed_fisher = mxhelp::fscanfMatrix(fname.c_str(), N1, N2);

    if (N1 != bins::TOTAL_KZ_BINS || N2 != bins::TOTAL_KZ_BINS
        || precomputed_fisher.size() != (unsigned long) bins::FISHER_SIZE)
        throw std::invalid_argument("Precomputed Fisher matrix does not have" 
            " correct number of rows or columns.");
}

#if defined(ENABLE_MPI)
void OneDQuadraticPowerEstimate::_savePEResult()
{
    auto tmppower = std::make_unique<double[]>(bins::TOTAL_KZ_BINS);

    std::copy(
        dbt_estimate_sum_before_fisher_vector[0].get(), 
        dbt_estimate_sum_before_fisher_vector[0].get() + bins::TOTAL_KZ_BINS, 
        tmppower.get());

    mxhelp::vector_sub(
        tmppower.get(),
        dbt_estimate_sum_before_fisher_vector[1].get(),
        bins::TOTAL_KZ_BINS);
    mxhelp::vector_sub(
        tmppower.get(),
        dbt_estimate_sum_before_fisher_vector[2].get(),
        bins::TOTAL_KZ_BINS);

    try
    {
        ioh::boot_saver->writeBoot(tmppower.get(), fisher_matrix_sum.get());
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR("ERROR: Saving PE results: %d\n", process::this_pe);
    }
}
#else
void OneDQuadraticPowerEstimate::_savePEResult()
{
    return;
}
#endif
