#include "cross/cross_exposure.hpp"

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
#include "io/logger.hpp"
#include "io/qso_file.hpp"

#if defined(ENABLE_MPI)
#include "mpi.h" 
#include "core/mpi_merge_sort.cpp"
#endif


void OneDCrossExposureQMLE::_countZbinHistogram() {
    for (const auto &[targetid, qso] : quasars)
        for (const auto &exp : qso->exposures)
            Z_BIN_COUNTS[exp->ZBIN + 1] += 1;

    // MPI Reduce ZBIN_COUNTS
    #if defined(ENABLE_MPI)
    MPI_Allreduce(
        MPI_IN_PLACE, Z_BIN_COUNTS.data(), bins::NUMBER_OF_Z_BINS + 2, 
        MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    #endif

    NUMBER_OF_QSOS_OUT = Z_BIN_COUNTS[0] + Z_BIN_COUNTS[bins::NUMBER_OF_Z_BINS+1];

    LOG::LOGGER.STD("Z bin counts: ");
    for (int zm = 0; zm < bins::NUMBER_OF_Z_BINS+2; zm++)
        LOG::LOGGER.STD("%d ", Z_BIN_COUNTS[zm]);
    LOG::LOGGER.STD("\nNumber of quasars: %d\nQSOs in z bins: %d\n", 
        NUMBER_OF_QSOS, NUMBER_OF_QSOS - NUMBER_OF_QSOS_OUT);
}


void OneDCrossExposureQMLE::_readOneDeltaFile(const std::string &fname) {
    qio::PiccaFile pFile(fname);
    int number_of_spectra = pFile.getNumberSpectra();

    for (int i = 0; i < number_of_spectra; ++i)
    {
        std::ostringstream fpath;
        fpath << fname << '[' << i + 1 << ']';
        auto oneqso = std::make_unique<OneQsoExposures>(fpath.str());

        auto kumap_itr = quasars.find(oneqso->targetid);

        if (kumap_itr == quasars.end())
            quasars[oneqso->targetid] = std::move(oneqso);
        else
            kumap_itr->second->addExposures(oneqso.get());
    }
}

void OneDCrossExposureQMLE::_readQSOFiles() {
    std::string flist  = config.get("FileNameList"),
                findir = config.get("FileInputDir");
    if (findir.back() != '/')
        findir += '/';

    double t1 = mytime::timer.getTime(), t2 = 0;
    std::vector<std::string> filepaths;
    std::vector< std::pair<double, int> > cpu_fname_vector;

    LOG::LOGGER.STD("Read delta files.\n");

    int number_of_files = ioh::readList(flist.c_str(), filepaths);
    // Add parent directory to file path
    for (auto &fq : filepaths)
        fq.insert(0, findir);

    // Each PE reads a different section of files
    int delta_nfiles = number_of_files / process::total_pes,
        fstart_this  = delta_nfiles * process::this_pe,
        fend_this    = delta_nfiles * (process::this_pe + 1);
    
    if (process::this_pe == process::total_pes - 1)
        fend_this = number_of_files;

    for (int findex = fstart_this; findex < fend_this; ++findex)
        _readOneDeltaFile(filepaths[findex]);

    if (specifics::INPUT_QSO_FILE == qio::Picca)
        qio::PiccaFile::clearCache();

    // Print out time it took to read all files into vector
    t2 = mytime::timer.getTime();
    LOG::LOGGER.STD("Reading QSO files took %.2f m.\n", t2 - t1);
    _countZbinHistogram();

    if (cpu_fname_vector.empty())
        throw std::runtime_error("No spectrum in queue. Check files & redshift range.");
}


void OneDCrossExposureQMLE::iterate()
{
    double total_time = 0, total_time_1it = 0;
    std::vector<double> time_all_pes;
    if (process::this_pe == 0)
        time_all_pes.resize(process::total_pes);

    _readQSOFiles();

    glmemory::allocMemory();

    total_time_1it = mytime::timer.getTime();

    // Set total Fisher matrix and omn before F to zero for all k, z bins
    initializeIteration();

    // Calculation for each spectrum
    DEBUG_LOG("Running on local queue size %zu\n", quasars.size());
    Progress prog_tracker(quasars.size());
    for (const auto &[targetid, one_qso] : quasars) {
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
        iterationOutput(0, total_time_1it, total_time, time_all_pes);
        throw e;
    }

    computePowerSpectrumEstimates();
    total_time_1it = mytime::timer.getTime() - total_time_1it;
    total_time    += total_time_1it;

    iterationOutput(0, total_time_1it, total_time, time_all_pes);

    #if defined(ENABLE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

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

    glmemory::dealloc();

    // // Save chunk estimates to a file
    // if (process::SAVE_EACH_CHUNK_RESULT)
    //     _saveChunkResults(quasars);

    // if (specifics::NUMBER_OF_BOOTS > 0) {
    //     PoissonBootstrapper pbooter(
    //         specifics::NUMBER_OF_BOOTS, solver_invfisher_matrix.get());
    //     pbooter.run(quasars);
    // }
}