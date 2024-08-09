#include "cross/cross_exposure.hpp"

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <sstream> // std::ostringstream

#include "core/bootstrapper.hpp"
#include "core/global_numbers.hpp"
#include "core/fiducial_cosmology.hpp"
#include "core/mpi_manager.hpp"
#include "core/progress.hpp"

#include "mathtools/matrix_helper.hpp"

#include "io/io_helper_functions.hpp"
#include "io/logger.hpp"
#include "io/qso_file.hpp"


void _saveQuasarResults(const targetid_quasar_map &quasars) {
    // Create FITS file
    ioh::BootstrapChunksFile bfile(process::FNAME_BASE, mympi::this_pe);

    // For each chunk to a different extention
    for (const auto &[targetid, qso] : quasars) {
        double *pk = qso->dbt_estimate_before_fisher_vector[0].get();
        double *nk = qso->dbt_estimate_before_fisher_vector[1].get();
        double *tk = qso->dbt_estimate_before_fisher_vector[2].get();

        bfile.writeChunk(
            pk, nk, tk,
            qso->fisher_matrix.get(), qso->ndim, qso->istart,
            targetid, qso->z_qso, qso->ra, qso->dec);
    }

    LOG::LOGGER.STD("Quasar results are saved.\n");
}


void OneDCrossExposureQMLE::_countZbinHistogram() {
    for (const auto &[targetid, qso] : quasars)
        for (const auto &exp : qso->exposures)
            Z_BIN_COUNTS[exp->ZBIN + 1] += 1;

    mympi::reduceInplace(Z_BIN_COUNTS.data(), bins::NUMBER_OF_Z_BINS + 2);
    mympi::reduceInplace(&NUMBER_OF_QSOS, 1);
    mympi::reduceInplace(&NUMBER_OF_QSOS_OUT, 1);

    LOG::LOGGER.STD(
        "Number of remaining quasars: %d\n"
        "Number of deleted quasars: %d\n", 
        NUMBER_OF_QSOS, NUMBER_OF_QSOS_OUT);

    LOG::LOGGER.STD(
        LOG::getLineTextFromVector(Z_BIN_COUNTS, "Z bin counts:").c_str());
}


void OneDCrossExposureQMLE::_readOneDeltaFile(const std::string &fname) {
    qio::PiccaFile pFile(fname);
    int number_of_spectra = pFile.getNumberSpectra();

    for (int i = 0; i < number_of_spectra; ++i)
    {
        std::ostringstream fpath;
        fpath << fname << '[' << i + 1 << ']';
        auto oneqso = std::make_unique<OneQsoExposures>(fpath.str());

        if (oneqso->exposures.size() == 0) {
            LOG::LOGGER.ERR(
                "OneDCrossExposureQMLE::_readOneDeltaFile::"
                "No valid exposures in quasar %s.\n",
                fpath.str().c_str());
            continue;
        }

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

    LOG::LOGGER.STD("Read delta files.\n");

    int number_of_files = ioh::readList(flist.c_str(), filepaths);
    // Add parent directory to file path
    for (auto &fq : filepaths)
        fq.insert(0, findir);

    // Each PE reads a different section of files
    int delta_nfiles = number_of_files / mympi::total_pes,
        fstart_this  = delta_nfiles * mympi::this_pe,
        fend_this    = delta_nfiles * (mympi::this_pe + 1);
    
    if (mympi::this_pe == mympi::total_pes - 1)
        fend_this = number_of_files;

    for (int findex = fstart_this; findex < fend_this; ++findex)
        _readOneDeltaFile(filepaths[findex]);

    if (specifics::INPUT_QSO_FILE == qio::Picca)
        qio::PiccaFile::clearCache();

    int init_num_qsos = quasars.size();

    // Remove quasars with not enough pairs
    // C++20 feature
    std::erase_if(quasars, [](const auto &x) {
        auto const& [targetid, one_qso] = x;
        return !one_qso->hasEnoughUniqueExpids();
    });

    NUMBER_OF_QSOS = quasars.size();
    NUMBER_OF_QSOS_OUT = init_num_qsos - NUMBER_OF_QSOS;

    // Print out time it took to read all files into vector
    t2 = mytime::timer.getTime();
    LOG::LOGGER.STD("Reading QSO files took %.2f m.\n", t2 - t1);
    _countZbinHistogram();

    if (quasars.empty())
        throw std::runtime_error("No spectrum in queue. Check files & redshift range.");
}


void OneDCrossExposureQMLE::xQmlEstimate() {
    double total_time = 0, total_time_1it = 0;
    std::vector<double> time_all_pes;
    if (mympi::this_pe == 0)
        time_all_pes.resize(mympi::total_pes);

    _readQSOFiles();

    glmemory::allocMemory();

    total_time_1it = mytime::timer.getTime();

    // Set total Fisher matrix and omn before F to zero for all k, z bins
    initializeIteration();

    std::vector<int> num_quasars_vec;
    mympi::gather<int>(quasars.size(), num_quasars_vec);
    LOG::LOGGER.STD(LOG::getLineTextFromVector(
        num_quasars_vec, "Number of quasars in each task:").c_str());

    // Calculation for each spectrum
    DEBUG_LOG("Running on local queue size %zu\n", quasars.size());
    Progress prog_tracker(quasars.size());
    int total_num_expo_combos = 0;
    for (auto &[targetid, one_qso] : quasars) {
        int num_expo_combos = one_qso->oneQSOiteration(
            dbt_estimate_sum_before_fisher_vector,
            fisher_matrix_sum.get()
        );

        if (num_expo_combos == 0)
            one_qso.reset();

        ++prog_tracker;
        total_num_expo_combos += num_expo_combos;
        DEBUG_LOG("One done.\n");
    }

    if (total_num_expo_combos == 0)
        LOG::LOGGER.ERR("No cross cross exposures in T:%d.\n", mympi::this_pe);

    // C++20 feature
    std::erase_if(quasars, [](const auto &x) {
        auto const& [targetid, one_qso] = x;
        return !one_qso;
    });

    DEBUG_LOG("All done.\n");

    // Scale and copy first before summing across PEs
    // cblas_dscal(bins::FISHER_SIZE, 0.5, fisher_matrix_sum.get(), 1);
    mxhelp::copyUpperToLower(fisher_matrix_sum.get(), bins::TOTAL_KZ_BINS);

    // Save bootstrap files only if MPI is enabled.
    #if defined(ENABLE_MPI)
    double timenow = mytime::timer.getTime() - total_time_1it;
    mympi::gather(timenow, time_all_pes);
    mympi::reduceInplace(&total_num_expo_combos, 1);

    // Save PE estimates to a file
    if (process::SAVE_EACH_PE_RESULT)
        _savePEResult();

    DEBUG_LOG("MPI All reduce.\n");
    if (!specifics::USE_PRECOMPUTED_FISHER)
        mympi::allreduceInplace(fisher_matrix_sum.get(), bins::FISHER_SIZE);

    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
        mympi::allreduceInplace(
            dbt_estimate_sum_before_fisher_vector[dbt_i].get(),
            bins::TOTAL_KZ_BINS);
    #endif

    // If fisher is precomputed, copy this into fisher_matrix_sum. 
    // oneQSOiteration iteration will not compute fishers as well.
    if (specifics::USE_PRECOMPUTED_FISHER)
        std::copy(
            precomputed_fisher.begin(), precomputed_fisher.end(), 
            fisher_matrix_sum.get());

    LOG::LOGGER.STD(
        "Total number of cross exposures: %d.\n", total_num_expo_combos);

    try
    {
        invertTotalFisherMatrix();
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR("ERROR while inverting Fisher matrix: %s.\n", e.what());
        total_time_1it = mytime::timer.getTime() - total_time_1it;
        total_time += total_time_1it;
        iterationOutput(0, total_time_1it, total_time, time_all_pes);
        throw e;
    }

    computePowerSpectrumEstimates();
    total_time_1it = mytime::timer.getTime() - total_time_1it;
    total_time += total_time_1it;

    iterationOutput(0, total_time_1it, total_time, time_all_pes);

    mympi::barrier();

    hasConverged();
    glmemory::dealloc();

    // Save chunk estimates to a file
    if (process::SAVE_EACH_CHUNK_RESULT)
        _saveQuasarResults(quasars);

    // clear exposures
    for (auto &[targetid, qso] : quasars)
        qso->trim(specifics::FAST_BOOTSTRAP);

    if (specifics::NUMBER_OF_BOOTS > 0) {
        PoissonBootstrapper pbooter(
            specifics::NUMBER_OF_BOOTS, solver_invfisher_matrix.get());

        std::vector<std::unique_ptr<OneQSOEstimate>> local_queue;
        local_queue.reserve(quasars.size());
        for (auto it = quasars.begin(); it != quasars.end(); ++it)
            local_queue.push_back(std::move(it->second));

        quasars.clear();
        pbooter.run(local_queue);
    }
}
