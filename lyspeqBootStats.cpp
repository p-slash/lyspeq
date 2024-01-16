#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

#include "core/global_numbers.hpp"

inline void usage() {
    fprintf(stdout, "Usage: lyspeqBootStats FNAME_BOOT FNAME_BASE.\n");
}


int printUse(char *argv[]) {
    std::string opt = argv[1];
    if (opt == "-h" || opt == "--help") { usage(); return 1; }
    return 0;
}

int main(int argc, char *argv[]) {
    process::this_pe = 0;
    process::total_pes = 1;

    if (argc < 3) {
        fprintf(stderr, "Missing arguments!\n");
        usage();
        return 1;
    }
    if (printUse(argv)) return 0;

    const std::string FNAME_BOOT = argv[1];

    ConfigFile config = ConfigFile();

    // Let all PEs to read config at the same time.
    try {
        config.readFile(FNAME_CONFIG);
        LOG::LOGGER.open(config.get("OutputDir", "."), process::this_pe);
        specifics::printBuildSpecifics();
        mytime::writeTimeLogHeader();
    } catch (std::exception& e) {
        fprintf(stderr, "Error while reading config file: %s\n", e.what());
        return 1;
    }

    try
    {
        process::readProcess(config);
        bins::readBins(config);
        specifics::readSpecifics(config);
        conv::readConversion(config);
        fidcosmo::readFiducialCosmo(config);
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR("Error while parsing config file: %s\n",
            e.what());
        #if defined(ENABLE_MPI)
        MPI_Abort(MPI_COMM_WORLD, 1);
        #endif
        return 1;
    }

    #if defined(ENABLE_MPI)
    try
    {
        if (process::SAVE_EACH_PE_RESULT)
            ioh::boot_saver = std::make_unique<ioh::BootstrapFile>(process::FNAME_BASE,
                bins::NUMBER_OF_K_BANDS, bins::NUMBER_OF_Z_BINS, process::this_pe);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR("Error while openning BootstrapFile: %s\n",
            e.what());
        #if defined(ENABLE_MPI)
        MPI_Abort(MPI_COMM_WORLD, 1);
        #endif

        return 1;
    }
    #endif

    try
    {
        // Allocate and read look up tables
        process::sq_private_table = std::make_unique<SQLookupTable>(config);

        // Readjust allocated memory wrt save tables
        if (process::SAVE_ALL_SQ_FILES || specifics::USE_RESOLUTION_MATRIX)
        {
            process::sq_private_table->readTables();
            process::updateMemory(-process::sq_private_table->getMaxMemUsage());
        }
        else
            process::updateMemory(-process::sq_private_table->getOneSetMemUsage());
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR("Error while SQ Table contructed: %s\n", e.what());
        #if defined(ENABLE_MPI)
        MPI_Abort(MPI_COMM_WORLD, 1);
        #endif

        return 1;
    }

    process::smoother = std::make_unique<Smoother>(config);

    try
    {
        qps = std::make_unique<OneDQuadraticPowerEstimate>(config);
        config.checkUnusedKeys();
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR("Error while Quadratic Estimator contructed: %s\n", e.what());
        #if defined(ENABLE_MPI)
        MPI_Abort(MPI_COMM_WORLD, 1);
        #endif

        return 1;
    }

    try
    {
        qps->iterate();
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR("Error while Iteration: %s\n", e.what());
        qps->printfSpectra();

        std::string buf = process::FNAME_BASE + "_error_dump_quadratic_power_estimate_detailed.dat";
        qps->writeDetailedSpectrumEstimates(buf.c_str());
        
        buf = process::FNAME_BASE + "_error_dump_fisher_matrix.dat";
        qps->writeFisherMatrix(buf.c_str());

        #if defined(ENABLE_MPI)
        MPI_Abort(MPI_COMM_WORLD, 1);
        #endif

        return 1;
    }

    qps.reset();

    #if defined(ENABLE_MPI)
    // Make sure bootsaver is deleted before
    // MPI finalized
    ioh::boot_saver.reset();
    MPI_Finalize();
    #endif

    return 0;
}
