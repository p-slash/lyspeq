#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

#include <gsl/gsl_errno.h>

#include "core/global_numbers.hpp"
#include "core/mpi_manager.hpp"
#include "core/sq_table.hpp"
#include "core/quadratic_estimate.hpp"
#include "core/fiducial_cosmology.hpp"

#include "mathtools/smoother.hpp"

#include "io/logger.hpp"
#include "io/io_helper_functions.hpp"
#include "io/config_file.hpp"
#include "io/bootstrap_file.hpp"

int main(int argc, char *argv[])
{
    mympi::init(argc, argv);

    if (argc<2)
    {
        fprintf(stderr, "Missing config file!\n");
        mympi::finalize();
        return 1;
    }

    const char *FNAME_CONFIG = argv[1];

    gsl_set_error_handler_off();

    ConfigFile config = ConfigFile();
    std::unique_ptr<OneDQuadraticPowerEstimate> qps;

    // Let all PEs to read config at the same time.
    try
    {
        config.readFile(FNAME_CONFIG);
        LOG::LOGGER.open(config.get("OutputDir", "."), mympi::this_pe);
        specifics::printBuildSpecifics();
        mytime::writeTimeLogHeader();
    }
    catch (std::exception& e)
    {
        fprintf(stderr, "Error while reading config file: %s\n", e.what());
        mympi::abort();
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
        mympi::abort();
        return 1;
    }

    if (process::SAVE_EACH_PE_RESULT) {
        try {
            ioh::boot_saver = std::make_unique<ioh::BootstrapFile>(
                process::FNAME_BASE, bins::NUMBER_OF_K_BANDS,
                bins::NUMBER_OF_Z_BINS, mympi::this_pe);
            mympi::barrier();
        }
        catch (std::exception& e) {
            LOG::LOGGER.ERR("Error while openning BootstrapFile: %s\n",
                e.what());
            mympi::abort();

            return 1;
        }
    }

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
        mympi::abort();

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
        mympi::abort();

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
        mympi::abort();

        return 1;
    }

    qps.reset();
    ioh::boot_saver.reset();
    mympi::finalize();

    return 0;
}










