#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <gsl/gsl_errno.h>

#if defined(ENABLE_MPI)
#include "mpi.h" 
#endif

#include "core/global_numbers.hpp"
#include "core/sq_table.hpp"
#include "core/quadratic_estimate.hpp"
#include "mathtools/smoother.hpp"

#include "io/logger.hpp"
#include "io/io_helper_functions.hpp"
#include "io/config_file.hpp"
#include "io/bootstrap_file.hpp"

int main(int argc, char *argv[])
{
    #if defined(ENABLE_MPI)
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &process::this_pe);
    MPI_Comm_size(MPI_COMM_WORLD, &process::total_pes);
    #else
    process::this_pe   = 0;
    process::total_pes = 1;
    #endif

    if (argc<2)
    {
        fprintf(stderr, "Missing config file!\n");
        #if defined(ENABLE_MPI)
        MPI_Finalize();
        #endif
        return 1;
    }

    const char *FNAME_CONFIG = argv[1];
    int r = 0;

    gsl_set_error_handler_off();

    ConfigFile config = ConfigFile(FNAME_CONFIG);
    OneDQuadraticPowerEstimate *qps = NULL;

    // Let all PEs to read config at the same time.
    try
    { 
        config.readAll();
        process::readProcess(config);
        bins::readBins(config);
        specifics::readSpecifics(config);
    }
    catch (std::exception& e)
    {
        fprintf(stderr, "Error while reading config file: %s\n", e.what());
        return 1;
    }

    try
    {
        LOG::LOGGER.open(config.get("OutputDir", "."), process::this_pe);

        #if defined(ENABLE_MPI)
        if (process::SAVE_EACH_PE_RESULT)
            ioh::boot_saver = new ioh::BootstrapFile(process::FNAME_BASE,
                bins::NUMBER_OF_K_BANDS, bins::NUMBER_OF_Z_BINS, bins::TOTAL_KZ_BINS);
        MPI_Barrier(MPI_COMM_WORLD);
        #endif


        if (specifics::TURN_OFF_SFID)
            LOG::LOGGER.STD("Fiducial signal matrix is turned off.\n");

        specifics::printBuildSpecifics();
        specifics::printConfigSpecifics();
        mytime::writeTimeLogHeader();
    }
    catch (std::exception& e)
    {   
        fprintf(stderr, "Error while logging contructed: %s\n", e.what());
        bins::cleanUpBins();

        #if defined(ENABLE_MPI)
        MPI_Abort(MPI_COMM_WORLD, 1);
        #endif

        return 1;
    }

    try
    {
        // Allocate and read look up tables
        process::sq_private_table = new SQLookupTable(config);

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
        bins::cleanUpBins();

        #if defined(ENABLE_MPI)
        delete ioh::boot_saver;
        MPI_Abort(MPI_COMM_WORLD, 1);
        #endif

        return 1;
    }

    Smoother::setParameters(config.getInteger("SmoothNoiseWeights", -1));
    Smoother::setGaussianKernel();

    try
    {
        qps = new OneDQuadraticPowerEstimate(config);
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR("Error while Quadratic Estimator contructed: %s\n", e.what());
        bins::cleanUpBins();

        delete process::sq_private_table;

        #if defined(ENABLE_MPI)
        delete ioh::boot_saver;
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

        char buf[512];
        sprintf(buf, "%s_error_dump_quadratic_power_estimate_detailed.dat", process::FNAME_BASE.c_str());
        qps->writeDetailedSpectrumEstimates(buf);
        
        sprintf(buf, "%s_error_dump_fisher_matrix.dat", process::FNAME_BASE.c_str());
        qps->writeFisherMatrix(buf);

        delete qps;
        delete process::sq_private_table;
        bins::cleanUpBins();
        #if defined(ENABLE_MPI)
        delete ioh::boot_saver;
        MPI_Abort(MPI_COMM_WORLD, 1);
        #endif

        return 1;
    }

    delete qps;
    delete process::sq_private_table;

    bins::cleanUpBins();

    #if defined(ENABLE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
    delete ioh::boot_saver;
    MPI_Finalize();
    #endif

    return r;
}










