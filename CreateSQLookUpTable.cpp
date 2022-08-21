// Q matrices are not scaled with redshift binning function
#include <cstdio>
#include <cstdlib>
#include <cstring> // strcmp
#include <stdexcept>

#include <gsl/gsl_errno.h>

#if defined(ENABLE_MPI)
#include "mpi.h" 
#endif

#include "core/global_numbers.hpp"
#include "core/sq_table.hpp"
#include "core/fiducial_cosmology.hpp"

#include "io/config_file.hpp"
#include "io/logger.hpp"

#define ONE_SIGMA_2_FWHM 2.35482004503

int main(int argc, char *argv[])
{
    #if defined(ENABLE_MPI)
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &process::this_pe);
    MPI_Comm_size(MPI_COMM_WORLD, &process::total_pes);
    #else
    process::this_pe = 0;
    process::total_pes = 1;
    #endif

    if (argc<2)
    {
        fprintf(stderr, "Missing config file!\n");
        #if defined(ENABLE_MPI)
        MPI_Finalize();
        #endif
        return -1;
    }

    const char *FNAME_CONFIG = argv[1];
    bool force_rewrite = true;

    if (argc == 3)
        force_rewrite = !(strcmp(argv[2], "--unforce") == 0);

    ConfigFile config = ConfigFile(FNAME_CONFIG);

    try
    {
        // Read variables from config file and set up bins.
        config.readAll();
        process::readProcess(config);
        bins::readBins(config);
        specifics::readSpecifics(config);
        fidcosmo::readFiducialCosmo(config);
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
        MPI_Barrier(MPI_COMM_WORLD);
        #endif

        if (specifics::TURN_OFF_SFID)  LOG::LOGGER.STD("Fiducial signal matrix is turned off.\n");
        
        specifics::printBuildSpecifics();
        specifics::printConfigSpecifics();
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

    gsl_set_error_handler_off();

    try
    {
        process::sq_private_table = new SQLookupTable(config);
        process::sq_private_table->computeTables(force_rewrite);
    }
    catch (std::exception& e)
    {   
        LOG::LOGGER.ERR("Error constructing SQ Table: %s\n", e.what());
        bins::cleanUpBins();

        #if defined(ENABLE_MPI)
        MPI_Abort(MPI_COMM_WORLD, 1);
        #endif
        
        return 1;
    }

    LOG::LOGGER.STD("Deleting...\n");
    delete process::sq_private_table;
    process::sq_private_table = NULL;

    bins::cleanUpBins();       

    #if defined(ENABLE_MPI)
    LOG::LOGGER.STD("MPI Finalize...\n");
    MPI_Finalize();
    #endif

    return 0;
}










