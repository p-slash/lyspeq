#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <gsl/gsl_errno.h>

#include "core/global_numbers.hpp"
#include "core/quadratic_estimate.hpp"

#include "io/logger.hpp"

#include "mpi.h" 

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    if (argc<2)
    {
        fprintf(stderr, "Missing config file!\n");
        return 1;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &t_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numthreads);

    const char *FNAME_CONFIG = argv[1];
    int r = 0;

    gsl_set_error_handler_off();

    char FNAME_LIST[300],
         FNAME_RLIST[300],
         INPUT_DIR[300],
         FILEBASE_S[300], FILEBASE_Q[300],
         OUTPUT_DIR[300],
         OUTPUT_FILEBASE[300],
         buf[700];

    int NUMBER_OF_ITERATIONS;
    
    OneDQuadraticPowerEstimate *qps = NULL;

    // Let all PEs to read config at the same time.
    try
    {
        // Read variables from config file and set up bins.
        ioh::readConfigFile( FNAME_CONFIG, FNAME_LIST, FNAME_RLIST, INPUT_DIR, OUTPUT_DIR,
            OUTPUT_FILEBASE, FILEBASE_S, FILEBASE_Q, &NUMBER_OF_ITERATIONS, NULL, NULL, NULL, NULL);
    }
    catch (std::exception& e)
    {
        fprintf(stderr, "Error while reading config file: %s\n", e.what());
        return 1;
    }

    try
    {
        LOG::LOGGER.open(OUTPUT_DIR);
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (t_rank == 0)
        {
            if (TURN_OFF_SFID)  LOG::LOGGER.STD("Fiducial signal matrix is turned off.\n");
    
            specifics::printBuildSpecifics();
            specifics::printConfigSpecifics();
        }
        
    }
    catch (std::exception& e)
    {   
        fprintf(stderr, "Error while logging contructed: %s\n", e.what());
        bins::cleanUpBins();
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    try
    {
        // Allocate and read look up tables
        sq_private_table = new SQLookupTable(INPUT_DIR, FILEBASE_S, FILEBASE_Q, FNAME_RLIST);
        sq_private_table->readTables();
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR("Error while SQ Table contructed: %s\n", e.what());
        bins::cleanUpBins();
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    try
    {
        qps = new OneDQuadraticPowerEstimate(FNAME_LIST, INPUT_DIR);
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR("Error while Quadratic Estimator contructed: %s\n", e.what());
        bins::cleanUpBins();

        delete sq_private_table;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    } 

    try
    {
        sprintf(buf, "%s/%s", OUTPUT_DIR, OUTPUT_FILEBASE);
        qps->iterate(NUMBER_OF_ITERATIONS, buf);
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR("Error while Iteration: %s\n", e.what());
        qps->printfSpectra();

        sprintf(buf, "%s/error_dump_%s_quadratic_power_estimate.dat", OUTPUT_DIR, OUTPUT_FILEBASE);
        qps->writeSpectrumEstimates(buf);
        
        sprintf(buf, "%s/error_dump_%s_fisher_matrix.dat", OUTPUT_DIR, OUTPUT_FILEBASE);
        qps->writeFisherMatrix(buf);

        r=1;
    }
    
    delete qps;

    delete sq_private_table;

    bins::cleanUpBins();
    MPI_Finalize();
    return r;
}










