// Q matrices are not scaled with redshift binning function
#include <cstdio>
#include <cstdlib>
#include <cstring> // strcmp
#include <stdexcept>

#include <gsl/gsl_errno.h>

#include "core/global_numbers.hpp"
#include "core/fiducial_cosmology.hpp"
#include "core/sq_table.hpp"
#include "io/logger.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

#define ONE_SIGMA_2_FWHM 2.35482004503

int main(int argc, char const *argv[])
{
    if (argc<2)
    {
        fprintf(stderr, "Missing config file!\n");
        return -1;
    }

    const char *FNAME_CONFIG = argv[1];
    bool force_rewrite = true;

    if (argc == 3)
        force_rewrite = !(strcmp(argv[2], "--unforce") == 0);

    char FNAME_RLIST[300], \
         OUTPUT_DIR[300], \
         OUTPUT_FILEBASE_S[300],\
         OUTPUT_FILEBASE_Q[300];

    int Nv, Nz;

    double PIXEL_WIDTH, LENGTH_V;

    try
    {
        // Read variables from config file and set up bins.
        readConfigFile( FNAME_CONFIG, 
                        NULL, FNAME_RLIST, OUTPUT_DIR, NULL, 
                        NULL, OUTPUT_FILEBASE_S, OUTPUT_FILEBASE_Q, 
                        NULL, 
                        &Nv, &Nz, &PIXEL_WIDTH, &LENGTH_V);
    }
    catch (std::exception& e)
    {
        fprintf(stderr, "Error while reading config file: %s\n", e.what());
        return -1;
    }

    try
    {
        LOG::LOGGER.open(OUTPUT_DIR);
        if (TURN_OFF_SFID)  LOG::LOGGER.STD("Fiducial signal matrix is turned off.\n");
        
        specifics::printBuildSpecifics();
        specifics::printConfigSpecifics();
    }
    catch (std::exception& e)
    {   
        fprintf(stderr, "Error while logging contructed: %s\n", e.what());
        bins::cleanUpBins();
        return -1;
    }

    gsl_set_error_handler_off();

    #if defined(_OPENMP)
    omp_set_dynamic(0); // Turn off dynamic threads
    numthreads = omp_get_max_threads();
    #endif

    try
    {
        sq_shared_table = new SQLookupTable(OUTPUT_DIR, OUTPUT_FILEBASE_S, OUTPUT_FILEBASE_Q, FNAME_RLIST);
        sq_shared_table->computeTables(PIXEL_WIDTH, Nv, Nz, LENGTH_V, force_rewrite);
    }
    catch (std::exception& e)
    {   
        LOG::LOGGER.ERR("Error constructing SQ Table (Reading R values): %s\n", e.what());
        bins::cleanUpBins();
        return -1;
    }

    // try
    // {
        

    //     // S matrices are written.
    //     // ---------------------
    // }
    // catch (std::exception& e)
    // {   
    //     LOG::LOGGER.ERR("Error in signal computation: %s\n", e.what());
    // }


    // try
    // {
    //     // Integrate derivative matrices
    //     // int subNz = Nz / NUMBER_OF_Z_BINS;
        
    //     // Q matrices are written.
    //     // ---------------------

    //     // delete [] big_temp_array;
    // }
    // catch (std::exception& e)
    // {   
    //     LOG::LOGGER.ERR("Error derivative computation: %s\n", e.what());
    //     bins::cleanUpBins();

    // }

    bins::cleanUpBins();       

    return 0;
}










