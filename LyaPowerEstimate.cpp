#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <gsl/gsl_errno.h>

#include "core/global_numbers.hpp"
#include "core/quadratic_estimate.hpp"
#include "core/spectrograph_functions.hpp"

#include "io/io_helper_functions.hpp"

#if defined(_OPENMP)
#include <omp.h> // omp_get_thread_num()
#endif

int main(int argc, char const *argv[])
{
    if (argc<2)
    {
        fprintf(stderr, "Missing config file!\n");
        return -1;
    }
    const char *FNAME_CONFIG = argv[1];
    
    char FNAME_LIST[300], \
         FNAME_RLIST[300], \
         INPUT_DIR[300], \
         FILEBASE_S[300], FILEBASE_Q[300], \
         OUTPUT_DIR[300], \
         OUTPUT_FILEBASE[300],\
         buf[700];

    int NUMBER_OF_ITERATIONS;
    
    pd13_fit_params FIDUCIAL_PD13_PARAMS;

    OneDQuadraticPowerEstimate *qps = NULL;
    
    print_build_specifics();

    try
    {
        // Read variables from config file and set up bins.
        read_config_file(   FNAME_CONFIG, \
                            FIDUCIAL_PD13_PARAMS, \
                            FNAME_LIST, FNAME_RLIST, INPUT_DIR, OUTPUT_DIR, \
                            OUTPUT_FILEBASE, FILEBASE_S, FILEBASE_Q, \
                            &NUMBER_OF_ITERATIONS, \
                            NULL, NULL, NULL, NULL);

        // Allocate and read look up tables
        sq_shared_table = new SQLookupTable(INPUT_DIR, FILEBASE_S, FILEBASE_Q, FNAME_RLIST);

        #if defined(_OPENMP)
        omp_set_dynamic(0); // Turn off dynamic threads
        numthreads = omp_get_max_threads();
        #endif

        gsl_set_error_handler_off();
        
#pragma omp parallel
{
        #if defined(_OPENMP)
        t_rank  = omp_get_thread_num();
        #endif

        // Create private copy for interpolation is not thread safe!
        if (t_rank == 0)     sq_private_table = sq_shared_table;
        else                    sq_private_table = new SQLookupTable(*sq_shared_table);
}

        qps = new OneDQuadraticPowerEstimate(   FNAME_LIST, INPUT_DIR, \
                                                &FIDUCIAL_PD13_PARAMS);

        sprintf(buf, "%s/%s", OUTPUT_DIR, OUTPUT_FILEBASE);
        qps->iterate(NUMBER_OF_ITERATIONS, buf);

        delete qps;
        // delete sq_shared_table;

        clean_up_bins();
    }
    catch (std::exception& e)
    {
        printf("%s\n", e.what());
        return -1;
    }
    catch (const char* msg)
    {
        if (qps != NULL)
        {
            qps->printfSpectra();

            sprintf(buf, "%s/error_dump_%s_quadratic_power_estimate.dat", OUTPUT_DIR, OUTPUT_FILEBASE);
            qps->write_spectrum_estimates(buf);
            
            sprintf(buf, "%s/error_dump_%s_fisher_matrix.dat", OUTPUT_DIR, OUTPUT_FILEBASE);
            qps->write_fisher_matrix(buf);

            delete qps;
        }
        
        // fprintf(stderr, "%s\n", msg);
        return -1;
    }

    return 0;
}










