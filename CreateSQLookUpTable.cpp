// Q matrices are not scaled with redshift binning function
#include <cstdio>
#include <cstdlib>
#include <cstring> // strcmp
#include <ctime>    /* clock_t, clock, CLOCKS_PER_SEC */

#include <gsl/gsl_errno.h>

#include "core/global_numbers.hpp"
#include "core/spectrograph_functions.hpp"
#include "core/fiducial_cosmology.hpp"

#include "gsltools/fourier_integrator.hpp"

#include "io/io_helper_functions.hpp"
#include "io/sq_lookup_table_file.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

int main(int argc, char const *argv[])
{
    if (argc<2)
    {
        fprintf(stderr, "Missing config file!\n");
        return -1;
    }

    const char *FNAME_CONFIG = argv[1];
    bool force_rewrite = true;
    double time_spent_table_sfid, time_spent_table_q;

    if (argc == 3)
        force_rewrite = !(strcmp(argv[2], "--unforce") == 0);

    char FNAME_RLIST[300], \
         OUTPUT_DIR[300], \
         OUTPUT_FILEBASE_S[300],\
         OUTPUT_FILEBASE_Q[300],\
         buf[500];

    int NUMBER_OF_Rs, *R_VALUES, Nv, Nz;

    double PIXEL_WIDTH, LENGTH_V;

    pd13_fit_params FIDUCIAL_PD13_PARAMS;

    print_build_specifics();
    
    try
    {
        // Read variables from config file and set up bins.
        read_config_file(   FNAME_CONFIG, \
                            FIDUCIAL_PD13_PARAMS, \
                            NULL, FNAME_RLIST, OUTPUT_DIR, NULL, \
                            NULL, OUTPUT_FILEBASE_S, OUTPUT_FILEBASE_Q, \
                            NULL, \
                            &Nv, &Nz, &PIXEL_WIDTH, &LENGTH_V);

        gsl_set_error_handler_off();

        // Read R values
        // These values are FWHM integer
        // spectrograph_windowfn_params takes 1 sigma km/s
        FILE *toRead = open_file(FNAME_RLIST, "r");
        fscanf(toRead, "%d\n", &NUMBER_OF_Rs);

        printf("Number of R values: %d\n", NUMBER_OF_Rs);

        R_VALUES = new int[NUMBER_OF_Rs];

        for (int r = 0; r < NUMBER_OF_Rs; ++r)
            fscanf(toRead, "%d\n", &R_VALUES[r]);

        fclose(toRead);
        // Reading R values done
        // ---------------------

        double  z_first  = ZBIN_CENTERS[0] - Z_BIN_WIDTH, \
                z_length = Z_BIN_WIDTH * (NUMBER_OF_Z_BINS+1);

        #if defined(_OPENMP)
        omp_set_dynamic(0); // Turn off dynamic threads
        numthreads = omp_get_max_threads();
        #endif

#pragma omp parallel private(buf, time_spent_table_sfid, time_spent_table_q)
{       
        #if defined(_OPENMP)
        t_rank = omp_get_thread_num();
        #endif
        
        struct spectrograph_windowfn_params     win_params             = {0, 0, PIXEL_WIDTH, 0};
        struct sq_integrand_params              integration_parameters = {&FIDUCIAL_PD13_PARAMS, &win_params};
        double *big_temp_array;

        // Integrate fiducial signal matrix
        FourierIntegrator s_integrator(GSL_INTEG_COSINE, signal_matrix_integrand, &integration_parameters);

        // Skip this section if fiducial signal matrix is turned off.
        if (TURN_OFF_SFID) goto DERIVATIVE;
        
        // Allocate memory to store results
        big_temp_array = new double[Nv * Nz];

        #pragma omp for nowait
        for (int r = 0; r < NUMBER_OF_Rs; r++)
        {
            time_spent_table_sfid = get_time();

            // Convert integer FWHM to 1 sigma km/s
            win_params.spectrograph_res = SPEED_OF_LIGHT / R_VALUES[r] / ONE_SIGMA_2_FWHM;
            
            printf("T%d/%d - Creating look up table for signal matrix. R = %d : %.2f km/s.\n", \
                    t_rank, numthreads, R_VALUES[r], win_params.spectrograph_res);
            fflush(stdout);

            STableFileNameConvention(buf, OUTPUT_DIR, OUTPUT_FILEBASE_S, R_VALUES[r]);
            
            if (!force_rewrite && file_exists(buf))
            {
                printf("File %s already exists. Skip to next.\n", buf);
                fflush(stdout);
                continue;
            }

            for (int xy = 0; xy < Nv*Nz; ++xy)
            {
                // xy = nv + Nv * nz
                int nz = xy / Nv, nv = xy % Nv;

                win_params.delta_v_ij = getLinearlySpacedValue(0, LENGTH_V, Nv, nv);        // 0 + LENGTH_V * nv / (Nv - 1.);
                win_params.z_ij       = getLinearlySpacedValue(z_first, z_length, Nz, nz);  // z_first + z_length * nz / (Nz - 1.);  
                
                s_integrator.setTableParameters(win_params.delta_v_ij, 10.);
                big_temp_array[xy]    = s_integrator.evaluate0ToInfty();
            }

            SQLookupTableFile signal_table(buf, 'w');

            signal_table.setHeader( Nv, Nz, LENGTH_V, z_length, \
                                    R_VALUES[r], PIXEL_WIDTH, \
                                    0, KBAND_EDGES[NUMBER_OF_K_BANDS]);

            signal_table.writeData(big_temp_array);

            time_spent_table_sfid = get_time() - time_spent_table_sfid;

            printf("T:%d/%d - Time spent on fiducial signal matrix table R %d is %.2f mins.\n", \
                    t_rank, numthreads, R_VALUES[r], time_spent_table_sfid);
        }

        delete [] big_temp_array;

        // S matrices are written.
        // ---------------------

DERIVATIVE:
        // Integrate derivative matrices
        // int subNz = Nz / NUMBER_OF_Z_BINS;
        FourierIntegrator q_integrator(GSL_INTEG_COSINE, q_matrix_integrand, &integration_parameters);

        big_temp_array = new double[Nv];

        #pragma omp for
        for (int r = 0; r < NUMBER_OF_Rs; r++)
        {
            time_spent_table_q = get_time();

            win_params.spectrograph_res = SPEED_OF_LIGHT / R_VALUES[r] / ONE_SIGMA_2_FWHM;
            printf("T:%d/%d - Creating look up tables for derivative signal matrices. R = %d : %.2f km/s.\n", \
                    t_rank, numthreads, R_VALUES[r], win_params.spectrograph_res);
            fflush(stdout);

            for (int kn = 0; kn < NUMBER_OF_K_BANDS; ++kn)
            {
                double kvalue_1 = KBAND_EDGES[kn];
                double kvalue_2 = KBAND_EDGES[kn + 1];

                printf("Q matrix for k = [%.1e - %.1e] s/km.\n", kvalue_1, kvalue_2);

                QTableFileNameConvention(buf, OUTPUT_DIR, OUTPUT_FILEBASE_Q, R_VALUES[r], kvalue_1, kvalue_2);
                
                if (!force_rewrite && file_exists(buf))
                {
                    printf("File %s already exists. Skip to next.\n", buf);
                    fflush(stdout);
                    continue;
                }

                for (int nv = 0; nv < Nv; ++nv)
                {
                    win_params.delta_v_ij = getLinearlySpacedValue(0, LENGTH_V, Nv, nv);

                    big_temp_array[nv] = q_integrator.evaluate(win_params.delta_v_ij, kvalue_1, kvalue_2, 0.);
                }

                SQLookupTableFile derivative_signal_table(buf, 'w');

                derivative_signal_table.setHeader(  Nv, 0, LENGTH_V, Z_BIN_WIDTH, \
                                                    R_VALUES[r], PIXEL_WIDTH, \
                                                    kvalue_1, kvalue_2);
                
                derivative_signal_table.writeData(big_temp_array);
            }
            
            time_spent_table_q = get_time() - time_spent_table_q;
            printf("T:%d/%d - Time spent on derivative matrix table R %d is %.2f mins.\n", \
                    t_rank, numthreads, R_VALUES[r], time_spent_table_q);
        }
        // Q matrices are written.
        // ---------------------

        delete [] big_temp_array;
}
        clean_up_bins();       
    }
    catch (std::exception& e)
    {
        printf("%s\n", e.what());
        return -1;
    }
    catch (const char* msg)
    {   
        printf("%s\n", msg);
        return -1;
    }

    return 0;
}










