/* Q matrices are not scaled with redshift binning function
 */
#include <cstdio>
#include <cstdlib>
#include <cstring> // strcmp
#include <ctime>    /* clock_t, clock, CLOCKS_PER_SEC */

#include <gsl/gsl_errno.h>

#include "core/global_numbers.hpp"
#include "core/spectrograph_functions.hpp"
#include "core/fiducial_cosmology.hpp"

#include "gsltools/integrator.hpp"
#include "gsltools/fourier_integrator.hpp"

#include "io/config_file.hpp"
#include "io/io_helper_functions.hpp"
#include "io/sq_lookup_table_file.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

#define SPEED_OF_LIGHT 299792.458

int main(int argc, char const *argv[])
{
    const char *FNAME_CONFIG = argv[1];
    bool force_rewrite = true;
    clock_t t;
    float time_spent_table_sfid, time_spent_table_q;

    if (argc == 3)
    {
        force_rewrite = !(strcmp(argv[2], "--unforce") == 0);
    }

    char FNAME_RLIST[300], \
         OUTPUT_DIR[300], \
         OUTPUT_FILEBASE_S[300],\
         OUTPUT_FILEBASE_Q[300],\
         buf[500];

    int N_KLIN_BIN, N_KLOG_BIN, \
        NUMBER_OF_Rs, *R_VALUES, \
        Nv, Nz;

    double  K_0, LIN_K_SPACING, LOG_K_SPACING, \
            Z_0, \
            PIXEL_WIDTH, LENGTH_V;

    struct palanque_fit_params FIDUCIAL_PD13_PARAMS;

    try
    {
        // Set up config file to read variables.
        ConfigFile cFile(FNAME_CONFIG);

        // Bin parameters
        cFile.addKey("K0", &K_0, DOUBLE);
        cFile.addKey("FirstRedshiftBinCenter", &Z_0, DOUBLE);

        cFile.addKey("LinearKBinWidth",  &LIN_K_SPACING, DOUBLE);
        cFile.addKey("Log10KBinWidth",   &LOG_K_SPACING, DOUBLE);
        cFile.addKey("RedshiftBinWidth", &Z_BIN_WIDTH,   DOUBLE);

        cFile.addKey("NumberOfLinearBins",   &N_KLIN_BIN, INTEGER);
        cFile.addKey("NumberOfLog10Bins",    &N_KLOG_BIN, INTEGER);
        cFile.addKey("NumberOfRedshiftBins", &NUMBER_OF_Z_BINS,   INTEGER);
        
        // File names and paths
        cFile.addKey("FileNameRList", FNAME_RLIST, STRING);
        cFile.addKey("FileInputDir",  OUTPUT_DIR,  STRING);

        cFile.addKey("SignalLookUpTableBase",       &OUTPUT_FILEBASE_S, STRING);
        cFile.addKey("DerivativeSLookUpTableBase",  &OUTPUT_FILEBASE_Q, STRING);

        // Integration grid parameters
        cFile.addKey("NumberVPoints",   &Nv, INTEGER);
        cFile.addKey("NumberZPoints",   &Nz, INTEGER);
        cFile.addKey("PixelWidth",      &PIXEL_WIDTH, DOUBLE);
        cFile.addKey("VelocityLength",  &LENGTH_V,    DOUBLE);

        // Fiducial Palanque fit function parameters
        cFile.addKey("FiducialAmplitude",           &FIDUCIAL_PD13_PARAMS.A,     DOUBLE);
        cFile.addKey("FiducialSlope",               &FIDUCIAL_PD13_PARAMS.n,     DOUBLE);
        cFile.addKey("FiducialCurvature",           &FIDUCIAL_PD13_PARAMS.alpha, DOUBLE);
        cFile.addKey("FiducialRedshiftPower",       &FIDUCIAL_PD13_PARAMS.B,     DOUBLE);
        cFile.addKey("FiducialRedshiftCurvature",   &FIDUCIAL_PD13_PARAMS.beta,  DOUBLE);
        cFile.addKey("FiducialLorentzianLambda",    &FIDUCIAL_PD13_PARAMS.lambda,  DOUBLE);

        // Read integer if testing outside of Lya region
        int out_lya;
        cFile.addKey("TurnOffBaseline", &out_lya, INTEGER);

        cFile.readAll();

        TURN_OFF_SFID = out_lya > 0;

        if (TURN_OFF_SFID)  printf("Fiducial signal matrix is turned off.\n");

        // Redshift and wavenumber bins are constructed
        set_up_bins(K_0, N_KLIN_BIN, LIN_K_SPACING, N_KLOG_BIN, LOG_K_SPACING, Z_0);

        gsl_set_error_handler_off();

        // Read R values
        FILE *toRead = open_file(FNAME_RLIST, "r");
        fscanf(toRead, "%d\n", &NUMBER_OF_Rs);

        printf("Number of R values: %d\n", NUMBER_OF_Rs);

        R_VALUES = new int[NUMBER_OF_Rs];

        for (int r = 0; r < NUMBER_OF_Rs; ++r)
        {
            fscanf(toRead, "%d\n", &R_VALUES[r]);
        }

        fclose(toRead);
        // Reading R values done
        // ---------------------

        // Integrate fiducial signal matrix

        printf("Creating look up table for signal matrix...\n");
        fflush(stdout);

        double  z_first  = Z_0 - Z_BIN_WIDTH / 2., \
                z_length = Z_BIN_WIDTH * NUMBER_OF_Z_BINS;

        t = clock();

#pragma omp parallel private(buf, time_spent_table_sfid, time_spent_table_q)
{
        struct spectrograph_windowfn_params     win_params             = {0, 0, PIXEL_WIDTH, 0};
        struct sq_integrand_params              integration_parameters = {&FIDUCIAL_PD13_PARAMS, &win_params};

        if (!TURN_OFF_SFID)
        {
            FourierIntegrator s_integrator(GSL_INTEG_COSINE, signal_matrix_integrand, &integration_parameters);

            // Allocate memory to store results
            double *big_temp_array = new double[Nv * Nz];

            #pragma omp for
            for (int r = 0; r < NUMBER_OF_Rs; r++)
            {
                win_params.spectrograph_res = SPEED_OF_LIGHT / R_VALUES[r] / ONE_SIGMA_2_FWHM;
                printf("%d of %d R values %d => %.2f km/s\n", \
                        r+1, NUMBER_OF_Rs, R_VALUES[r], win_params.spectrograph_res);
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
                    int nz = xy / Nv;
                    int nv = xy % Nv;

                    // LENGTH_V * nv / (Nv - 1.);
                    win_params.delta_v_ij = getLinearlySpacedValue(0, LENGTH_V, Nv, nv); 

                    s_integrator.setTableParameters(win_params.delta_v_ij, 10.);

                    // z_first + z_length * nz / (double) Nz;       
                    win_params.z_ij       = getLinearlySpacedValue(z_first, z_length, Nz, nz); 
                    
                    big_temp_array[xy]    = s_integrator.evaluate0ToInfty();
                }

                SQLookupTableFile signal_table(buf, 'w');

                signal_table.setHeader( Nv, Nz, LENGTH_V, z_length, \
                                        R_VALUES[r], PIXEL_WIDTH, \
                                        0, KBAND_EDGES[NUMBER_OF_K_BANDS]);

                signal_table.writeData(big_temp_array);
            }

            delete [] big_temp_array;

            t = clock() - t;
            time_spent_table_sfid = ((float) t) / CLOCKS_PER_SEC;

            // S matrices are written.
            // ---------------------
        }

        // Integrate derivative matrices
        t = clock();

        printf("Creating look up table for derivative signal matrices...\n");
        fflush(stdout);

        // int subNz = Nz / NUMBER_OF_Z_BINS;
        FourierIntegrator q_integrator(GSL_INTEG_COSINE, q_matrix_integrand, &integration_parameters);

        big_temp_array = new double[Nv];

        #pragma omp for
        for (int r = 0; r < NUMBER_OF_Rs; r++)
        {
            win_params.spectrograph_res = SPEED_OF_LIGHT / R_VALUES[r] / ONE_SIGMA_2_FWHM;
            printf("%d of %d R values %d => %.2f km/s\n", \
                    r+1, NUMBER_OF_Rs, R_VALUES[r], win_params.spectrograph_res);
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

                    big_temp_array[nv] = q_integrator.evaluate(win_params.delta_v_ij, kvalue_1, kvalue_2);
                }

                SQLookupTableFile derivative_signal_table(buf, 'w');

                derivative_signal_table.setHeader(  Nv, 0, LENGTH_V, Z_BIN_WIDTH, \
                                                    R_VALUES[r], PIXEL_WIDTH, \
                                                    kvalue_1, kvalue_2);
                
                derivative_signal_table.writeData(big_temp_array);
            }
        }
        
        t = clock() - t;
        time_spent_table_q = ((float) t) / CLOCKS_PER_SEC;

        // Q matrices are written.
        // ---------------------

        printf("Time spent on fiducial signal matrix table is %.2f mins.\n", time_spent_table_sfid / 60.);
        printf("Time spent on derivatibe matrix table is %.2f mins.\n", time_spent_table_q / 60.);

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










