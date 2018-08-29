/* Q matrices are not scaled with redshift binning function
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "core/spectrograph_functions.hpp"
#include "core/fiducial_cosmology.hpp"

#include "gsltools/integrator.hpp"

#include "io/config_file.hpp"
#include "io/io_helper_functions.hpp"
#include "io/sq_lookup_table_file.hpp"

#define SPEED_OF_LIGHT 299792.458

int main(int argc, char const *argv[])
{
    const char *FNAME_CONFIG = argv[1];
    
    char FNAME_RLIST[300], \
         OUTPUT_DIR[300], \
         OUTPUT_FILEBASE_S[300],\
         OUTPUT_FILEBASE_Q[300],\
         buf[500];

    int N_KLIN_BIN, N_KLOG_BIN, N_KTOTAL_BINS, \
        N_Z_BINS, \
        NUMBER_OF_Rs, *R_VALUES, \
        Nv, Nz;

    double  K_0, LIN_K_SPACING, LOG_K_SPACING, *k_edges, \
            Z_0, Z_BIN_WIDTH, *z_centers, \
            PIXEL_WIDTH, LENGTH_V;

    try
    {
        // Set up config file to read variables.
        ConfigFile cFile(FNAME_CONFIG);

        cFile.addKey("K0", &K_0, DOUBLE);
        cFile.addKey("FirstRedshiftBinCenter", &Z_0, DOUBLE);

        cFile.addKey("LinearKBinWidth", &LIN_K_SPACING, DOUBLE);
        cFile.addKey("Log10KBinWidth", &LOG_K_SPACING, DOUBLE);
        cFile.addKey("RedshiftBinWidth", &Z_BIN_WIDTH, DOUBLE);

        cFile.addKey("NumberOfLinearBins", &N_KLIN_BIN, INTEGER);
        cFile.addKey("NumberOfLog10Bins", &N_KLOG_BIN, INTEGER);
        cFile.addKey("NumberOfRedshiftBins", &N_Z_BINS, INTEGER);
        
        cFile.addKey("FileNameRList", FNAME_RLIST, STRING);
        cFile.addKey("FileInputDir", OUTPUT_DIR, STRING);

        cFile.addKey("SignalLookUpTableBase", &OUTPUT_FILEBASE_S, STRING);
        cFile.addKey("DerivativeSLookUpTableBase", &OUTPUT_FILEBASE_Q, STRING);

        cFile.addKey("NumberVPoints", &Nv, INTEGER);
        cFile.addKey("NumberZPoints", &Nz, INTEGER);
        cFile.addKey("PixelWidth", &PIXEL_WIDTH, DOUBLE);
        cFile.addKey("VelocityLength", &LENGTH_V, DOUBLE);

        cFile.readAll();

        // Construct k edges
        N_KTOTAL_BINS = N_KLIN_BIN + N_KLOG_BIN;

        k_edges = new double[N_KTOTAL_BINS + 1];

        for (int i = 0; i < N_KLIN_BIN + 1; i++)
        {
            k_edges[i] = K_0 + LIN_K_SPACING * i;
        }
        for (int i = 1, j = N_KLIN_BIN + 1; i < N_KLOG_BIN + 1; i++, j++)
        {
            k_edges[j] = k_edges[N_KLIN_BIN] * pow(10., i * LOG_K_SPACING);
        }

        // Construct redshift bins
        z_centers = new double[N_Z_BINS];

        for (int zm = 0; zm < N_Z_BINS; ++zm)
        {
            z_centers[zm] = Z_0 + Z_BIN_WIDTH * zm;
        }

        for (int i = 0; i < N_KTOTAL_BINS + 1; ++i)
        {
            printf("%le ", k_edges[i]);
        }
        printf("\n");
        // Redshift and wavenumber bins are constructed
        // ---------------------

        // gsl_set_error_handler_off();

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

        // Integrate signal matrix
        printf("Creating look up table for signal matrix...\n");
        fflush(stdout);

        double  z_first = Z_0 - Z_BIN_WIDTH / 2., \
                z_length = Z_BIN_WIDTH * N_Z_BINS;

        struct spectrograph_windowfn_params win_params = {0, 0, PIXEL_WIDTH, 0};

        Integrator s_integrator(GSL_QAG, signal_matrix_integrand, &win_params);

        // Allocate memory to store results
        double *big_temp_array = new double[Nv * Nz];

        for (int r = 0; r < NUMBER_OF_Rs; ++r)
        {
            win_params.spectrograph_res = SPEED_OF_LIGHT / R_VALUES[r];
            printf("%d of %d R values %d => %.2f km/s\n", r+1, NUMBER_OF_Rs, R_VALUES[r], win_params.spectrograph_res);

            for (int xy = 0; xy < Nv*Nz; ++xy)
            {
                // xy = nv + Nv * nz
                int nz = xy / Nv;
                int nv = xy % Nv;

                win_params.delta_v_ij = getLinearlySpacedValue(0, LENGTH_V, Nv, nv);        // LENGTH_V * nv / (Nv - 1.);
                win_params.z_ij       = getLinearlySpacedValue(z_first, z_length, Nz, nz);  // z_first + z_length * nz / (double) Nz;

                big_temp_array[xy] = s_integrator.evaluateAToInfty(0);
            }
            STableFileNameConvention(buf, OUTPUT_DIR, OUTPUT_FILEBASE_S, R_VALUES[r]);

            SQLookupTableFile signal_table(buf, 'w');

            signal_table.setHeader( Nv, Nz, LENGTH_V, z_length, \
                                    R_VALUES[r], PIXEL_WIDTH, \
                                    0, k_edges[N_KTOTAL_BINS]);

            signal_table.writeData(big_temp_array);
        }

        delete [] big_temp_array;
        // S matrices are written.
        // ---------------------

        // Integrate derivative matrices
        printf("Creating look up table for derivative signal matrices...\n");
        fflush(stdout);

        // int subNz = Nz / N_Z_BINS;
        Integrator q_integrator(GSL_QAG, q_matrix_integrand, &win_params);

        big_temp_array = new double[Nv];
        // double *temp_array_zscaled = new double[Nv * subNz];
        double kvalue_1, kvalue_2;

        for (int r = 0; r < NUMBER_OF_Rs; ++r)
        {
            win_params.spectrograph_res = SPEED_OF_LIGHT / R_VALUES[r];
            printf("%d of %d R values %d => %.2f km/s\n", r+1, NUMBER_OF_Rs, R_VALUES[r], win_params.spectrograph_res);

            for (int kn = 0; kn < N_KTOTAL_BINS; ++kn)
            {
                kvalue_1 = k_edges[kn];
                kvalue_2 = k_edges[kn + 1];

                printf("Q matrix for k = [%.1e - %.1e] s/km.\n", kvalue_1, kvalue_2);

                for (int nv = 0; nv < Nv; ++nv)
                {
                    win_params.delta_v_ij = getLinearlySpacedValue(0, LENGTH_V, Nv, nv);

                    big_temp_array[nv] = q_integrator.evaluate(kvalue_1, kvalue_2);
                }

                QTableFileNameConvention(buf, OUTPUT_DIR, OUTPUT_FILEBASE_Q, R_VALUES[r], kvalue_1, kvalue_2);

                SQLookupTableFile derivative_signal_table(buf, 'w');

                derivative_signal_table.setHeader(  Nv, 0, LENGTH_V, Z_BIN_WIDTH, \
                                                    R_VALUES[r], PIXEL_WIDTH, \
                                                    kvalue_1, kvalue_2);
                
                derivative_signal_table.writeData(big_temp_array);

                // for (int zm = 0; zm < N_Z_BINS; ++zm)
                // {
                //     printf("Q matrix for k = [%.1e - %.1e] s/km. Now scaling for triangular redshift bin %.1f.\n", kvalue_1, kvalue_2, z_centers[zm]);
                    
                //     for (int xy = 0; xy < Nv*subNz; ++xy)
                //     {
                //         // xy = nz + Nv * nv
                //         int nz = xy % Nv;

                //         double z_ij = getLinearlySpacedValue(z_centers[zm] - Z_BIN_WIDTH/2., Z_BIN_WIDTH, subNz, nz); 
                //         // (z_centers[zm] - Z_BIN_WIDTH/2.) + Z_BIN_WIDTH * nz / (double) subNz;

                //         temp_array_zscaled[xy] = triangular_z_bin(z_ij, z_centers[zm], Z_BIN_WIDTH) * big_temp_array[xy];
                //     }

                //     QTableFileNameConvention(buf, OUTPUT_DIR, OUTPUT_FILEBASE_Q, R_VALUES[r], kvalue_1, kvalue_2, z_centers[zm]);

                //     SQLookupTableFile derivative_signal_table(buf, 'w');

                //     derivative_signal_table.setHeader(  Nv, subNz, LENGTH_V, Z_BIN_WIDTH, \
                //                                         R_VALUES[r], PIXEL_WIDTH, \
                //                                         z_centers[zm], kvalue_1, kvalue_2);
                    
                //     derivative_signal_table.writeData(temp_array_zscaled);
                // }
            }
        }
        // Q matrices are written.
        // ---------------------

        delete [] big_temp_array;
        delete [] k_edges;
        delete [] z_centers;
       
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










