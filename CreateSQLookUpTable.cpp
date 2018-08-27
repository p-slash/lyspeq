#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "core/spectrograph_functions.hpp"
#include "core/fiducial_cosmology.hpp"

#include "gsltools/integrator.hpp"

#include "io/config_file.hpp"
#include "io/io_helper_functions.hpp"
#include "io/sq_lookup_table.hpp"

#define SPEED_OF_LIGHT 299792.458

double triangular_z_bin(double z, double zm, double deltaz)
{
    if (zm - deltaz < z && z < zm)
    {
        return (z - zm + deltaz) / deltaz;
    }
    else if (zm < z && z < zm + deltaz)
    {
        return (zm + deltaz - z) / deltaz;
    }

    return 0;
}

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
                // xy = nz + Nv * nv
                int nv = xy / Nv;
                int nz = xy % Nv;

                double v_ij = LENGTH_V * nv / (double) Nv;
                double z_ij = z_first + z_length * nz / (double) Nz;

                win_params.delta_v_ij = v_ij;
                win_params.z_ij       = z_ij;

                big_temp_array[xy] = s_integrator.evaluateAToInfty(0);
            }
            sprintf(buf, "%s/%s_R%d.dat", OUTPUT_DIR, OUTPUT_FILEBASE_S, R_VALUES[r]);

            SQLookupTable signal_table(buf, 'w');
            signal_table.setHeader(Nv, Nz, R_VALUES[r], PIXEL_WIDTH, z_centers[N_Z_BINS/2], 0, k_edges[N_KTOTAL_BINS]);
            signal_table.writeData(big_temp_array);
        }

        delete [] big_temp_array;

        // Integrate derivative matrices
        printf("Creating look up table for derivative signal matrices...\n");
        fflush(stdout);

        int subNz = Nz / N_Z_BINS;
        Integrator q_integrator(GSL_QAG, q_matrix_integrand, &win_params);

        big_temp_array = new double[Nv * subNz];
        double *temp_array_zscaled = new double[Nv * subNz];
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

                for (int xy = 0; xy < Nv*subNz; ++xy)
                {
                    // xy = nz + Nv * nv
                    int nv = xy / Nv;

                    double v_ij = LENGTH_V * nv / (double) Nv;

                    win_params.delta_v_ij = v_ij;

                    big_temp_array[xy] = q_integrator.evaluate(kvalue_1, kvalue_2);
                }

                for (int zm = 0; zm < N_Z_BINS; ++zm)
                {
                    printf("Q matrix for k = [%.1e - %.1e] s/km. Now scaling for triangular redshift bin %.1f.\n", kvalue_1, kvalue_2, z_centers[zm]);
                    
                    for (int xy = 0; xy < Nv*subNz; ++xy)
                    {
                        // xy = nz + Nv * nv
                        int nz = xy % Nv;

                        double z_ij = (z_centers[zm] - Z_BIN_WIDTH/2.) + Z_BIN_WIDTH * nz / (double) subNz;

                        temp_array_zscaled[xy] = triangular_z_bin(z_ij, z_centers[zm], Z_BIN_WIDTH) * big_temp_array[xy];
                    }

                    sprintf(buf, "%s/%s_R%d_k%.1e_%.1e_z%.1f.dat", OUTPUT_DIR, OUTPUT_FILEBASE_Q, R_VALUES[r], kvalue_1, kvalue_2, z_centers[zm]);

                    SQLookupTable derivative_signal_table(buf, 'w');
                    derivative_signal_table.setHeader(Nv, subNz, R_VALUES[r], PIXEL_WIDTH, z_centers[zm], kvalue_1, kvalue_2);
                    derivative_signal_table.writeData(temp_array_zscaled);
                }
            }
        }

        delete [] big_temp_array;
        delete [] temp_array_zscaled;
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










