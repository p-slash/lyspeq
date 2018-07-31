#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "core/quadratic_estimate.hpp"
#include "core/spectrograph_functions.hpp"
#include "io/config_file.hpp"
#include "io/io_helper_functions.hpp"

int main(int argc, char const *argv[])
{
    const char *FNAME_CONFIG = argv[1];
    
    char FNAME_LIST[300], \
         INPUT_DIR[300], \
         OUTPUT_DIR[300], \
         OUTPUT_FILEBASE[300],\
         buf[500];

    int N_LIN_BIN, N_LOG_BIN, N_TOTAL_BINS, NUMBER_OF_ITERATIONS;
    double K_0, LIN_K_SPACING, LOG_K_SPACING, *k_edges;

    OneDQuadraticPowerEstimate *qps;

    try
    {
        // Set up config file to read variables.
        ConfigFile cFile(FNAME_CONFIG);

        cFile.addKey("K0", &K_0, DOUBLE);

        cFile.addKey("LinearKBinWidth", &LIN_K_SPACING, DOUBLE);
        cFile.addKey("Log10KBinWidth", &LOG_K_SPACING, DOUBLE);

        cFile.addKey("NumberOfLinearBins", &N_LIN_BIN, INTEGER);
        cFile.addKey("NumberOfLog10Bins", &N_LOG_BIN, INTEGER);

        // cFile.addKey("LinLog", &LINEAR_LOG, INTEGER);

        cFile.addKey("PolynomialDegree", &POLYNOMIAL_FIT_DEGREE, INTEGER);
        
        // cFile.addKey("SpectographRes", &R_SPECTOGRAPH, DOUBLE);
        
        cFile.addKey("FileNameList", FNAME_LIST, STRING);
        cFile.addKey("FileInputDir", INPUT_DIR, STRING);

        cFile.addKey("OutputDir", OUTPUT_DIR, STRING);
        cFile.addKey("OutputFileBase", OUTPUT_FILEBASE, STRING);

        cFile.addKey("NumberOfIterations", &NUMBER_OF_ITERATIONS, INTEGER);

        cFile.readAll();

        // Construct k edges
        N_TOTAL_BINS = N_LIN_BIN + N_LOG_BIN;

        k_edges = new double[N_TOTAL_BINS + 1];

        for (int i = 0; i < N_LIN_BIN + 1; i++)
        {
            k_edges[i] = K_0 + LIN_K_SPACING * i;
        }
        for (int i = 1, j = N_LIN_BIN + 1; i < N_LOG_BIN + 1; i++, j++)
        {
            k_edges[j] = k_edges[N_LIN_BIN] * pow(10., i * LOG_K_SPACING);
        }

        // for (int i = 0; i < N_TOTAL_BINS + 1; i++)
        // {
        //     k_edges[i] = K_0 + K_1 * i;

        //     if (LINEAR_LOG == 10)
        //         k_edges[i] = pow(10., k_edges[i]);
        // }
        
        // if (LINEAR_LOG == 10)
        //     printf("Using log spaced k bands:\n");
        // else
        //     printf("Using linearly spaced k bands:\n");
        
        for (int i = 0; i < N_TOTAL_BINS + 1; ++i)
        {
            printf("%le ", k_edges[i]);
        }
        printf("\n");
        gsl_set_error_handler_off();

        qps = new OneDQuadraticPowerEstimate(FNAME_LIST, INPUT_DIR, N_TOTAL_BINS, k_edges);

        // qps->setInitialScaling();
        // qps->setInitialPSestimateFFT();
        // sprintf(buf, "%s/%s_qso_fft_power_estimate.dat", OUTPUT_DIR, OUTPUT_FILEBASE);
        // qps->write_spectrum_estimate(buf);

        qps->iterate(NUMBER_OF_ITERATIONS);

        sprintf(buf, "%s/%s_qso_power_estimate.dat", OUTPUT_DIR, OUTPUT_FILEBASE);
        qps->write_spectrum_estimate(buf);

        delete qps;
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
            sprintf(buf, "%s/%s_qso_power_estimate.dat", OUTPUT_DIR, OUTPUT_FILEBASE);
            qps->write_spectrum_estimate(buf);

            delete qps;
        }
        
        printf("%s\n", msg);
        return -1;
    }

    return 0;
}










