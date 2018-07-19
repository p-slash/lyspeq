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

    int NBin, NUMBER_OF_ITERATIONS, LINEAR_LOG;
    double K_0, K_1, *k_edges;

    OneDQuadraticPowerEstimate *qps;

    try
    {
        // Set up config file to read variables.
        ConfigFile cFile(FNAME_CONFIG);

        cFile.addKey("K0", &K_0, DOUBLE);
        cFile.addKey("K1", &K_1, DOUBLE);
        cFile.addKey("NumberOfBins", &NBin, INTEGER);
        cFile.addKey("LinLog", &LINEAR_LOG, INTEGER);

        cFile.addKey("PolynomialDegree", &POLYNOMIAL_FIT_DEGREE, INTEGER);
        
        // cFile.addKey("SpectographRes", &R_SPECTOGRAPH, DOUBLE);
        
        cFile.addKey("FileNameList", FNAME_LIST, STRING);
        cFile.addKey("FileInputDir", INPUT_DIR, STRING);

        cFile.addKey("OutputDir", OUTPUT_DIR, STRING);
        cFile.addKey("OutputFileBase", OUTPUT_FILEBASE, STRING);

        cFile.addKey("NumberOfIterations", &NUMBER_OF_ITERATIONS, INTEGER);

        cFile.readAll();

        // Construct k edges
        k_edges = new double[NBin + 1];

        for (int i = 0; i < NBin + 1; i++)
        {
            k_edges[i] = K_0 + K_1 * i;

            if (LINEAR_LOG == 10)
                k_edges[i] = pow(10., k_edges[i]);
        }
        
        if (LINEAR_LOG == 10)
            printf("Using log spaced k bands:\n");
        else
            printf("Using linearly spaced k bands:\n");
        
        for (int i = 0; i < NBin + 1; ++i)
        {
            printf("%le ", k_edges[i]);
        }
        printf("\n");
        gsl_set_error_handler_off();

        qps = new OneDQuadraticPowerEstimate(FNAME_LIST, INPUT_DIR, NBin, k_edges);

        qps->setInitialScaling();
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










