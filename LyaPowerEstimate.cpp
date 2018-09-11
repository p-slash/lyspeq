#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "core/global_numbers.hpp"
#include "core/quadratic_estimate.hpp"
#include "core/spectrograph_functions.hpp"
#include "core/sq_table.hpp"

#include "io/config_file.hpp"
#include "io/io_helper_functions.hpp"

int main(int argc, char const *argv[])
{
    const char *FNAME_CONFIG = argv[1];
    
    char FNAME_LIST[300], \
         FNAME_RLIST[300], \
         INPUT_DIR[300], \
         FILEBASE_S[300], FILEBASE_Q[300], \
         OUTPUT_DIR[300], \
         OUTPUT_FILEBASE[300],\
         buf[500];

    int N_KLIN_BIN, N_KLOG_BIN, \
        NUMBER_OF_ITERATIONS;

    double  K_0, LIN_K_SPACING, LOG_K_SPACING, \
            Z_0;

    OneDQuadraticPowerEstimate *qps;
    
    struct palanque_fit_params FIDUCIAL_PD13_PARAMS;

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
        cFile.addKey("NumberOfRedshiftBins", &NUMBER_OF_Z_BINS, INTEGER);

        // cFile.addKey("PolynomialDegree", &POLYNOMIAL_FIT_DEGREE, INTEGER);
        
        cFile.addKey("SignalLookUpTableBase", FILEBASE_S, STRING);
        cFile.addKey("DerivativeSLookUpTableBase", FILEBASE_Q, STRING);

        cFile.addKey("FileNameList", FNAME_LIST, STRING);
        cFile.addKey("FileNameRList", FNAME_RLIST, STRING);
        cFile.addKey("FileInputDir", INPUT_DIR, STRING);

        cFile.addKey("OutputDir", OUTPUT_DIR, STRING);
        cFile.addKey("OutputFileBase", OUTPUT_FILEBASE, STRING);

        cFile.addKey("NumberOfIterations", &NUMBER_OF_ITERATIONS, INTEGER);

        // Fiducial Palanque fit function parameters
        cFile.addKey("FiducialAmplitude",           &FIDUCIAL_PD13_PARAMS.A,     DOUBLE);
        cFile.addKey("FiducialSlope",               &FIDUCIAL_PD13_PARAMS.n,     DOUBLE);
        cFile.addKey("FiducialCurvature",           &FIDUCIAL_PD13_PARAMS.alpha, DOUBLE);
        cFile.addKey("FiducialRedshiftPower",       &FIDUCIAL_PD13_PARAMS.B,     DOUBLE);
        cFile.addKey("FiducialRedshiftCurvature",   &FIDUCIAL_PD13_PARAMS.beta,  DOUBLE);
        cFile.readAll();

        // Construct k edges
        NUMBER_OF_K_BANDS = N_KLIN_BIN + N_KLOG_BIN;

        KBAND_EDGES   = new double[NUMBER_OF_K_BANDS + 1];
        KBAND_CENTERS = new double[NUMBER_OF_K_BANDS];

        for (int i = 0; i < N_KLIN_BIN + 1; i++)
        {
            KBAND_EDGES[i] = K_0 + LIN_K_SPACING * i;
        }
        for (int i = 1, j = N_KLIN_BIN + 1; i < N_KLOG_BIN + 1; i++, j++)
        {
            KBAND_EDGES[j] = KBAND_EDGES[N_KLIN_BIN] * pow(10., i * LOG_K_SPACING);
        }
        for (int kn = 0; kn < NUMBER_OF_K_BANDS; kn++)
        {
            KBAND_CENTERS[kn] = (KBAND_EDGES[kn] + KBAND_EDGES[kn + 1]) / 2.;
        }

        // Construct redshift bins
        ZBIN_CENTERS = new double[NUMBER_OF_Z_BINS];

        for (int zm = 0; zm < NUMBER_OF_Z_BINS; ++zm)
        {
            ZBIN_CENTERS[zm] = Z_0 + Z_BIN_WIDTH * zm;
        }

        SQLookupTable sq_Table(INPUT_DIR, FILEBASE_S, FILEBASE_Q, FNAME_RLIST);

        gsl_set_error_handler_off();

        qps = new OneDQuadraticPowerEstimate(   FNAME_LIST, INPUT_DIR, \
                                                &sq_Table, \
                                                &FIDUCIAL_PD13_PARAMS);

        // qps->setInitialScaling();
        // qps->setInitialPSestimateFFT();
        // sprintf(buf, "%s/%s_qso_fft_power_estimate.dat", OUTPUT_DIR, OUTPUT_FILEBASE);
        // qps->write_spectrum_estimate(buf);

        qps->iterate(NUMBER_OF_ITERATIONS);

        sprintf(buf, "%s/%s", OUTPUT_DIR, OUTPUT_FILEBASE);
        qps->write_spectrum_estimates(buf);

        delete qps;
        delete [] KBAND_EDGES;
        delete [] KBAND_CENTERS;
        delete [] ZBIN_CENTERS;
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
            sprintf(buf, "%s/%s", OUTPUT_DIR, OUTPUT_FILEBASE);
            qps->printfSpectra();
            qps->write_spectrum_estimates(buf);

            delete qps;
        }
        
        printf("%s\n", msg);
        return -1;
    }

    return 0;
}










