#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <gsl/gsl_errno.h>

#include "core/global_numbers.hpp"
#include "core/quadratic_estimate.hpp"
#include "core/spectrograph_functions.hpp"
#include "core/sq_table.hpp"

#include "io/config_file.hpp"
#include "io/io_helper_functions.hpp"

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
         buf[500];

    int N_KLIN_BIN, N_KLOG_BIN, \
        NUMBER_OF_ITERATIONS;

    double  K_0, LIN_K_SPACING, LOG_K_SPACING, \
            Z_0;
    
    struct palanque_fit_params FIDUCIAL_PD13_PARAMS;

    OneDQuadraticPowerEstimate *qps = NULL;

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
        cFile.addKey("FiducialLorentzianLambda",    &FIDUCIAL_PD13_PARAMS.lambda,  DOUBLE);

        // Read integer if testing outside of Lya region
        int out_lya;
        cFile.addKey("TurnOffBaseline", &out_lya, INTEGER);

        cFile.readAll();

        TURN_OFF_SFID = out_lya > 0;

        if (TURN_OFF_SFID)  printf("Fiducial signal matrix is turned off.\n");
        
        // Redshift and wavenumber bins are constructed
        set_up_bins(K_0, N_KLIN_BIN, LIN_K_SPACING, N_KLOG_BIN, LOG_K_SPACING, Z_0);

        sq_lookup_table = new SQLookupTable(INPUT_DIR, FILEBASE_S, FILEBASE_Q, FNAME_RLIST);

        gsl_set_error_handler_off();

        qps = new OneDQuadraticPowerEstimate(   FNAME_LIST, INPUT_DIR, \
                                                &FIDUCIAL_PD13_PARAMS);

        sprintf(buf, "%s/%s", OUTPUT_DIR, OUTPUT_FILEBASE);
        qps->iterate(NUMBER_OF_ITERATIONS, buf);

        delete qps;
        delete sq_lookup_table;

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










