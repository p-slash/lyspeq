#include "core/global_numbers.hpp"
#include "io/config_file.hpp"

#include <cstdio>
#include <cmath>

#if defined(_OPENMP)
#include <omp.h> /*omp_get_wtime();*/
#else
#include <ctime>    /* clock_t, clock, CLOCKS_PER_SEC */
#endif

char TMP_FOLDER[300] = ".";
Logger LOGGER;

double CHISQ_CONVERGENCE_EPS = 0.01;
double MEMORY_ALLOC          = 0;

int NUMBER_OF_K_BANDS, NUMBER_OF_Z_BINS, TOTAL_KZ_BINS;
int t_rank = 0, numthreads = 1;

SQLookupTable *sq_shared_table, *sq_private_table;

double *KBAND_EDGES, *KBAND_CENTERS;
double  Z_BIN_WIDTH, *ZBIN_CENTERS;

double   time_spent_on_c_inv    = 0, time_spent_on_f_inv   = 0;
double   time_spent_on_set_sfid = 0, time_spent_set_qs     = 0, \
         time_spent_set_modqs   = 0, time_spent_set_fisher = 0;

double   time_spent_on_q_interp = 0, time_spent_on_q_copy = 0;
long     number_of_times_called_setq = 0, number_of_times_called_setsfid = 0;

bool TURN_OFF_SFID;

void printf_time_spent_details()
{
    printf("Total time spent on inverting C is %.2f mins.\n", time_spent_on_c_inv);
    printf("Total time spent on inverting F is %.2f mins.\n", time_spent_on_f_inv);

    printf("Total time spent on setting Sfid is %.2f mins with %lu calls.\n", \
            time_spent_on_set_sfid, number_of_times_called_setsfid);
    printf("Total time spent on setting Qs is %.2f mins with %lu calls. \nInterpolation: %.2f and Copy: %.2f.\n", \
            time_spent_set_qs, number_of_times_called_setq, time_spent_on_q_interp, time_spent_on_q_copy);
    
    printf("Total time spent on setting Mod Qs is %.2f mins.\n", time_spent_set_modqs  );
    printf("Total time spent on setting F is %.2f mins.\n",      time_spent_set_fisher );
}

void print_build_specifics()
{
    #if defined(TOPHAT_Z_BINNING_FN)
    #define BINNING_SHAPE "Top Hat"
    #elif defined(TRIANGLE_Z_BINNING_FN)
    #define BINNING_SHAPE "Triangular"
    #else
    #define BINNING_SHAPE "ERROR NOT DEFINED"
    #endif

    #if defined(DEBUG_FIT_FUNCTION)
    #define FITTING_FUNC "Debug"
    #elif defined(PD13_FIT_FUNCTION)
    #define FITTING_FUNC "PD13"
    #else
    #define FITTING_FUNC "ERROR NOT DEFINED"
    #endif

    #define tostr(a) #a
    #define tovstr(a) tostr(a)
    
    #if defined(LAST_K_EDGE)
    #define HIGH_K_TXT tovstr(LAST_K_EDGE)
    #else
    #define HIGH_K_TXT "OFF"
    #endif

    printf("This version is build by the following options:\n");
    printf("1D Interpolation: %s\n", tovstr(INTERP_1D_TYPE));
    printf("2D Interpolation: %s\n", tovstr(INTERP_2D_TYPE));
    printf("Fitting function: %s\n", FITTING_FUNC);
    printf("Redshift binning shape: %s\n", BINNING_SHAPE);
    printf("Last k bin: %s\n", HIGH_K_TXT);
    fflush(stdout);

    #undef tostr
    #undef tovstr
    #undef BINNING_SHAPE
    #undef FITTING_FUNC
    #undef HIGH_K_TXT
}

void set_up_bins(double k0, int nlin, double dklin, \
                            int nlog, double dklog, \
                 double z0)
{
    // Construct k edges
    NUMBER_OF_K_BANDS = nlin + nlog;
    
    // One last bin is created when LAST_K_EDGE is set in Makefile.
    #ifdef LAST_K_EDGE
    NUMBER_OF_K_BANDS++;
    #endif

    TOTAL_KZ_BINS     = NUMBER_OF_K_BANDS * NUMBER_OF_Z_BINS;

    KBAND_EDGES   = new double[NUMBER_OF_K_BANDS + 1];
    KBAND_CENTERS = new double[NUMBER_OF_K_BANDS];

    // Linearly spaced bins
    for (int i = 0; i < nlin + 1; i++)
        KBAND_EDGES[i] = k0 + dklin * i;
    // Logarithmicly spaced bins
    for (int i = 1, j = nlin + 1; i < nlog + 1; i++, j++)
        KBAND_EDGES[j] = KBAND_EDGES[nlin] * pow(10., i * dklog);
    
    // Last bin
    #ifdef LAST_K_EDGE
    KBAND_EDGES[NUMBER_OF_K_BANDS] = LAST_K_EDGE;
    #endif

    // Set up k bin centers
    for (int kn = 0; kn < NUMBER_OF_K_BANDS; kn++)
        KBAND_CENTERS[kn] = (KBAND_EDGES[kn] + KBAND_EDGES[kn + 1]) / 2.;

    // Construct redshift bins
    ZBIN_CENTERS = new double[NUMBER_OF_Z_BINS];

    for (int zm = 0; zm < NUMBER_OF_Z_BINS; ++zm)
        ZBIN_CENTERS[zm] = z0 + Z_BIN_WIDTH * zm;
}

void clean_up_bins()
{
    delete [] KBAND_EDGES;
    delete [] KBAND_CENTERS;
    delete [] ZBIN_CENTERS;
}

double get_time()
{
    #if defined(_OPENMP)
    return omp_get_wtime() / 60.;
    #else
    clock_t t = clock();
    return ((double) t) / CLOCKS_PER_SEC / 60.;
    #endif
}

// Pass NULL for not needed variables!
void read_config_file(  const char *FNAME_CONFIG, \
                        pd13_fit_params &FIDUCIAL_PD13_PARAMS, \
                        char *FNAME_LIST, char *FNAME_RLIST, char *INPUT_DIR, char *OUTPUT_DIR, \
                        char *OUTPUT_FILEBASE, char *FILEBASE_S, char *FILEBASE_Q, \
                        int *NUMBER_OF_ITERATIONS, \
                        int *Nv, int *Nz, double *PIXEL_WIDTH, double *LENGTH_V)
{
    int N_KLIN_BIN, N_KLOG_BIN, sfid_off;

    double  K_0, LIN_K_SPACING, LOG_K_SPACING, \
            Z_0, temp_chisq = -1;

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
    
    // // File names and paths
    cFile.addKey("FileNameList", FNAME_LIST, STRING);

    cFile.addKey("FileNameRList",  FNAME_RLIST, STRING);
    cFile.addKey("FileInputDir",   INPUT_DIR, STRING);
    cFile.addKey("OutputDir",      OUTPUT_DIR, STRING); // Lya
    cFile.addKey("OutputFileBase", OUTPUT_FILEBASE, STRING);

    cFile.addKey("SignalLookUpTableBase",       FILEBASE_S, STRING);
    cFile.addKey("DerivativeSLookUpTableBase",  FILEBASE_Q, STRING);

    // Integration grid parameters
    cFile.addKey("NumberVPoints",   Nv, INTEGER);
    cFile.addKey("NumberZPoints",   Nz, INTEGER);
    cFile.addKey("PixelWidth",      PIXEL_WIDTH, DOUBLE);
    cFile.addKey("VelocityLength",  LENGTH_V,    DOUBLE);

    // Fiducial Palanque fit function parameters
    cFile.addKey("FiducialAmplitude",           &FIDUCIAL_PD13_PARAMS.A,     DOUBLE);
    cFile.addKey("FiducialSlope",               &FIDUCIAL_PD13_PARAMS.n,     DOUBLE);
    cFile.addKey("FiducialCurvature",           &FIDUCIAL_PD13_PARAMS.alpha, DOUBLE);
    cFile.addKey("FiducialRedshiftPower",       &FIDUCIAL_PD13_PARAMS.B,     DOUBLE);
    cFile.addKey("FiducialRedshiftCurvature",   &FIDUCIAL_PD13_PARAMS.beta,  DOUBLE);
    cFile.addKey("FiducialLorentzianLambda",    &FIDUCIAL_PD13_PARAMS.lambda,  DOUBLE);

    cFile.addKey("NumberOfIterations", NUMBER_OF_ITERATIONS, INTEGER);
    cFile.addKey("ChiSqConvergence", &temp_chisq, DOUBLE);

    // Read integer if testing outside of Lya region
    cFile.addKey("TurnOffBaseline", &sfid_off, INTEGER);

    cFile.addKey("AllocatedMemoryMB", &MEMORY_ALLOC, DOUBLE);

    cFile.addKey("TemporaryFolder", &TMP_FOLDER, STRING);

    cFile.readAll();

    char tmp_ps_fname[320];
    sprintf(tmp_ps_fname, "%s/tmppsfileXXXXXX", TMP_FOLDER);
    // Test access

    TURN_OFF_SFID = sfid_off > 0;

    if (TURN_OFF_SFID)  printf("Fiducial signal matrix is turned off.\n");
    if (temp_chisq > 0) CHISQ_CONVERGENCE_EPS = temp_chisq;

    // Redshift and wavenumber bins are constructed
    set_up_bins(K_0, N_KLIN_BIN, LIN_K_SPACING, N_KLOG_BIN, LOG_K_SPACING, Z_0);
    LOGGER.open(OUTPUT_DIR);
}














