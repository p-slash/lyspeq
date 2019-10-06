#include "core/global_numbers.hpp"
#include "core/fiducial_cosmology.hpp"
#include "io/config_file.hpp"
#include "io/logger.hpp"

#include <cstdio>
#include <cmath>

#if defined(_OPENMP)
#include <omp.h> /*omp_get_wtime();*/
#else
#include <ctime>    /* clock_t, clock, CLOCKS_PER_SEC */
#endif

char TMP_FOLDER[300] = ".";

double CHISQ_CONVERGENCE_EPS = 0.01;
double MEMORY_ALLOC          = 0;

int t_rank = 0, numthreads = 1;

SQLookupTable *sq_private_table;

bool TURN_OFF_SFID;

namespace bins
{
    int NUMBER_OF_K_BANDS, NUMBER_OF_Z_BINS, TOTAL_KZ_BINS, DEGREE_OF_FREEDOM;
    double *KBAND_EDGES, *KBAND_CENTERS;
    double  Z_BIN_WIDTH, *ZBIN_CENTERS, z0_edge;

    void setUpBins(double k0, int nlin, double dklin,
                                int nlog, double dklog,
                     double z0)
    {
        // Construct k edges
        NUMBER_OF_K_BANDS = nlin + nlog;
        
        DEGREE_OF_FREEDOM = NUMBER_OF_K_BANDS * NUMBER_OF_Z_BINS;

        // One last bin is created when LAST_K_EDGE is set in Makefile.
        #ifdef LAST_K_EDGE
        ++NUMBER_OF_K_BANDS;
        #endif

        TOTAL_KZ_BINS = NUMBER_OF_K_BANDS * NUMBER_OF_Z_BINS;

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

        z0_edge = ZBIN_CENTERS[0] - Z_BIN_WIDTH/2.;
    }

    void cleanUpBins()
    {
        delete [] KBAND_EDGES;
        delete [] KBAND_CENTERS;
        delete [] ZBIN_CENTERS;
    }

    int findRedshiftBin(double z)
    {
        if (z < z0_edge)
            return -1;
        
        int r = (z - z0_edge) / Z_BIN_WIDTH;

        if (r >= NUMBER_OF_Z_BINS)
            r = NUMBER_OF_Z_BINS;

        return r;        
    }

    // This needs more thought
    // What happens when a pixel goes outside of triangular bins?
    //    Imagine our central bin is z(m) and its left most pixel is at z_ij less than z(m-1).
    //    Its behavior can be well defined in z(m) bin, simply take z_ij < z(m).
    //    However, it should be distributed to z(m-1), and there occurs the problem.
    //    Since z_ij is also on the left of z(m-1), it belongs to the wrong interpolation kernel.
    //    In other words, that pixel is not normally distributed (sum of weights not equal to 1).
    //    That pixel does not belong to the correct z bin. There should not be such pixels.
    //    Below is my fix by keeping track of left and right bins.
    double zBinTriangular(double z, int zm, int zc)
    {
        double zm_center = bins::ZBIN_CENTERS[zm], z_pivot = ZBIN_CENTERS[zc];
        
        if ((zm < zc && z >= z_pivot) || (zm > zc && z <= z_pivot))
            return 0.;

        // if (zm < zc)
        //     return (z_pivot - z) / Z_BIN_WIDTH;
        // else if (zm > zc)
        //     return (z - z_pivot) / Z_BIN_WIDTH;
        if (zm != zc)
            return (zm-zc) * (z - z_pivot) / Z_BIN_WIDTH;
        else if (z <= zm_center)
        {
            if (zm == 0)    return 1.;
            else            return (z - zm_center + Z_BIN_WIDTH) / Z_BIN_WIDTH;
        }
        else
        {
            if (zm == NUMBER_OF_Z_BINS - 1) return 1.;
            else                            return (zm_center + Z_BIN_WIDTH - z) / Z_BIN_WIDTH;
        }
    }

    double redshiftBinningFunction(double z, int zm, int zc)
    {
        #if defined(TOPHAT_Z_BINNING_FN) || defined(TURN_OFF_REDSHIFT_EVOLUTION)
        double zz  __attribute__((unused)) = z;
        int    zzm __attribute__((unused)) = zm,
               zzc __attribute__((unused)) = zc;
        return 1.;
        #endif

        #ifdef TRIANGLE_Z_BINNING_FN
        return zBinTriangular(z, zm, zc);
        #endif
    }

    int getFisherMatrixIndex(int kn, int zm)
    {
        return kn + bins::NUMBER_OF_K_BANDS * zm;
    }

    void getFisherMatrixBinNoFromIndex(int i, int &kn, int &zm)
    {
        kn = i % bins::NUMBER_OF_K_BANDS;
        zm = i / bins::NUMBER_OF_K_BANDS;
    }
}

namespace mytime
{
    double   time_spent_on_c_inv    = 0, time_spent_on_f_inv   = 0;
    double   time_spent_on_set_sfid = 0, time_spent_set_qs     = 0,
             time_spent_set_modqs   = 0, time_spent_set_fisher = 0;

    double   time_spent_on_q_interp = 0, time_spent_on_q_copy = 0;
    long     number_of_times_called_setq = 0, number_of_times_called_setsfid = 0;

    double getTime()
    {
        #if defined(_OPENMP)
        return omp_get_wtime() / 60.;
        #else
        clock_t t = clock();
        return ((double) t) / CLOCKS_PER_SEC / 60.;
        #endif
    }

    void printfTimeSpentDetails()
    {
        LOG::LOGGER.STD("Total time spent on inverting C is %.2f mins.\n", time_spent_on_c_inv);
        LOG::LOGGER.STD("Total time spent on inverting F is %.2f mins.\n", time_spent_on_f_inv);

        LOG::LOGGER.STD("Total time spent on setting Sfid is %.2f mins with %lu calls.\n",
                time_spent_on_set_sfid, number_of_times_called_setsfid);
        LOG::LOGGER.STD("Total time spent on setting Qs is %.2f mins with %lu calls.\nInterpolation: %.2f and Copy: %.2f.\n",
                time_spent_set_qs, number_of_times_called_setq, time_spent_on_q_interp, time_spent_on_q_copy);
        
        LOG::LOGGER.STD("Total time spent on setting Mod Qs is %.2f mins.\n", time_spent_set_modqs  );
        LOG::LOGGER.STD("Total time spent on setting F is %.2f mins.\n",      time_spent_set_fisher );

        //                Cinv     Finv     S      NS      Q       N_Q     TQmod   T_F
        LOG::LOGGER.TIME("%9.3e | %9.3e | %9.3e | %9.3e | %9.3e | %9.3e | %9.3e | %9.3e | ",
            time_spent_on_c_inv, time_spent_on_f_inv, time_spent_on_set_sfid, (double)number_of_times_called_setsfid,
            time_spent_set_qs, (double)number_of_times_called_setq, time_spent_set_modqs, time_spent_set_fisher);
    }
}

void specifics::printBuildSpecifics()
{
    LOG::LOGGER.STD(specifics::BUILD_SPECIFICS);
}

void specifics::printConfigSpecifics(FILE *toWrite)
{
    if (toWrite == NULL)
    {
        LOG::LOGGER.STD("Using following configuration parameters:\n"
        "Fiducial Signal Baseline: %s\n"
        "Velocity Spacing: %s\n"
        "Divide by mean flux of the chunk: %s\n", 
        TURN_OFF_SFID ? "OFF" : "ON",
        conv::USE_LOG_V ? "LOGARITHMIC" : "EdS",
        conv::FLUX_TO_DELTAF_BY_CHUNKS ? "ON" : "OFF");
    }
    else
    {
        fprintf(toWrite, "# Using following configuration parameters:\n"
        "# Fiducial Signal Baseline: %s\n"
        "# Velocity Spacing: %s\n"
        "# Divide by mean flux of the chunk: %s\n", 
        TURN_OFF_SFID ? "OFF" : "ON",
        conv::USE_LOG_V ? "LOGARITHMIC" : "EdS",
        conv::FLUX_TO_DELTAF_BY_CHUNKS ? "ON" : "OFF");
    }
    
}

// Pass NULL for not needed variables!
void ioh::readConfigFile(  const char *FNAME_CONFIG,
                        char *FNAME_LIST, char *FNAME_RLIST, char *INPUT_DIR, char *OUTPUT_DIR,
                        char *OUTPUT_FILEBASE, char *FILEBASE_S, char *FILEBASE_Q,
                        int *NUMBER_OF_ITERATIONS,
                        int *Nv, int *Nz, double *PIXEL_WIDTH, double *LENGTH_V)
{
    int N_KLIN_BIN, N_KLOG_BIN, sfid_off, ulogv=-1, uchunkmean=-1;

    double  K_0, LIN_K_SPACING, LOG_K_SPACING,
            Z_0, temp_chisq = -1;

    // Set up config file to read variables.
    ConfigFile cFile(FNAME_CONFIG);

    // Bin parameters
    cFile.addKey("K0", &K_0, DOUBLE);
    cFile.addKey("FirstRedshiftBinCenter", &Z_0, DOUBLE);

    cFile.addKey("LinearKBinWidth",  &LIN_K_SPACING, DOUBLE);
    cFile.addKey("Log10KBinWidth",   &LOG_K_SPACING, DOUBLE);
    cFile.addKey("RedshiftBinWidth", &bins::Z_BIN_WIDTH,   DOUBLE);

    cFile.addKey("NumberOfLinearBins",   &N_KLIN_BIN, INTEGER);
    cFile.addKey("NumberOfLog10Bins",    &N_KLOG_BIN, INTEGER);
    cFile.addKey("NumberOfRedshiftBins", &bins::NUMBER_OF_Z_BINS,   INTEGER);
    
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
    cFile.addKey("FiducialAmplitude",           &fidpd13::FIDUCIAL_PD13_PARAMS.A,     DOUBLE);
    cFile.addKey("FiducialSlope",               &fidpd13::FIDUCIAL_PD13_PARAMS.n,     DOUBLE);
    cFile.addKey("FiducialCurvature",           &fidpd13::FIDUCIAL_PD13_PARAMS.alpha, DOUBLE);
    cFile.addKey("FiducialRedshiftPower",       &fidpd13::FIDUCIAL_PD13_PARAMS.B,     DOUBLE);
    cFile.addKey("FiducialRedshiftCurvature",   &fidpd13::FIDUCIAL_PD13_PARAMS.beta,  DOUBLE);
    cFile.addKey("FiducialLorentzianLambda",    &fidpd13::FIDUCIAL_PD13_PARAMS.lambda,  DOUBLE);

    cFile.addKey("NumberOfIterations", NUMBER_OF_ITERATIONS, INTEGER);
    cFile.addKey("ChiSqConvergence", &temp_chisq, DOUBLE);

    // Read integer if testing outside of Lya region
    cFile.addKey("TurnOffBaseline", &sfid_off, INTEGER);

    cFile.addKey("AllocatedMemoryMB", &MEMORY_ALLOC, DOUBLE);

    cFile.addKey("TemporaryFolder", &TMP_FOLDER, STRING);
    cFile.addKey("UseLogarithmicVelocity", &ulogv, INTEGER);
    cFile.addKey("ConvertFromFluxToDeltaf", &uchunkmean, INTEGER);

    cFile.readAll();

    // char tmp_ps_fname[320];
    // sprintf(tmp_ps_fname, "%s/tmppsfileXXXXXX", TMP_FOLDER);
    // TODO: Test access here
    
    TURN_OFF_SFID   = sfid_off > 0;
    conv::USE_LOG_V = ulogv > 0;
    conv::FLUX_TO_DELTAF_BY_CHUNKS = uchunkmean > 0;

    if (temp_chisq > 0) CHISQ_CONVERGENCE_EPS = temp_chisq;

    // Redshift and wavenumber bins are constructed
    bins::setUpBins(K_0, N_KLIN_BIN, LIN_K_SPACING, N_KLOG_BIN, LOG_K_SPACING, Z_0);
}














