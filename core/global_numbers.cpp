#include "core/global_numbers.hpp"
#include "core/fiducial_cosmology.hpp"
#include "core/quadratic_estimate.hpp"
#include "io/config_file.hpp"
#include "io/logger.hpp"

#include <cstdio>
#include <cmath>
#include <chrono>    /* clock_t, clock, CLOCKS_PER_SEC */

namespace process
{
    int this_pe=0, total_pes=1;
    char TMP_FOLDER[300] = ".";
    double MEMORY_ALLOC  = 0;
    SQLookupTable *sq_private_table;
    bool SAVE_EACH_PE_RESULT = false;
    bool SAVE_ALL_SQ_FILES = false;

    void updateMemory(double deltamem)
    {
        MEMORY_ALLOC += deltamem;
        if (MEMORY_ALLOC < 10)
            LOG::LOGGER.ERR("Remaining memory is less than 10 MB!\n");
    }
}

namespace bins
{
    int NUMBER_OF_K_BANDS, NUMBER_OF_Z_BINS, TOTAL_KZ_BINS, DEGREE_OF_FREEDOM;
    double *KBAND_EDGES, *KBAND_CENTERS;
    double  Z_BIN_WIDTH, *ZBIN_CENTERS, Z_LOWER_EDGE, Z_UPPER_EDGE;
    double (*redshiftBinningFunction)(double z, int zm);

    void setUpBins(double k0, int nlin, double dklin, int nlog, double dklog, double klast, double z0)
    {
        // Construct k edges
        NUMBER_OF_K_BANDS = nlin + nlog;
        
        // Add one more bin if klast is larger than the last bin
        double ktemp = (k0 + dklin*nlin)*pow(10, nlog*dklog);
        if (klast > ktemp)
            ++NUMBER_OF_K_BANDS;

        DEGREE_OF_FREEDOM = NUMBER_OF_K_BANDS * NUMBER_OF_Z_BINS;

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
        if (klast > ktemp)
            KBAND_EDGES[NUMBER_OF_K_BANDS] = klast;


        // Set up k bin centers
        for (int kn = 0; kn < NUMBER_OF_K_BANDS; ++kn)
            KBAND_CENTERS[kn] = (KBAND_EDGES[kn] + KBAND_EDGES[kn + 1]) / 2.;

        // Construct redshift bins
        ZBIN_CENTERS = new double[NUMBER_OF_Z_BINS];

        for (int zm = 0; zm < NUMBER_OF_Z_BINS; ++zm)
            ZBIN_CENTERS[zm] = z0 + Z_BIN_WIDTH * zm;

        Z_LOWER_EDGE = ZBIN_CENTERS[0] - Z_BIN_WIDTH/2.;
        Z_UPPER_EDGE = ZBIN_CENTERS[NUMBER_OF_Z_BINS-1] + Z_BIN_WIDTH/2.;
    }

    void cleanUpBins()
    {
        delete [] KBAND_EDGES;
        delete [] KBAND_CENTERS;
        delete [] ZBIN_CENTERS;
    }

    int findRedshiftBin(double z)
    {
        if (z < Z_LOWER_EDGE)
            return -1;
        
        int r = (z - Z_LOWER_EDGE) / Z_BIN_WIDTH;

        if (r >= NUMBER_OF_Z_BINS)
            r = NUMBER_OF_Z_BINS;

        return r;
    }

    // This is an extrapolating approach when limited to to only neighbouring bins
    // Not used in latest versions
    // What happens when a pixel goes outside of triangular bins?
    //    Imagine our central bin is z(m) and its left most pixel is at z_ij less than z(m-1).
    //    Its behavior can be well defined in z(m) bin, simply take z_ij < z(m).
    //    However, it should be distributed to z(m-1), and there occurs the problem.
    //    Since z_ij is also on the left of z(m-1), it belongs to the wrong interpolation kernel.
    //    In other words, that pixel is not normally distributed (sum of weights not equal to 1).
    //    That pixel does not belong to the correct z bin. There should not be such pixels.
    //    Below is my fix by keeping track of left and right bins.
    double zBinTriangularExtrapolate(double z, int zm, int zc)
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

    // binning functio for zm=0
    inline
    double zBinTriangular1(double z, int zm)
    {
        int zmm __attribute__((unused)) = zm;
        double x=z-ZBIN_CENTERS[0], r = 1-fabs(x)/Z_BIN_WIDTH;
        if (r<0) return 0;
        if (x<0) return 1;

        return r;
    }

    // binning function for last zm
    inline
    double zBinTriangular2(double z, int zm)
    {
        int zmm __attribute__((unused)) = zm;
        double x=z-ZBIN_CENTERS[NUMBER_OF_Z_BINS-1], r = 1-fabs(x)/Z_BIN_WIDTH;
        if (r<0) return 0;
        if (x>0) return 1;

        return r;
    }

    // binning functio for non-boundary zm
    inline
    double zBinTriangular(double z, int zm)
    {
        double x=z-ZBIN_CENTERS[zm], r = 1-fabs(x)/Z_BIN_WIDTH;
        if (r<0) return 0;

        return r;
    }

    // Assumes z always in bin, because it's cut in OneQSO.
    inline
    double zBinTopHat(double z, int zm)
    {
        int zz __attribute__((unused))  = z;
        int zmm __attribute__((unused)) = zm;
        return 1;
    }

    void setRedshiftBinningFunction(int zm)
    {
        #if defined(TOPHAT_Z_BINNING_FN)
        redshiftBinningFunction = &zBinTopHat;

        #elif defined(TRIANGLE_Z_BINNING_FN)
        if (zm == 0)
            redshiftBinningFunction = &zBinTriangular1;
        else if (zm == NUMBER_OF_Z_BINS-1)
            redshiftBinningFunction = &zBinTriangular2;
        else
            redshiftBinningFunction = &zBinTriangular;
        #endif
    }

    // Given the redshift z, returns binning weight. 1 for top-hats, interpolation for triangular
    // zm: Bin number to consider
    // zc: Central bin number for triangular bins. Binning weights depend on being to the left 
    // or to the right of this number.
    // extern inline 

    // double redshiftBinningFunction(double z, int zm)
    // {
    //     #if defined(TOPHAT_Z_BINNING_FN)
    //     if (zm == findRedshiftBin(z)) return 1;
    //     else                          return 0;

    //     #elif defined(TRIANGLE_Z_BINNING_FN)
    //     double x=z-ZBIN_CENTERS[zm], r = 1-fabs(x)/Z_BIN_WIDTH;
    //     if (r<0) return 0;
    //     if (zm==0 && x<0) return 1;
    //     if (zm==(NUMBER_OF_Z_BINS-1) && x>0) return 1;
    //     return r;
    //     #endif
    // }

}

namespace mytime
{
    Timer timer;
    double   time_spent_on_c_inv    = 0, time_spent_on_f_inv   = 0;
    double   time_spent_on_set_sfid = 0, time_spent_set_qs     = 0,
             time_spent_set_modqs   = 0, time_spent_set_fisher = 0;

    double   time_spent_on_q_interp = 0, time_spent_on_q_copy = 0;
    long     number_of_times_called_setq = 0, number_of_times_called_setsfid = 0;

    void printfTimeSpentDetails()
    {
        LOG::LOGGER.STD("Total time spent on inverting C is %.2f mins.\n", time_spent_on_c_inv);
        LOG::LOGGER.STD("Total time spent on inverting F is %.2f mins.\n", time_spent_on_f_inv);

        LOG::LOGGER.STD("Total time spent on setting Sfid is %.2f mins with %lu calls.\n",
            time_spent_on_set_sfid, number_of_times_called_setsfid);
        LOG::LOGGER.STD("Total time spent on setting Qs is %.2f mins with %lu calls.\n"
            "Interpolation: %.2f and Copy: %.2f.\n",
            time_spent_set_qs, number_of_times_called_setq, time_spent_on_q_interp, time_spent_on_q_copy);
        
        LOG::LOGGER.STD("Total time spent on setting Mod Qs is %.2f mins.\n", time_spent_set_modqs  );
        LOG::LOGGER.STD("Total time spent on setting F is %.2f mins.\n",      time_spent_set_fisher );

        //                Cinv     Finv     S      NS      Q       N_Q     TQmod   T_F
        LOG::LOGGER.TIME("%9.3e | %9.3e | %9.3e | %9.3e | %9.3e | %9.3e | %9.3e | %9.3e | ",
            time_spent_on_c_inv, time_spent_on_f_inv, time_spent_on_set_sfid, (double)number_of_times_called_setsfid,
            time_spent_set_qs, (double)number_of_times_called_setq, time_spent_set_modqs, time_spent_set_fisher);
    }

    void writeTimeLogHeader()
    {
        LOG::LOGGER.TIME("| %2s | %9s | %9s | %9s | %9s | %9s | %9s | %9s | %9s | %9s | %9s | %9s | %9s |\n", 
        "i", "T_i", "T_tot", "T_Cinv", "T_Finv", "T_Sfid", "N_Sfid", "T_Q", "N_Q", "T_Qmod", "T_F", "DChi2", "DMean");
    }
}

namespace specifics
{
    double CHISQ_CONVERGENCE_EPS = 0.01;
    bool   TURN_OFF_SFID, SMOOTH_LOGK_LOGP, USE_RESOLUTION_MATRIX;
    int CONT_LOGLAM_MARG_ORDER = 1, CONT_LAM_MARG_ORDER =1, NUMBER_OF_CHUNKS = 1;
    double RESOMAT_DECONVOLUTION_M = 0;
    qio::ifileformat INPUT_QSO_FILE = qio::Binary;
    int OVERSAMPLING_FACTOR = -1;
    
    #if defined(TOPHAT_Z_BINNING_FN)
    #define BINNING_SHAPE "Top Hat"
    #elif defined(TRIANGLE_Z_BINNING_FN)
    #define BINNING_SHAPE "Triangular"
    #else
    #define BINNING_SHAPE "ERROR NOT DEFINED"
    #endif

    #define tostr(a) #a
    #define tovstr(a) tostr(a)

    #if defined(FISHER_OPTIMIZATION)
    #define FISHER_TXT "ON"
    #else
    #define FISHER_TXT "OFF"
    #endif
    
    #if defined(REDSHIFT_GROWTH_POWER)
    #define RGP_TEXT "ON"
    #else
    #define RGP_TEXT "OFF"
    #endif
    
    const char BUILD_SPECIFICS[] =  
        "# This version is build by the following options:\n"
        "# Fisher optimization: " FISHER_TXT "\n"
        "# Redshift binning shape: " BINNING_SHAPE "\n" 
        "# Redshift growth scaling: " RGP_TEXT "\n";

    #undef tostr
    #undef tovstr
    #undef BINNING_SHAPE
    #undef FISHER_TXT
    #undef RGP_TEXT
    #undef TORE_TEXT

    void printBuildSpecifics(FILE *toWrite)
    {
        if (toWrite == NULL)
            LOG::LOGGER.STD(BUILD_SPECIFICS);
        else
            fprintf(toWrite, specifics::BUILD_SPECIFICS);
    }

    void printConfigSpecifics(FILE *toWrite)
    {
        #define CONFIG_TXT "# Using following configuration parameters:\n" \
            "# Fiducial Signal Baseline: %s\n" \
            "# Input is delta flux: %s\n" \
            "# Divide by mean flux of the chunk: %s\n" \
            "# ContinuumLogLamMargOrder: %d\n" \
            "# ContinuumLamMargOrder: %d\n" \
            "# Number of chunks: %d\n", \
            TURN_OFF_SFID ? "OFF" : "ON", \
            conv::INPUT_IS_DELTA_FLUX ? "YES" : "NO", \
            conv::FLUX_TO_DELTAF_BY_CHUNKS ? "ON" : "OFF", \
            CONT_LOGLAM_MARG_ORDER, CONT_LAM_MARG_ORDER, \
            NUMBER_OF_CHUNKS

        if (toWrite == NULL)
            LOG::LOGGER.STD(CONFIG_TXT);
        else
            fprintf(toWrite, CONFIG_TXT);

        #undef CONFIG_TXT
    }
}

// Pass NULL for not needed variables!
void ioh::readConfigFile(const char *FNAME_CONFIG,
    char *FNAME_LIST, char *FNAME_RLIST, char *INPUT_DIR, char *OUTPUT_DIR,
    char *OUTPUT_FILEBASE, char *FILEBASE_S, char *FILEBASE_Q,
    int *NUMBER_OF_ITERATIONS,
    int *Nv, int *Nz, double *LENGTH_V)
{
    int N_KLIN_BIN, N_KLOG_BIN, 
        sfid_off=-1, uchunkmean=-1, udeltaf=-1, usmoothlogs=-1,
        save_pe_res=-1, use_picca_file=-1, use_reso_mat=-1,
        cache_all_sq=-1, noise_smoothing_factor=-1;

    double  K_0, LIN_K_SPACING, LOG_K_SPACING, Z_0, temp_chisq = -1, klast=-1;
    char    FNAME_FID_POWER[300]="", FNAME_MEAN_FLUX[300]="", FNAME_PREFISHER[300]="";

    // Set up config file to read variables.
    ConfigFile cFile(FNAME_CONFIG);

    // Bin parameters
    cFile.addKey("K0", &K_0, DOUBLE);
    cFile.addKey("LinearKBinWidth",  &LIN_K_SPACING, DOUBLE);
    cFile.addKey("Log10KBinWidth",   &LOG_K_SPACING, DOUBLE);
    cFile.addKey("NumberOfLinearBins",   &N_KLIN_BIN, INTEGER);
    cFile.addKey("NumberOfLog10Bins",    &N_KLOG_BIN, INTEGER);
    cFile.addKey("LastKEdge",    &klast, DOUBLE);

    cFile.addKey("FirstRedshiftBinCenter", &Z_0, DOUBLE);
    cFile.addKey("RedshiftBinWidth", &bins::Z_BIN_WIDTH, DOUBLE);
    cFile.addKey("NumberOfRedshiftBins", &bins::NUMBER_OF_Z_BINS, INTEGER);
    
    // // File names and paths
    cFile.addKey("FileNameList", FNAME_LIST, STRING);

    cFile.addKey("FileNameRList",  FNAME_RLIST, STRING);
    cFile.addKey("FileInputDir",   INPUT_DIR, STRING);
    cFile.addKey("InputIsPicca",   &use_picca_file, INTEGER);
    cFile.addKey("UseResoMatrix",  &use_reso_mat, INTEGER);
    cFile.addKey("ResoMatDeconvolutionM", &specifics::RESOMAT_DECONVOLUTION_M, DOUBLE);
    cFile.addKey("OversampleRmat", &specifics::OVERSAMPLING_FACTOR, INTEGER);
    cFile.addKey("SmoothNoiseWeights", &noise_smoothing_factor, INTEGER);
    cFile.addKey("DynamicChunkNumber", &specifics::NUMBER_OF_CHUNKS, INTEGER);

    cFile.addKey("OutputDir",      OUTPUT_DIR, STRING); 
    cFile.addKey("OutputFileBase", OUTPUT_FILEBASE, STRING);

    cFile.addKey("SaveEachProcessResult", &save_pe_res, INTEGER);

    cFile.addKey("SignalLookUpTableBase",       FILEBASE_S, STRING);
    cFile.addKey("DerivativeSLookUpTableBase",  FILEBASE_Q, STRING);
    cFile.addKey("CacheAllSQTables", &cache_all_sq, INTEGER);

    // Integration grid parameters
    cFile.addKey("NumberVPoints",   Nv, INTEGER);
    cFile.addKey("NumberZPoints",   Nz, INTEGER);
    cFile.addKey("VelocityLength",  LENGTH_V,    DOUBLE);

    // Fiducial cosmology
    cFile.addKey("TurnOffBaseline", &sfid_off,  INTEGER);    // Turns off the signal matrix
    cFile.addKey("SmoothLnkLnP",  &usmoothlogs, INTEGER);    // Smooth lnk, lnP

    // How to convert from flux to delta_flux if at all
    cFile.addKey("MeanFluxFile",        FNAME_MEAN_FLUX, STRING);  // File to interpolate for F-bar
    cFile.addKey("UseChunksMeanFlux",   &uchunkmean,     INTEGER); // If 1, uses mean of each chunk as F-bar
    cFile.addKey("InputIsDeltaFlux",    &udeltaf,        INTEGER); // If 1, input is delta_f

    // Baseline Power Spectrum
    cFile.addKey("FiducialPowerFile",           FNAME_FID_POWER,                      STRING);
    // Fiducial Palanque fit function parameters
    cFile.addKey("FiducialAmplitude",           &fidpd13::FIDUCIAL_PD13_PARAMS.A,     DOUBLE);
    cFile.addKey("FiducialSlope",               &fidpd13::FIDUCIAL_PD13_PARAMS.n,     DOUBLE);
    cFile.addKey("FiducialCurvature",           &fidpd13::FIDUCIAL_PD13_PARAMS.alpha, DOUBLE);
    cFile.addKey("FiducialRedshiftPower",       &fidpd13::FIDUCIAL_PD13_PARAMS.B,     DOUBLE);
    cFile.addKey("FiducialRedshiftCurvature",   &fidpd13::FIDUCIAL_PD13_PARAMS.beta,  DOUBLE);
    cFile.addKey("FiducialLorentzianLambda",    &fidpd13::FIDUCIAL_PD13_PARAMS.lambda,DOUBLE);

    cFile.addKey("PrecomputedFisher", FNAME_PREFISHER, STRING);

    cFile.addKey("NumberOfIterations", NUMBER_OF_ITERATIONS, INTEGER);
    cFile.addKey("ChiSqConvergence", &temp_chisq, DOUBLE);

    // Continuum marginalization order. Pass <=0 to turn off
    cFile.addKey("ContinuumLogLambdaMargOrder", &specifics::CONT_LOGLAM_MARG_ORDER, INTEGER);
    cFile.addKey("ContinuumLambdaMargOrder", &specifics::CONT_LAM_MARG_ORDER, INTEGER);

    // Read integer if testing outside of Lya region
    cFile.addKey("AllocatedMemoryMB", &process::MEMORY_ALLOC, DOUBLE);
    cFile.addKey("TemporaryFolder", &process::TMP_FOLDER, STRING);

    cFile.readAll((process::this_pe!=0));

    // char tmp_ps_fname[320];
    // sprintf(tmp_ps_fname, "%s/tmppsfileXXXXXX", TMP_FOLDER);
    // TODO: Test access here
    
    specifics::TURN_OFF_SFID        = sfid_off > 0;
    specifics::SMOOTH_LOGK_LOGP     = usmoothlogs > 0;
    specifics::USE_RESOLUTION_MATRIX= use_reso_mat > 0;
    conv::FLUX_TO_DELTAF_BY_CHUNKS  = uchunkmean > 0;
    conv::INPUT_IS_DELTA_FLUX       = udeltaf > 0;
    process::SAVE_EACH_PE_RESULT    = save_pe_res > 0;
    process::SAVE_ALL_SQ_FILES      = cache_all_sq > 0;

    if (use_picca_file>0)
        specifics::INPUT_QSO_FILE = qio::Picca;

    if (specifics::INPUT_QSO_FILE != qio::Picca && specifics::USE_RESOLUTION_MATRIX)
        throw std::invalid_argument("Resolution matrix is only supported with picca files."
            " Add 'InputIsPicca 1' to config file if so.");

    #if !defined(ENABLE_MPI)
    if (process::SAVE_EACH_PE_RESULT)
        throw std::invalid_argument("Bootstrap saving only supported when compiled with MPI.");
    #endif

    // resolve conflict: Input delta flux overrides all
    // Then, chunk means.
    if (conv::INPUT_IS_DELTA_FLUX && conv::FLUX_TO_DELTAF_BY_CHUNKS)
    {
        LOG::LOGGER.ERR("Both input delta flux and conversion using chunk's mean "
            "flux is turned on. Assuming input is flux fluctuations delta_f.\n");
        conv::FLUX_TO_DELTAF_BY_CHUNKS = false;
    }

    conv::setMeanFlux();

    if (FNAME_MEAN_FLUX[0] != '\0')
    {
        if (conv::FLUX_TO_DELTAF_BY_CHUNKS)
        {
            LOG::LOGGER.ERR("Both mean flux file and using chunk's mean flux is turned on. "
            "Using chunk's mean flux.\n");
        }
        else if (conv::INPUT_IS_DELTA_FLUX)
        {
            LOG::LOGGER.ERR("Both input delta flux and conversion using mean flux file is turned on. "
            "Assuming input is flux fluctuations delta_f.\n");
        }
        else
            conv::setMeanFlux(FNAME_MEAN_FLUX);
    }
    else if (!(conv::INPUT_IS_DELTA_FLUX || conv::FLUX_TO_DELTAF_BY_CHUNKS))
        conv::INPUT_IS_DELTA_FLUX = true;

    if (temp_chisq > 0) specifics::CHISQ_CONVERGENCE_EPS = temp_chisq;

    if (FNAME_FID_POWER[0] != '\0')
        fidcosmo::setFiducialPowerFromFile(FNAME_FID_POWER);

    // Redshift and wavenumber bins are constructed
    bins::setUpBins(K_0, N_KLIN_BIN, LIN_K_SPACING, N_KLOG_BIN, LOG_K_SPACING, klast, Z_0);

    // Call after setting bins, because this function checks for consistency.
    if (FNAME_PREFISHER[0] != '\0')
        OneDQuadraticPowerEstimate::readPrecomputedFisher(FNAME_PREFISHER);

    Smoother::setParameters(noise_smoothing_factor);
    Smoother::setGaussianKernel();
}














