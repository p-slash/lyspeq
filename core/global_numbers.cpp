#include "core/global_numbers.hpp"
#include "io/config_file.hpp"
#include "io/logger.hpp"

#include <cstdio>
#include <cmath>
#include <chrono>    /* clock_t, clock, CLOCKS_PER_SEC */
#include <stdexcept> // std::invalid_argument

namespace process
{
    int this_pe=0, total_pes=1;
    std::string TMP_FOLDER = ".";
    std::string FNAME_BASE;
    double MEMORY_ALLOC  = 0;
    bool SAVE_EACH_PE_RESULT = false;
    bool SAVE_ALL_SQ_FILES = false;

    void updateMemory(double deltamem)
    {
        MEMORY_ALLOC += deltamem;
        if (MEMORY_ALLOC < 10)
            LOG::LOGGER.ERR("Remaining memory is less than 10 MB!\n");
    }

    void readProcess(const ConfigFile &config)
    {
        int save_pe_res = config.getInteger("SaveEachProcessResult", -1), 
            cache_all_sq = config.getInteger("CacheAllSQTables", -1);
        MEMORY_ALLOC = config.getDouble("AllocatedMemoryMB");

        SAVE_EACH_PE_RESULT    = save_pe_res > 0;
        SAVE_ALL_SQ_FILES      = cache_all_sq > 0;
        FNAME_BASE = config.get("OutputDir", ".") + '/' + config.get("OutputFileBase");
        TMP_FOLDER = config.get("TemporaryFolder");

        #if !defined(ENABLE_MPI)
        if (SAVE_EACH_PE_RESULT)
            throw std::invalid_argument("Bootstrap saving only supported when compiled with MPI.");
        #endif
    }
}

namespace bins
{
    int NUMBER_OF_K_BANDS, NUMBER_OF_Z_BINS, TOTAL_KZ_BINS, 
        FISHER_SIZE, DEGREE_OF_FREEDOM;
    double *KBAND_EDGES, *KBAND_CENTERS;
    double  Z_BIN_WIDTH, *ZBIN_CENTERS, Z_LOWER_EDGE, Z_UPPER_EDGE;
    double (*redshiftBinningFunction)(double z, int zm) = &zBinTopHat;

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
        FISHER_SIZE = TOTAL_KZ_BINS * TOTAL_KZ_BINS;

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

    void readBins(const ConfigFile &config)
    {
        int N_KLIN_BIN, N_KLOG_BIN;
        double  K_0, LIN_K_SPACING, LOG_K_SPACING, Z_0, klast=-1;

        K_0 = config.getDouble("K0");
        LIN_K_SPACING = config.getDouble("LinearKBinWidth");
        LOG_K_SPACING = config.getDouble("Log10KBinWidth");
        N_KLIN_BIN = config.getInteger("NumberOfLinearBins");
        N_KLOG_BIN = config.getInteger("NumberOfLog10Bins");
        klast = config.getDouble("LastKEdge");

        Z_0 = config.getDouble("FirstRedshiftBinCenter");
        Z_BIN_WIDTH = config.getDouble("RedshiftBinWidth");
        NUMBER_OF_Z_BINS = config.getInteger("NumberOfRedshiftBins");

        // Redshift and wavenumber bins are constructed
        bins::setUpBins(K_0, N_KLIN_BIN, LIN_K_SPACING, N_KLOG_BIN, LOG_K_SPACING, klast, Z_0);
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
        double zz __attribute__((unused))  = z;
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
    bool   TURN_OFF_SFID, SMOOTH_LOGK_LOGP, USE_RESOLUTION_MATRIX,
           PRECOMPUTED_FISHER;
    int CONT_LOGLAM_MARG_ORDER = 1, CONT_LAM_MARG_ORDER = 1, 
        CONT_NVECS = 3, NUMBER_OF_CHUNKS = 1;
    double RESOMAT_DECONVOLUTION_M = 0;
    qio::ifileformat INPUT_QSO_FILE = qio::Binary;
    int OVERSAMPLING_FACTOR = -1;

    void calcNvecs()
    {
        if (CONT_LOGLAM_MARG_ORDER >= 0 || CONT_LAM_MARG_ORDER >= 0)
        {
            CONT_NVECS = 1;
            if (CONT_LOGLAM_MARG_ORDER > 0)
                CONT_NVECS += CONT_LOGLAM_MARG_ORDER;
            if (CONT_LAM_MARG_ORDER > 0)
                CONT_NVECS += CONT_LAM_MARG_ORDER;
        }
        else
            CONT_NVECS = 0;
    }

    void readSpecifics(const ConfigFile &config)
    {
        int sfid_off, usmoothlogs, use_picca_file, use_reso_mat;
        double  temp_chisq;

        use_picca_file = config.getInteger("InputIsPicca", -1);
        use_reso_mat = config.getInteger("UseResoMatrix", -1);
        RESOMAT_DECONVOLUTION_M = config.getDouble("ResoMatDeconvolutionM");
        OVERSAMPLING_FACTOR = config.getInteger("OversampleRmat", -1);
        NUMBER_OF_CHUNKS = config.getInteger("DynamicChunkNumber", 1);

        sfid_off = config.getInteger("TurnOffBaseline", -1);    // Turns off the signal matrix
        usmoothlogs = config.getInteger("SmoothLnkLnP", -1);    // Smooth lnk, lnP
        temp_chisq = config.getDouble("ChiSqConvergence", -1.);

        // Continuum marginalization order. Pass <=0 to turn off
        CONT_LOGLAM_MARG_ORDER = config.getInteger("ContinuumLogLambdaMargOrder", 1);
        CONT_LAM_MARG_ORDER = config.getInteger("ContinuumLambdaMargOrder", 1);

        // char tmp_ps_fname[320];
        // sprintf(tmp_ps_fname, "%s/tmppsfileXXXXXX", TMP_FOLDER);
        // TODO: Test access here
        
        TURN_OFF_SFID        = sfid_off > 0;
        SMOOTH_LOGK_LOGP     = usmoothlogs > 0;
        USE_RESOLUTION_MATRIX= use_reso_mat > 0;
        PRECOMPUTED_FISHER   = !config.get("PrecomputedFisher").empty();

        if (use_picca_file>0)
            INPUT_QSO_FILE = qio::Picca;

        if (INPUT_QSO_FILE != qio::Picca && USE_RESOLUTION_MATRIX)
            throw std::invalid_argument("Resolution matrix is only supported with picca files."
                " Add 'InputIsPicca 1' to config file if so.");

        if (temp_chisq > 0) CHISQ_CONVERGENCE_EPS = temp_chisq;

        calcNvecs();
    }
    
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
        // "# Input is delta flux: %s\n" 
        // "# Divide by mean flux of the chunk: %s\n"
        // conv::INPUT_IS_DELTA_FLUX ? "YES" : "NO",
        // conv::FLUX_TO_DELTAF_BY_CHUNKS ? "ON" : "OFF", 
        #define CONFIG_TXT "# Using following configuration parameters:\n" \
            "# Fiducial Signal Baseline: %s\n" \
            "# ContinuumLogLamMargOrder: %d\n" \
            "# ContinuumLamMargOrder: %d\n" \
            "# Number of chunks: %d\n", \
            TURN_OFF_SFID ? "OFF" : "ON", \
            CONT_LOGLAM_MARG_ORDER, CONT_LAM_MARG_ORDER, \
            NUMBER_OF_CHUNKS

        if (toWrite == NULL)
            LOG::LOGGER.STD(CONFIG_TXT);
        else
            fprintf(toWrite, CONFIG_TXT);

        #undef CONFIG_TXT
    }
}














