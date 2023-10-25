#include "core/global_numbers.hpp"
#include "io/config_file.hpp"
#include "io/logger.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept> // std::invalid_argument

namespace process
{
    int this_pe=0, total_pes=1;
    std::string TMP_FOLDER = ".";
    std::string FNAME_BASE;
    double MEMORY_ALLOC  = 0;
    bool SAVE_EACH_PE_RESULT = false;
    bool SAVE_EACH_CHUNK_RESULT = false;
    bool SAVE_ALL_SQ_FILES = false;

    void updateMemory(double deltamem)
    {
        MEMORY_ALLOC += deltamem;
        if (MEMORY_ALLOC < 10)
            LOG::LOGGER.ERR("Remaining memory is less than 10 MB!\n");
    }

    void readProcess(ConfigFile &config)
    {
        LOG::LOGGER.STD("###############################################\n");
        LOG::LOGGER.STD("Reading process parameters from config.\n");

        config.addDefaults(process_default_parameters);

        int save_pe_res = config.getInteger("SaveEachProcessResult"),
            save_chunk_res = config.getInteger("SaveEachChunkResult"),
            cache_all_sq = config.getInteger("CacheAllSQTables");
        MEMORY_ALLOC = config.getDouble("AllocatedMemoryMB");

        SAVE_EACH_PE_RESULT = save_pe_res > 0;
        SAVE_EACH_CHUNK_RESULT = save_chunk_res > 0;
        SAVE_ALL_SQ_FILES = cache_all_sq > 0;
        FNAME_BASE = config.get("OutputDir") + '/'
            + config.get("OutputFileBase");
        TMP_FOLDER = config.get("TemporaryFolder");

        #if !defined(ENABLE_MPI)
        if (SAVE_EACH_PE_RESULT)
            throw std::invalid_argument("Bootstrap saving only supported when compiled with MPI.");
        #endif

        LOG::LOGGER.STD("Fname base is set to %s.\n",
            FNAME_BASE.c_str());
        #define booltostr(x) x ? "true" : "false"
        LOG::LOGGER.STD("SaveEachProcessResult is set to %s.\n",
            booltostr(SAVE_EACH_PE_RESULT));
        LOG::LOGGER.STD("SaveEachChunkResult is set to %s.\n",
            booltostr(SAVE_EACH_CHUNK_RESULT));
        LOG::LOGGER.STD("CacheAllSQTables is set to %s.\n",
            booltostr(SAVE_ALL_SQ_FILES));
        LOG::LOGGER.STD("AllocatedMemoryMB is set to %.1e MB.\n",
            MEMORY_ALLOC);
        LOG::LOGGER.STD("TemporaryFolder is set to %s.\n\n",
            TMP_FOLDER.c_str());
        #undef booltostr
    }
}

namespace specifics
{
    double CHISQ_CONVERGENCE_EPS = 0.01;
    bool   TURN_OFF_SFID, SMOOTH_LOGK_LOGP, USE_RESOLUTION_MATRIX,
           USE_PRECOMPUTED_FISHER, USE_FFT_FOR_SQ;
    int CONT_LOGLAM_MARG_ORDER = 1, CONT_LAM_MARG_ORDER = 1, 
        CONT_NVECS = 3, NUMBER_OF_CHUNKS = 1, NUMBER_OF_BOOTS = 0;
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


    void readSpecifics(ConfigFile &config)
    {
        LOG::LOGGER.STD("###############################################\n");
        LOG::LOGGER.STD("Reading specifics from config.\n");
        config.addDefaults(specifics_default_parameters);
        int usmoothlogs, use_picca_file;
        double  temp_chisq;

        use_picca_file = config.getInteger("InputIsPicca", -1);
        USE_RESOLUTION_MATRIX = config.getInteger("UseResoMatrix", -1) > 0;
        RESOMAT_DECONVOLUTION_M = config.getDouble("ResoMatDeconvolutionM");
        OVERSAMPLING_FACTOR = config.getInteger("OversampleRmat", -1);
        USE_FFT_FOR_SQ = config.getInteger("UseFftMeanResolution", -1) > 0;
        NUMBER_OF_CHUNKS = config.getInteger("DynamicChunkNumber", 1);
        NUMBER_OF_BOOTS = config.getInteger("NumberOfBoots");

        // Turns off the signal matrix
        TURN_OFF_SFID = config.getInteger("TurnOffBaseline", -1) > 0;
        // Smooth lnk, lnP
        usmoothlogs = config.getInteger("SmoothLnkLnP", 1);
        temp_chisq = config.getDouble("ChiSqConvergence");

        // Continuum marginalization order. Pass <=0 to turn off
        CONT_LOGLAM_MARG_ORDER = config.getInteger(
            "ContinuumLogLambdaMargOrder", 1);
        CONT_LAM_MARG_ORDER = config.getInteger(
            "ContinuumLambdaMargOrder", 0);

        // char tmp_ps_fname[320];
        // sprintf(tmp_ps_fname, "%s/tmppsfileXXXXXX", TMP_FOLDER);
        // TODO: Test access here

        SMOOTH_LOGK_LOGP     = usmoothlogs > 0;
        std::string precomp_fisher_str = config.get("PrecomputedFisher");
        USE_PRECOMPUTED_FISHER   = !precomp_fisher_str.empty();

        if (use_picca_file>0)
            INPUT_QSO_FILE = qio::Picca;

        if (INPUT_QSO_FILE != qio::Picca && USE_RESOLUTION_MATRIX)
            throw std::invalid_argument(
                "Resolution matrix is only supported with picca files."
                " Add 'InputIsPicca 1' to config file if so.");

        if (USE_FFT_FOR_SQ && OVERSAMPLING_FACTOR > 0)
            throw std::invalid_argument(
                "Oversampling is not supported with FFTs");

        if (temp_chisq > 0) CHISQ_CONVERGENCE_EPS = temp_chisq;

        calcNvecs();

        #define booltostr(x) x ? "true" : "false"
        LOG::LOGGER.STD("InputIsPicca is set to %s.\n",
            booltostr(INPUT_QSO_FILE));
        LOG::LOGGER.STD("UseResoMatrix is set to %s.\n",
            booltostr(USE_RESOLUTION_MATRIX));
        LOG::LOGGER.STD("ResoMatDeconvolutionM is set to %.2f.\n",
            RESOMAT_DECONVOLUTION_M);
        LOG::LOGGER.STD("OversampleRmat is set to %d.\n",
            OVERSAMPLING_FACTOR);
        LOG::LOGGER.STD("DynamicChunkNumber is set to %d.\n",
            NUMBER_OF_CHUNKS);
        LOG::LOGGER.STD("Fiducial signal matrix is set to turned %s.\n",
            TURN_OFF_SFID ? "OFF" : "ON");
        LOG::LOGGER.STD("SmoothLnkLnP is set to %s.\n",
            booltostr(SMOOTH_LOGK_LOGP));
        LOG::LOGGER.STD("ChiSqConvergence is set to %.2e.\n",
            CHISQ_CONVERGENCE_EPS);
        LOG::LOGGER.STD("PrecomputedFisher is turn %s, and set to %s.\n",
            booltostr(USE_PRECOMPUTED_FISHER), precomp_fisher_str.c_str());
        LOG::LOGGER.STD("ContinuumLogLambdaMargOrder is set to %d.\n",
            CONT_LOGLAM_MARG_ORDER);
        LOG::LOGGER.STD("ContinuumLambdaMargOrder is set to %d.\n\n",
            CONT_LAM_MARG_ORDER);
        #undef booltostr
    }

    #if defined(TOPHAT_Z_BINNING_FN)
    const std::string BINNING_SHAPE = "Top Hat";
    #elif defined(TRIANGLE_Z_BINNING_FN)
    const std::string BINNING_SHAPE = "Triangular";
    #else
    const std::string BINNING_SHAPE = "ERROR NOT DEFINED";
    #endif

    #if defined(FISHER_OPTIMIZATION)
    const std::string FISHER_TXT = "ON";
    #else
    const std::string FISHER_TXT = "OFF";
    #endif

    #if defined(REDSHIFT_GROWTH_POWER)
    const std::string RGP_TEXT = "ON";
    #else
    const std::string RGP_TEXT = "OFF";
    #endif

    const std::string BUILD_SPECIFICS = 
        std::string("# You are using lyspeq version " __LYSPEQ_VERSION__ ".\n")
        + std::string("# This version is build by the following options:\n")
        + "# Fisher optimization: " + FISHER_TXT + "\n"
        + "# Redshift binning shape: " + BINNING_SHAPE + "\n"
        + "# Redshift growth scaling: " + RGP_TEXT + "\n";

    void printBuildSpecifics(FILE *toWrite)
    {
        if (toWrite == NULL)
            LOG::LOGGER.STD(BUILD_SPECIFICS.c_str());
        else
            fprintf(toWrite, "%s", BUILD_SPECIFICS.c_str());
    }
}

namespace bins
{
    int NUMBER_OF_K_BANDS, NUMBER_OF_Z_BINS, TOTAL_KZ_BINS, 
        FISHER_SIZE;
    std::vector<double> KBAND_EDGES, KBAND_CENTERS, ZBIN_CENTERS;
    double  Z_BIN_WIDTH, Z_LOWER_EDGE, Z_UPPER_EDGE;

    void setUpBins(double k0, int nlin, double dklin, int nlog, double dklog, double klast, double z0)
    {
        // Construct k edges
        NUMBER_OF_K_BANDS = nlin + nlog;

        // Add one more bin if klast is larger than the last bin
        double ktemp = (k0 + dklin*nlin)*pow(10, nlog*dklog);
        if (klast > ktemp)
            ++NUMBER_OF_K_BANDS;

        TOTAL_KZ_BINS = NUMBER_OF_K_BANDS * NUMBER_OF_Z_BINS;
        FISHER_SIZE = TOTAL_KZ_BINS * TOTAL_KZ_BINS;

        KBAND_EDGES.reserve(NUMBER_OF_K_BANDS + 1);
        KBAND_CENTERS.reserve(NUMBER_OF_K_BANDS);

        // Linearly spaced bins
        for (int i = 0; i < nlin + 1; i++)
            KBAND_EDGES.push_back(k0 + dklin * i);
        // Logarithmicly spaced bins
        for (int i = 1; i < nlog + 1; i++)
            KBAND_EDGES.push_back(KBAND_EDGES[nlin] * pow(10., i * dklog));
        
        // Last bin
        if (klast > ktemp)
            KBAND_EDGES.push_back(klast);

        // Set up k bin centers
        for (int kn = 0; kn < NUMBER_OF_K_BANDS; ++kn)
            KBAND_CENTERS.push_back((KBAND_EDGES[kn] + KBAND_EDGES[kn + 1]) / 2.);

        // Construct redshift bins
        ZBIN_CENTERS.reserve(NUMBER_OF_Z_BINS);

        for (int zm = 0; zm < NUMBER_OF_Z_BINS; ++zm)
            ZBIN_CENTERS.push_back(z0 + Z_BIN_WIDTH * zm);

        Z_LOWER_EDGE = ZBIN_CENTERS[0] - Z_BIN_WIDTH/2.;
        Z_UPPER_EDGE = ZBIN_CENTERS[NUMBER_OF_Z_BINS-1] + Z_BIN_WIDTH/2.;
    }

    /* This function reads following keys from config file:
    K0: double
        First edge for the k bins. 0 by default.
    LinearKBinWidth: double
        Linear k bin spacing. Need to be present 
        and > 0 if NumberOfLinearBins > 0.
    Log10KBinWidth: double
        Logarithmic k bins spacing. Need to be present 
        and > 0 if NumberOfLinearBins > 0.
    NumberOfLinearBins: int
        Number of linear bins.
    NumberOfLog10Bins: int
        Number of log bins.
    LastKEdge: double
        The last k edge will be this by adding a k bin if the value is valid.
    FirstRedshiftBinCenter: double
    RedshiftBinWidth: double
    NumberOfRedshiftBins: double
    */
    void readBins(ConfigFile &config)
    {
        LOG::LOGGER.STD("###############################################\n");
        LOG::LOGGER.STD("Reading binning parameters from config.\n");

        config.addDefaults(bins_default_parameters);

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

        if (N_KLIN_BIN > 0 && !(LIN_K_SPACING > 0))
            throw std::invalid_argument("NumberOfLinearBins > 0, so "
                "LinearKBinWidth must be > 0.");

        if (N_KLOG_BIN > 0 && !(LOG_K_SPACING > 0))
            throw std::invalid_argument("NumberOfLog10Bins > 0, so "
                "Log10KBinWidth must be > 0.");

        if (N_KLIN_BIN <= 0 && N_KLOG_BIN <= 0)
            throw std::invalid_argument("At least NumberOfLinearBins or "
                "NumberOfLog10Bins must be present.");

        if (Z_0 <= 0)
            throw std::invalid_argument("FirstRedshiftBinCenter must be > 0.");

        if (Z_BIN_WIDTH <= 0)
            throw std::invalid_argument("RedshiftBinWidth must be > 0.");

        if (NUMBER_OF_Z_BINS <= 0)
            throw std::invalid_argument("NumberOfRedshiftBins must be > 0.");

        // Redshift and wavenumber bins are constructed
        bins::setUpBins(K_0, N_KLIN_BIN, LIN_K_SPACING, N_KLOG_BIN,
            LOG_K_SPACING, klast, Z_0);

        LOG::LOGGER.STD("K0 %.3e\n", K_0);
        LOG::LOGGER.STD("LinearKBinWidth %.3e\n", LIN_K_SPACING);
        LOG::LOGGER.STD("NumberOfLinearBins %d\n", N_KLIN_BIN);
        LOG::LOGGER.STD("Log10KBinWidth %.3e\n", LOG_K_SPACING);
        LOG::LOGGER.STD("NumberOfLog10Bins %d\n", N_KLOG_BIN);
        LOG::LOGGER.STD("LastKEdge %.3e\n", klast);

        LOG::LOGGER.STD("FirstRedshiftBinCenter %.3e\n", Z_0);
        LOG::LOGGER.STD("RedshiftBinWidth %.3e\n", Z_BIN_WIDTH);
        LOG::LOGGER.STD("NumberOfRedshiftBins %d\n\n", NUMBER_OF_Z_BINS);
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
    void zBinTriangular1(const double *z, int N, int zm, double *out, int &low, int &up)
    {
        int zmm __attribute__((unused)) = zm;
        double zc = ZBIN_CENTERS[0];

        low = std::lower_bound(z, z + N, zc) - z;
        up = std::upper_bound(z, z + N, zc + Z_BIN_WIDTH) - z;

        std::fill(out, out + low, 1);
        #pragma omp simd
        for (int i = low; i < up; ++i)
            out[i] = 1 - (z[i] - zc) / Z_BIN_WIDTH;
        // std::fill(out + up, out + N, 0);
        low = 0;
    }

    // binning function for last zm
    void zBinTriangular2(const double *z, int N, int zm, double *out, int &low, int &up)
    {
        int zmm __attribute__((unused)) = zm;
        double zc = ZBIN_CENTERS[NUMBER_OF_Z_BINS - 1];

        low = std::lower_bound(z, z + N, zc - Z_BIN_WIDTH) - z;
        up = std::upper_bound(z, z + N, zc) - z;

        // std::fill(out, out + low, 0);
        #pragma omp simd
        for (int i = low; i < up; ++i)
            out[i] = 1 - (zc - z[i]) / Z_BIN_WIDTH;
        std::fill(out + up, out + N, 1);
        up = N;
    }

    // binning functio for non-boundary zm
    void zBinTriangular(const double *z, int N, int zm, double *out, int &low, int &up)
    {
        double zc = ZBIN_CENTERS[zm];
        low = std::lower_bound(z, z + N, zc - Z_BIN_WIDTH) - z;
        up = std::upper_bound(z, z + N, zc + Z_BIN_WIDTH) - z;

        // std::fill_n(out, N, 0);
        #pragma omp simd
        for (int i = low; i < up; ++i)
            out[i] = 1 - fabs(z[i] - zc) / Z_BIN_WIDTH;
    }

    void zBinTopHat(const double *z, int N, int zm, double *out, int &low, int &up)
    {
        double zc = ZBIN_CENTERS[zm];
        low = std::lower_bound(z, z + N, zc - Z_BIN_WIDTH / 2) - z;
        up = std::upper_bound(z, z + N, zc + Z_BIN_WIDTH / 2) - z;

        std::fill_n(out, N, 0);
        #pragma omp simd
        for (int i = low; i < up; ++i)
            out[i] = 1;
    }

    void (*redshiftBinningFunction)(
        const double *z, int N, int zm, double *out, int &low, int &up) = &zBinTopHat;

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







