#ifndef GLOBAL_NUMBERS_H
#define GLOBAL_NUMBERS_H

#include <cstdio>
#include <chrono>
#include <string>
#include <memory>
#include <vector>

#include "io/config_file.hpp"

#define __LYSPEQ_VERSION__ "4.6.1"

// Debugging flags. Comment out to turn off
// #define DEBUG_MATRIX_OUT

const double
SPEED_OF_LIGHT = 299792.458,
LYA_REST = 1215.67,
MY_PI = 3.14159265359,
ONE_SIGMA_2_FWHM = 2.35482004503,
DOUBLE_EPSILON = 1e-15;

namespace qio
{
    enum ifileformat {Binary, Picca};
}

// PE rank and total number of threads
namespace process
{
    extern int this_pe, total_pes;
    extern std::string TMP_FOLDER;
    extern std::string FNAME_BASE;
    extern double MEMORY_ALLOC;
    extern bool SAVE_EACH_PE_RESULT, SAVE_EACH_CHUNK_RESULT;
    extern bool SAVE_ALL_SQ_FILES;

    const config_map process_default_parameters ({
        {"OutputDir", "."}, {"OutputFileBase", "qmle"},
        {"TemporaryFolder", "."}, {"SaveEachProcessResult", "-1"},
        {"SaveEachChunkResult", "-1"},
        {"CacheAllSQTables", "1"} , {"AllocatedMemoryMB", "1500."}});

    inline double getMemoryMB(int n) {
        return (double)sizeof(double) * n / 1048576.;
    };

    void updateMemory(double deltamem);

    /* This function reads following keys from config file:
    OutputDir: string
        Output directory. Current dir by default.
    OutputFileBase: string
        Base string for output files. "qmle" by default.
    TemporaryFolder: string
        Folder to save temporary smooting files. Current dir by default.
    SaveEachProcessResult: int
        Pass > 0 to enable mpi saving each result from pe. Turned off
        by default.
    SaveEachChunkResult: int
        Pass > 0 to enable saving each result from chunks. Turned off
        by default.
    CacheAllSQTables: int
        Pass > 0 to cache all SQ tables into memory. Otherwise, read
        each file when needed. On by default.
    AllocatedMemoryMB: double
        Allocated memory in MB, so that QMLE can save matrices into
        memory. Default 1500.
    */
    void readProcess(ConfigFile &config);
}

namespace specifics
{
    extern bool
        TURN_OFF_SFID, SMOOTH_LOGK_LOGP, USE_RESOLUTION_MATRIX, 
        USE_PRECOMPUTED_FISHER, FAST_BOOTSTRAP;
    extern double CHISQ_CONVERGENCE_EPS;
    extern int
        CONT_LOGLAM_MARG_ORDER, CONT_LAM_MARG_ORDER, CONT_NVECS,
        NUMBER_OF_CHUNKS, NUMBER_OF_BOOTS, OVERSAMPLING_FACTOR;
    extern double RESOMAT_DECONVOLUTION_M;
    extern qio::ifileformat INPUT_QSO_FILE;

    const config_map specifics_default_parameters ({
        {"InputIsPicca", "-1"}, {"UseResoMatrix", "-1"},
        {"ResoMatDeconvolutionM", "-1"}, {"OversampleRmat", "-1"},
        {"DynamicChunkNumber", "1"}, {"TurnOffBaseline", "-1"},
        {"SmoothLnkLnP", "1"}, {"ChiSqConvergence", "1e-2"},
        {"ContinuumLogLambdaMargOrder", "1"}, {"ContinuumLambdaMargOrder", "-1"},
        {"PrecomputedFisher", ""},
        {"NumberOfBoots", "20000"}, {"FastBootstrap", "1"} });

    /* This function reads following keys from config file:
    InputIsPicca: int
        If > 0, input file format is from picca. Off by default.
    UseResoMatrix: int
        If > 0, reads and uses the resolution matrix picca files.
        Off by default.
    ResoMatDeconvolutionM: double
        Deconvolve the resolution matrix by this factor in terms of pixel.
        For example, 1.0 deconvolves one top hat. Off by default and when
        <= 0.
    OversampleRmat: int
        Oversample the resolution matrix by this factor per row. Off when <= 0
        and by default.
    DynamicChunkNumber: int
        Dynamiccaly chunk spectra into this number when > 1. Off by default.
    TurnOffBaseline: int
        Turns off the fiducial signal matrix if > 0. Fid is on by default.
    SmoothLnkLnP: int
        Smooth the ln k and ln P values when iterating. On by default
        and when > 0.
    ChiSqConvergence: int
        Criteria for chi square convergance. Valid when > 0. Default is 1e-2
    ContinuumLogLambdaMargOrder: int
        Polynomial order for log lambda cont marginalization. Default 1.
    ContinuumLambdaMargOrder: int
        Polynomial order for lambda cont marginalization. Default -1.
    PrecomputedFisher: string
        File to precomputed Fisher matrix. If present, Fisher matrix is not
        calculated for spectra. Off by default.
    NumberOfBoots: int
        Number of bootstrap realizations.
    FastBootstrap: int
        Fast bootstrap method. Does not recalculates the Fisher matrix. Default 1
    */
    void readSpecifics(ConfigFile &config);

    void printBuildSpecifics(FILE *toWrite=NULL);
}

namespace bins
{
    enum BinningMethod { TophatBinningMethod, TriangleBinningMethod };
    // Binning numbers
    // One last bin is created when LAST_K_EDGE is set in config to absorb high k power such as alias effect.
    // This last bin is calculated into covariance matrix, smooth power spectrum fitting or convergence test.
    // TOTAL_KZ_BINS = NUMBER_OF_K_BANDS * NUMBER_OF_Z_BINS
    extern int
        NUMBER_OF_K_BANDS, NUMBER_OF_Z_BINS, FISHER_SIZE, TOTAL_KZ_BINS,
        NewDegreesOfFreedom;
    extern std::vector<double> KBAND_EDGES, KBAND_CENTERS, ZBIN_CENTERS;
    extern double Z_BIN_WIDTH, Z_LOWER_EDGE, Z_UPPER_EDGE;
    extern BinningMethod Z_BINNING_METHOD;

    const config_map bins_default_parameters ({
        {"K0", "0"}, {"LastKEdge", "-1"}, {"RedshiftBinningMethod", "1"}
    });
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
    RedshiftBinningMethod: int
    FirstRedshiftBinCenter: double
    RedshiftBinWidth: double
    NumberOfRedshiftBins: double
    */
    void readBins(ConfigFile &config);
    // void setUpBins(double k0, int nlin, double dklin, int nlog, double dklog, double klast, double z0);

    // returns -1 if below, NUMBER_OF_Z_BINS if above
    int findRedshiftBin(double z);

    // Given the redshift z, returns binning weight. 1 for top-hats, interpolation for triangular
    // zm: Bin number to consider
    void setRedshiftBinningFunction(int zm);
    // redshiftBinningFunction takes z array and assumes it is sorted.
    extern void (*redshiftBinningFunction)(
        const double *z, int N, int zm, double *out, int &low, int &up);

    inline
    int getFisherMatrixIndex(int kn, int zm)
    { return kn + NUMBER_OF_K_BANDS * zm; }

    inline
    void getFisherMatrixBinNoFromIndex(int ikz, int &kn, int &zm)
    {
        kn = (ikz) % NUMBER_OF_K_BANDS;
        zm = (ikz) / NUMBER_OF_K_BANDS;
    }
}

namespace mytime
{
    // Keeping track of time
    extern double time_spent_on_c_inv, time_spent_on_f_inv;
    extern double time_spent_on_set_sfid, time_spent_set_qs,
                  time_spent_set_modqs, time_spent_set_fisher;

    extern double time_spent_on_q_interp, time_spent_on_q_copy;
    extern long   number_of_times_called_setq, number_of_times_called_setsfid;

    void printfTimeSpentDetails();
    void writeTimeLogHeader();

    class Timer
    {
        using steady_c  = std::chrono::steady_clock;
        using minutes_t = std::chrono::duration<double, std::ratio<60>>;

        std::chrono::time_point<steady_c> m0;
    public:
        Timer() : m0(steady_c::now()) {};
        ~Timer() {};

        double getTime() const
        {
            return std::chrono::duration_cast<minutes_t>(steady_c::now() - m0).count();
        } 
    };

    extern Timer timer;
}

#endif
