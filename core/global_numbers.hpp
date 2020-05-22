#ifndef GLOBAL_NUMBERS_H
#define GLOBAL_NUMBERS_H

#include "core/sq_table.hpp"
#include <cstdio>
#include <chrono>

// Quadratic Estimate numbers
#define CONVERGENCE_EPS 1E-4

// PE rank and total number of threads
namespace process
{
    extern int this_pe, total_pes;
    extern char TMP_FOLDER[300];
    extern double MEMORY_ALLOC;
    // Look up table, global and thread copy
    extern SQLookupTable *sq_private_table;
}

namespace bins
{
    // Binning numbers
    // One last bin is created when LAST_K_EDGE is set in Makefile to absorb high k power such as alias effect.
    // This last bin is not calculated into covariance matrix, smooth power spectrum fitting or convergence test.
    // But it has a place in Fisher matrix and in power spectrum estimates.
    // NUMBER_OF_K_BANDS counts the last bin. So must use NUMBER_OF_K_BANDS - 1 when last bin ignored
    // TOTAL_KZ_BINS = NUMBER_OF_K_BANDS * NUMBER_OF_Z_BINS
    extern int NUMBER_OF_K_BANDS, NUMBER_OF_Z_BINS, TOTAL_KZ_BINS, DEGREE_OF_FREEDOM;
    extern double *KBAND_EDGES, *KBAND_CENTERS;
    extern double Z_BIN_WIDTH, *ZBIN_CENTERS;

    void setUpBins(double k0, int nlin, double dklin, int nlog, double dklog, double z0);
    void cleanUpBins();

    // returns -1 if below, NUMBER_OF_Z_BINS if above
    int findRedshiftBin(double z);

    // Given the redshift z, returns binning weight. 1 for top-hats, interpolation for triangular
    // zm: Bin number to consider
    // zc: Central bin number for triangular bins. Binning weights depend on being to the left 
    // or to the right of this number.
    double redshiftBinningFunction(double z, int zm);

    int getFisherMatrixIndex(int kn, int zm);
    void getFisherMatrixBinNoFromIndex(int ikz, int &kn, int &zm);
    
    #ifdef LAST_K_EDGE
    // #define SKIP_LAST_K_BIN_WHEN_ENABLED(x) if (((x)+1) % NUMBER_OF_K_BANDS == 0)   continue;
    #else
    #define SKIP_LAST_K_BIN_WHEN_ENABLED(x) 
    #endif
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

namespace specifics
{
    extern bool TURN_OFF_SFID, SMOOTH_LOGK_LOGP;
    extern double CHISQ_CONVERGENCE_EPS;
    extern double CONTINUUM_MARGINALIZATION_AMP, CONTINUUM_MARGINALIZATION_DERV;

    #if defined(TOPHAT_Z_BINNING_FN)
    #define BINNING_SHAPE "Top Hat"
    #elif defined(TRIANGLE_Z_BINNING_FN)
    #define BINNING_SHAPE "Triangular"
    #else
    #define BINNING_SHAPE "ERROR NOT DEFINED"
    #endif

    #define tostr(a) #a
    #define tovstr(a) tostr(a)
    
    #if defined(LAST_K_EDGE)
    #define HIGH_K_TXT tovstr(LAST_K_EDGE)
    #else
    #define HIGH_K_TXT "OFF"
    #endif
    
    #if defined(REDSHIFT_GROWTH_POWER)
    #define RGP_TEXT "ON"
    #else
    #define RGP_TEXT "OFF"
    #endif
    
    const char BUILD_SPECIFICS[] =  "# This version is build by the following options:\n"
                                    "# 1D Interpolation: " tovstr(INTERP_1D_TYPE) "\n"
                                    "# 2D Interpolation: " tovstr(INTERP_2D_TYPE) "\n"
                                    "# Redshift binning shape: " BINNING_SHAPE "\n" 
                                    "# Redshift growth scaling: " RGP_TEXT "\n"
                                    "# Last k bin: " HIGH_K_TXT "\n";
    #undef tostr
    #undef tovstr
    #undef BINNING_SHAPE
    #undef HIGH_K_TXT
    #undef RGP_TEXT
    #undef TORE_TEXT

    void printBuildSpecifics();
    void printConfigSpecifics(FILE *toWrite=NULL);
}

namespace ioh
{
    void readConfigFile(const char *FNAME_CONFIG, 
                        char *FNAME_LIST, char *FNAME_RLIST, char *INPUT_DIR, char *OUTPUT_DIR,
                        char *OUTPUT_FILEBASE, char *FILEBASE_S, char *FILEBASE_Q,
                        int *NUMBER_OF_ITERATIONS,
                        int *Nv, int *Nz, double *PIXEL_WIDTH, double *LENGTH_V);
}

#endif
