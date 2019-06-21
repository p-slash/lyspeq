#ifndef GLOBAL_NUMBERS_H
#define GLOBAL_NUMBERS_H

#include "core/sq_table.hpp"

// Mathematical numbers defined in fiducial_cosmology.hpp
// #define PI 3.14159265359
#define ONE_SIGMA_2_FWHM 2.35482004503

// Physical constants defined in fiducial_cosmology.hpp
// #define LYA_REST 1215.67
// #define SPEED_OF_LIGHT 299792.458

// One QSO Estimate numbers
#define ADDED_CONST_TO_COVARIANCE 10.0

// Quadratic Estimate numbers
#define CONVERGENCE_EPS       1E-4
extern double CHISQ_CONVERGENCE_EPS;

extern char TMP_FOLDER[300];

// OpenMP thread rank and total number of threads
// t_rank is threadprivate
extern int t_rank, numthreads;

namespace bins
{
    // Binning numbers
    // One last bin is created when LAST_K_EDGE is set in Makefile to absorb high k power such as alias effect.
    // This last bin is not calculated into covariance matrix, smooth power spectrum fitting or convergence test.
    // But it has a place in Fisher matrix and in power spectrum estimates.
    // NUMBER_OF_K_BANDS counts the last bin. So must use NUMBER_OF_K_BANDS - 1 when last bin ignored
    // TOTAL_KZ_BINS = NUMBER_OF_K_BANDS * NUMBER_OF_Z_BINS
    extern int NUMBER_OF_K_BANDS, NUMBER_OF_Z_BINS, TOTAL_KZ_BINS;
    extern double *KBAND_EDGES, *KBAND_CENTERS;
    extern double Z_BIN_WIDTH, *ZBIN_CENTERS;

    void set_up_bins(double k0, int nlin, double dklin, \
                            int nlog, double dklog, \
                     double z0);
    void clean_up_bins();

    #ifdef LAST_K_EDGE
    #define SKIP_LAST_K_BIN_WHEN_ENABLED(x) if (((x)+1) % NUMBER_OF_K_BANDS == 0)   continue;
    #else
    #define SKIP_LAST_K_BIN_WHEN_ENABLED(x) 
    #endif
}

namespace mytime
{
    // Keeping track of time
    extern double time_spent_on_c_inv, time_spent_on_f_inv;
    extern double time_spent_on_set_sfid, time_spent_set_qs, \
                  time_spent_set_modqs, time_spent_set_fisher;

    extern double time_spent_on_q_interp, time_spent_on_q_copy;
    extern long   number_of_times_called_setq, number_of_times_called_setsfid;

    double get_time(); // in minutes
    void printf_time_spent_details();
}

extern double MEMORY_ALLOC;



extern bool TURN_OFF_SFID;

// Look up table, global and thread copy
extern SQLookupTable *sq_shared_table, *sq_private_table;

// OpenMP Threadprivate variables
#pragma omp threadprivate(sq_private_table, t_rank)

void print_build_specifics();

void read_config_file(  const char *FNAME_CONFIG, 
                        char *FNAME_LIST, char *FNAME_RLIST, char *INPUT_DIR, char *OUTPUT_DIR, \
                        char *OUTPUT_FILEBASE, char *FILEBASE_S, char *FILEBASE_Q, \
                        int *NUMBER_OF_ITERATIONS, \
                        int *Nv, int *Nz, double *PIXEL_WIDTH, double *LENGTH_V);


#endif
