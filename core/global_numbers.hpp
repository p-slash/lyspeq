#ifndef GLOBAL_NUMBERS_H
#define GLOBAL_NUMBERS_H

#include "sq_table.hpp"

// Mathematical numbers
#define PI 3.14159265359
#define ONE_SIGMA_2_FWHM 2.35482004503

// Physical constants
#define LYA_REST 1215.67
#define SPEED_OF_LIGHT 299792.458

// One QSO Estimate numbers
#define ADDED_CONST_TO_COVARIANCE 10.0

// Quadratic Estimate numbers
#define CONVERGENCE_EPS       1E-4
#define CHISQ_CONVERGENCE_EPS 0.01

// Binning numbers
extern int NUMBER_OF_K_BANDS, NUMBER_OF_Z_BINS, TOTAL_KZ_BINS;
extern double *KBAND_EDGES, *KBAND_CENTERS;
extern double Z_BIN_WIDTH, *ZBIN_CENTERS;

// Keeping track of time
extern float time_spent_on_c_inv, time_spent_on_f_inv;
extern float time_spent_on_set_sfid, time_spent_set_qs, \
             time_spent_set_modqs, time_spent_set_fisher;

extern bool TURN_OFF_SFID;

extern SQLookupTable *sq_lookup_table;

void printf_time_spent_details();

void set_up_bins(double k0, int nlin, double dklin, \
                            int nlog, double dklog, \
                 double z0);

void clean_up_bins();

void read_config_file();

float get_time();
#endif
