// Mathematical numbers
#define PI 3.14159265359

// Physical constants
#define LYA_REST 1215.67
#define SPEED_OF_LIGHT 299792.458

// One QSO Estimate numbers
#define ADDED_CONST_TO_COVARIANCE 10.0

// Quadratic Estimate numbers
#define CONVERGENCE_EPS 1E-4

// Binning numbers
extern int NUMBER_OF_K_BANDS, NUMBER_OF_Z_BINS, TOTAL_KZ_BINS;
extern double *KBAND_EDGES, *KBAND_CENTERS;
extern double Z_BIN_WIDTH, *ZBIN_CENTERS;

// Keeping track of time
extern float time_spent_on_c_inv, time_spent_on_f_inv;
extern float time_spent_on_set_sfid, time_spent_set_qs, \
             time_spent_set_modqs, time_spent_set_fisher;

void printf_time_spent_details();
