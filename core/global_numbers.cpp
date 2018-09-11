#include "global_numbers.hpp"

#include <cstdio>
int NUMBER_OF_K_BANDS, NUMBER_OF_Z_BINS, TOTAL_KZ_BINS;
double *KBAND_EDGES, *KBAND_CENTERS;
double Z_BIN_WIDTH, *ZBIN_CENTERS;

float   time_spent_on_c_inv    = 0, time_spent_on_f_inv   = 0;
float   time_spent_on_set_sfid = 0, time_spent_set_qs     = 0, \
        time_spent_set_modqs   = 0, time_spent_set_fisher = 0;

void printf_time_spent_details()
{
    printf("Total time spent on inverting C is %.2f mins.\n", time_spent_on_c_inv / 60.);
    printf("Total time spent on inverting F is %.2f mins.\n", time_spent_on_f_inv / 60.);

    printf("Total time spent on setting Sfid is %.2f mins.\n", time_spent_on_set_sfid / 60.);
    printf("Total time spent on setting Qs is %.2f mins.\n", time_spent_set_qs / 60.);
    printf("Total time spent on setting Mod Qs is %.2f mins.\n", time_spent_set_modqs / 60.);
    printf("Total time spent on setting F is %.2f mins.\n", time_spent_set_fisher / 60.);
}
