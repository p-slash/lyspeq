#include "global_numbers.hpp"

#include <cstdio>
#include <cmath>

#if defined(_OPENMP)
#include <omp.h> /*omp_get_wtime();*/
#else
#include <ctime>    /* clock_t, clock, CLOCKS_PER_SEC */
#endif

int NUMBER_OF_K_BANDS, NUMBER_OF_Z_BINS, TOTAL_KZ_BINS;

SQLookupTable *sq_lookup_table;

double *KBAND_EDGES, *KBAND_CENTERS;
double  Z_BIN_WIDTH, *ZBIN_CENTERS;

float   time_spent_on_c_inv    = 0, time_spent_on_f_inv   = 0;
float   time_spent_on_set_sfid = 0, time_spent_set_qs     = 0, \
        time_spent_set_modqs   = 0, time_spent_set_fisher = 0;

bool TURN_OFF_SFID;

void printf_time_spent_details()
{
    printf("Total time spent on inverting C is %.2f mins.\n", time_spent_on_c_inv);
    printf("Total time spent on inverting F is %.2f mins.\n", time_spent_on_f_inv);

    printf("Total time spent on setting Sfid is %.2f mins.\n",   time_spent_on_set_sfid);
    printf("Total time spent on setting Qs is %.2f mins.\n",     time_spent_set_qs     );
    printf("Total time spent on setting Mod Qs is %.2f mins.\n", time_spent_set_modqs  );
    printf("Total time spent on setting F is %.2f mins.\n",      time_spent_set_fisher );
}

void set_up_bins(double k0, int nlin, double dklin, \
                            int nlog, double dklog, \
                 double z0)
{
    // Construct k edges
    NUMBER_OF_K_BANDS = nlin + nlog;
    TOTAL_KZ_BINS     = NUMBER_OF_K_BANDS * NUMBER_OF_Z_BINS;

    KBAND_EDGES   = new double[NUMBER_OF_K_BANDS + 1];
    KBAND_CENTERS = new double[NUMBER_OF_K_BANDS];

    for (int i = 0; i < nlin + 1; i++)
    {
        KBAND_EDGES[i] = k0 + dklin * i;
    }
    for (int i = 1, j = nlin + 1; i < nlog + 1; i++, j++)
    {
        KBAND_EDGES[j] = KBAND_EDGES[nlin] * pow(10., i * dklog);
    }
    for (int kn = 0; kn < NUMBER_OF_K_BANDS; kn++)
    {
        KBAND_CENTERS[kn] = (KBAND_EDGES[kn] + KBAND_EDGES[kn + 1]) / 2.;
    }

    // Construct redshift bins
    ZBIN_CENTERS = new double[NUMBER_OF_Z_BINS];

    for (int zm = 0; zm < NUMBER_OF_Z_BINS; ++zm)
    {
        ZBIN_CENTERS[zm] = z0 + Z_BIN_WIDTH * zm;
    }
}

void clean_up_bins()
{
    delete [] KBAND_EDGES;
    delete [] KBAND_CENTERS;
    delete [] ZBIN_CENTERS;
}

float get_time()
{
    #if defined(_OPENMP)
    return omp_get_wtime() / 60.;
    #else
    clock_t t = clock();
    return ((float) t) / CLOCKS_PER_SEC / 60.;
    #endif
}
