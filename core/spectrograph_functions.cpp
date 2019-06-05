#include "core/spectrograph_functions.hpp"
#include "core/global_numbers.hpp"

#include <cmath>
#include <cstdio>

double sinc(double x)
{
    if (fabs(x) < 1E-6)     return 1.;

    return sin(x) / x;
}

bool check_linearly_spaced(double *v, int size)
{
    double dv, cv = v[1]-v[0], eps = 1e-3;

    for (int i = 1; i < size-1; i++)
    {
        dv = v[i+1] - v[i];

        if (fabs(dv - cv) > cv * eps)   return false;
    }

    return true;
}

double becker13_meanflux(double z)
{
    double tau = 0.751 * pow((1. + z) / 4.5, 2.90) - 0.132;

    return exp(-tau);
}

void convert_flux2deltaf_mean(double *flux, double *noise, int size)
{
    double mean_f = 0.;

    for (int i = 0; i < size; i++)
        mean_f += flux[i] / size;

    for (int i = 0; i < size; i++)
    {
        flux[i]   = (flux[i] / mean_f) - 1.;
        noise[i] /= mean_f;
    }
}
void convert_flux2deltaf_becker(const double *lambda, double *flux, double *noise, int size)
{
    double mean_f, z_i;

    for (int i = 0; i < size; i++)
    {
        z_i       = lambda[i] / LYA_REST - 1.;
        mean_f    = becker13_meanflux(z_i);
        flux[i]   = (flux[i] / mean_f) - 1.;
        noise[i] /= mean_f;
    }
}

void convert_lambda2v(double &median_z, double *v_array, const double *lambda, int size)
{
    double median_lambda = lambda[size / 2];
    
    median_z = median_lambda / LYA_REST - 1.;

    for (int i = 0; i < size; i++)
        v_array[i] = 2. * SPEED_OF_LIGHT * (1. - sqrt(median_lambda / lambda[i]));
}

double spectral_response_window_fn(double k, struct spectrograph_windowfn_params *spec_params)
{
    double R = spec_params->spectrograph_res, dv_kms = spec_params->pixel_width;

    return sinc(k * dv_kms / 2.) * exp(-k*k * R*R / 2.);
}


