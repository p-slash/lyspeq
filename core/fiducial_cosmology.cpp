#include "core/fiducial_cosmology.hpp"

#include <cmath>

double sinc(double x)
{
    if (fabs(x) < 1E-6)     return 1.;

    return sin(x) / x;
}

// bool check_linearly_spaced(double *v, int size)
// {
//     double dv, cv = v[1]-v[0], eps = 1e-3;

//     for (int i = 1; i < size-1; i++)
//     {
//         dv = v[i+1] - v[i];

//         if (fabs(dv - cv) > cv * eps)   return false;
//     }

//     return true;
// }

// double becker13_meanflux(double z)
// {
//     double tau = 0.751 * pow((1. + z) / 4.5, 2.90) - 0.132;

//     return exp(-tau);
// }

// void convert_flux2deltaf_becker(const double *lambda, double *flux, double *noise, int size)
// {
//     double mean_f, z_i;

//     for (int i = 0; i < size; i++)
//     {
//         z_i       = lambda[i] / LYA_REST - 1.;
//         mean_f    = becker13_meanflux(z_i);
//         flux[i]   = (flux[i] / mean_f) - 1.;
//         noise[i] /= mean_f;
//     }
// }

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

double fiducial_power_spectrum(double k, double z, void *params)
{
    // Using Palanque-Delabrouille function
    #ifdef PD13_FIT_FUNCTION
    
    pd13_fit_params *pfp = (pd13_fit_params *) params;
    return Palanque_Delabrouille_etal_2013_fit(k, z, pfp);   
    
    #endif

    // If you want to use another function
    // Code and pass to preprocessor
    #ifdef SOME_OTHER_FIT

    return another_fitting_function(k, z, params);
    
    #endif
}

double signal_matrix_integrand(double k, void *params)
{
    struct sq_integrand_params          *sqip = (struct sq_integrand_params*) params;
    struct spectrograph_windowfn_params *wp   = sqip->spec_window_params;
    pd13_fit_params                     *pfp  = sqip->fiducial_pd_params;

    double result = spectral_response_window_fn(k, wp);

    result *= fiducial_power_spectrum(k, wp->z_ij, pfp) * result / PI;

    return result;
}

double q_matrix_integrand(double k, void *params)
{
    struct sq_integrand_params          *sqip = (struct sq_integrand_params*) params;
    struct spectrograph_windowfn_params *wp   = sqip->spec_window_params;

    double result = spectral_response_window_fn(k, wp);

    result *= result / PI;

    return result;
}

double Palanque_Delabrouille_etal_2013_fit(double k, double z, pd13_fit_params *params)
{
    #define K_0 0.009 // s km^-1
    #define Z_0 3.0

    double  q0 = k / K_0 + 1E-10, \
            x0 = (1. + z) / (1. + Z_0);

    return    (params->A * PI / K_0) \
            * pow(q0,   2. + params->n + \
                        params->alpha * log(q0) + \
                        params->beta  * log(x0)) \
            * pow(x0,   params->B) \
            / (1. + params->lambda * k * k);

    #undef K_0
    #undef Z_0
}

double another_fitting_function(double k, double z, void *params)
{
    // Code your own fitting function here
    void  *pp __attribute__((unused)) = params; // remove this line if using params
    return k*z;
}











