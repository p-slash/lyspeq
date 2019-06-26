#include "core/fiducial_cosmology.hpp"

#include <cmath>

// Palanque_Delabrouille et al. 2013 based functions. Extra Lorentzian decay
namespace pd13
{
// Defined values for PD fit
#define K_0 0.009 // s km^-1
#define Z_0 3.0

    pd13_fit_params FIDUCIAL_PD13_PARAMS;

    double Palanque_Delabrouille_etal_2013_fit(double k, double z, pd13_fit_params *params)
    {
        double  q0 = k / K_0 + 1E-10, \
                x0 = (1. + z) / (1. + Z_0);

        return    (params->A * PI / K_0) \
                * pow(q0,   2. + params->n + \
                            params->alpha * log(q0) + \
                            params->beta  * log(x0)) \
                * pow(x0,   params->B) \
                / (1. + params->lambda * k * k);
    }

#undef Z_0 // Do not need Z_0 after this point

    double Palanque_Delabrouille_etal_2013_fit_growth_factor(double z_ij, double k_kn, double z_zm, pd13_fit_params *params)
    {
        double  q0 = k_kn / K_0 + 1E-10, \
                x0 = (1. + z_ij) / (1. + z_zm);

        return  pow(x0, params->B + params->beta  * log(q0));
    }

#undef K_0 // Do not need K_0 either
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


// Conversion functions
namespace conv
{
    bool USE_LOG_V = false, USE_FID_LEE12_MEAN_FLUX = false;
    
    void convertLambdaToVelocity(double &median_z, double *v_array, const double *lambda, int size)
    {
        double median_lambda = lambda[size / 2];
    
        median_z = median_lambda / LYA_REST - 1.;

        if (USE_LOG_V)
        {
            for (int i = 0; i < size; ++i)
                v_array[i] = SPEED_OF_LIGHT * log(lambda[i]/median_lambda);
        }
        else
        {
            for (int i = 0; i < size; ++i)
                v_array[i] = 2. * SPEED_OF_LIGHT * (1. - sqrt(median_lambda / lambda[i]));
        }
    }

    void convertLambdaToRedshift(double *lambda, int size)
    {
        for (int i = 0; i < size; ++i)
        {
            lambda[i] /= LYA_REST;
            lambda[i] -= 1.;
        }
    }

    double meanFluxBecker13(double z)
    {
        double tau = 0.751 * pow((1. + z) / 4.5, 2.90) - 0.132;
        return exp(-tau);
    }

    double meanFluxLee12(double z)
    {
        double tau = 0.001845 * pow(1. + z, 3.924);
        return exp(-tau);
    }

    void convertFluxToDeltaf(double *flux, double *noise, int size)
    {
        double mean_f = 0.;

        for (int i = 0; i < size; ++i)
            mean_f += flux[i];
        mean_f /= size;

        for (int i = 0; i < size; ++i)
        {
            flux[i]  /= mean_f;
            flux[i]  -= 1.;
            noise[i] /= mean_f;
        }
    }

    void convertFluxToDeltafLee12(const double *lambda, double *flux, double *noise, int size)
    {
        double mean_f;

        for (int i = 0; i < size; ++i)
        {
            mean_f    = meanFluxLee12(lambda[i]/LYA_REST-1);
            flux[i]  /= mean_f;
            flux[i]  -= 1.;
            noise[i] /= mean_f;
        }
    }
}

// Fiducial Functions

// Using Palanque-Delabrouille fit
// Add #ifdef to implement other functions
// #ifdef SOME_OTHER_FIT

// return another_fitting_function(k, z, params);

// #endif
// Code another fitting function here
// double another_fitting_function(double k, double z, void *params)
// {
//     // Code your own fitting function here
//     void  *pp __attribute__((unused)) = params; // remove this line if using params
//     return k*z;
// }

// double another_fitting_function_growth_factor(double z_ij, double k_kn, double z_zm, void *params)
// {
//     // Code your own fitting function here
//     void  *pp __attribute__((unused)) = params; // remove this line if using params
//     return k_kn*z_ij/z_zm;
// }

namespace fidcosmo
{
    // This function is defined in preprocessing
    double fiducialPowerSpectrum(double k, double z, void *params)
    {    
        pd13::pd13_fit_params *pfp = (pd13::pd13_fit_params *) params;
        return pd13::Palanque_Delabrouille_etal_2013_fit(k, z, pfp);   
    }
    double fiducialPowerGrowthFactor(double z_ij, double k_kn, double z_zm, void *params)
    {
        pd13::pd13_fit_params *pfp = (pd13::pd13_fit_params *) params;
        return pd13::Palanque_Delabrouille_etal_2013_fit_growth_factor(z_ij, k_kn, z_zm, pfp);
    }
}

// Signal and Derivative Integrands and Window Function
double sinc(double x)
{
    if (fabs(x) < 1E-6)     return 1.;

    return sin(x) / x;
}

double spectral_response_window_fn(double k, struct spectrograph_windowfn_params *spec_params)
{
    double R = spec_params->spectrograph_res, dv_kms = spec_params->pixel_width;

    return sinc(k * dv_kms / 2.) * exp(-k*k * R*R / 2.);
}

double signal_matrix_integrand(double k, void *params)
{
    struct sq_integrand_params          *sqip = (struct sq_integrand_params*) params;
    struct spectrograph_windowfn_params *wp   = sqip->spec_window_params;
    pd13::pd13_fit_params               *pfp  = sqip->fiducial_pd_params;

    double result = spectral_response_window_fn(k, wp);

    result *= fidcosmo::fiducialPowerSpectrum(k, wp->z_ij, pfp) * result / PI;

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












