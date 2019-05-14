// #include "fiducial_cosmology.hpp"
#include "global_numbers.hpp"

#include <cmath>

double fiducial_power_spectrum(double k, double z, void *params)
{
    #ifdef PD13_FIT_FUNCTION
    {
        pd13_fit_params *pfp = (pd13_fit_params *) params;
        return Palanque_Delabrouille_etal_2013_fit(k, z, pfp);   
    }
    #endif

    #ifdef DEBUG_FIT_FUNCTION
    {
        double zz __attribute__((unused)) = z;
        void  *pp __attribute__((unused)) = params;
        return debuggin_power_spectrum(k);
    }
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

    result *= result / PI; // * cos(k * wp->delta_v_ij)

    return result;
}

double debuggin_power_spectrum(double k)
{
    const double dv = 200.;

    double kc = PI / dv / 2.0;

    double r = k/kc;

    return dv * r * exp(- r*r);
}

double lnpoly2_power_spectrum(double lnk)
{
    double  c0 = -7.89e-01, \
            c1 = 1.30e-01, \
            c2 = -1.87e-02;

    double lnkP = c0 + c1 * lnk + c2 * lnk*lnk;

    return exp(lnkP);
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













