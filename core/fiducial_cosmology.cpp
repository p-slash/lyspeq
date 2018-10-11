#include "fiducial_cosmology.hpp"
#include "global_numbers.hpp"

#include <cmath>

double fiducial_power_spectrum(double k, double z, void *params)
{
    #ifdef PD13_FIT_FUNCTION
    {
        struct palanque_fit_params *pfp = (struct palanque_fit_params *) params;
        return Palanque_Delabrouille_etal_2013_fit(k, z, pfp);   
    }
    #endif

    #ifdef DEBUG_FIT_FUNCTION
    {
        return debuggin_power_spectrum(k);
    }
    #endif
}

double signal_matrix_integrand(double k, void *params)
{
    struct sq_integrand_params          *sqip = (struct sq_integrand_params*) params;
    struct spectrograph_windowfn_params *wp   = sqip->spec_window_params;
    struct palanque_fit_params          *pfp  = sqip->fiducial_pd_params;

    double result = spectral_response_window_fn(k, wp);

    result *= fiducial_power_spectrum(k, wp->z_ij, pfp) * result / PI;

    return result;
}

double q_matrix_integrand(double k, void *params)
{
    struct sq_integrand_params          *sqip = (struct sq_integrand_params*) params;
    struct spectrograph_windowfn_params *wp   = sqip->spec_window_params;

    double result = spectral_response_window_fn(k, wp);

    result *= result * cos(k * wp->delta_v_ij) / PI;

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

double Palanque_Delabrouille_etal_2013_fit(double k, double z, struct palanque_fit_params *params)
{
    const double k_0 = 0.009; // s/km
    const double z_0 = 3.0;
    double  q = k + 1E-9;

    // if (k < 1E-5)
    // {
    //     q = 1E-5;
    // }
    
    double  lnk = log(q / k_0), \
            lnz = log((1.+z) / (1.+z_0)),\
            lnkP_pi = 0;

    lnkP_pi = log(params->A) \
            + (3. + params->n) * lnk \
            + params->alpha * lnk * lnk \
            + (params->B + params->beta * lnk) * lnz;

    return exp(lnkP_pi) * PI / q;
}
