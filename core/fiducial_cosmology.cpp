#include "fiducial_cosmology.hpp"
#include "spectrograph_functions.hpp"

#include <cmath>
#define PI 3.14159265359

double fiducial_power_spectrum(double k, double z, void *params)
{
    struct spectrograph_windowfn_params *wp = (struct spectrograph_windowfn_params*) params;
    return debuggin_power_spectrum(k, wp->pixel_width);
}

double signal_matrix_integrand(double k, void *params)
{
    struct spectrograph_windowfn_params *wp = (struct spectrograph_windowfn_params*) params;
    double result = spectral_response_window_fn(k, params);

    result *= fiducial_power_spectrum(k, wp->z_ij, params) * result * cos(k * wp->delta_v_ij) / PI;

    return result;
}

double q_matrix_integrand(double k, void *params)
{
    struct spectrograph_windowfn_params *wp = (struct spectrograph_windowfn_params*) params;
    double result = spectral_response_window_fn(k, params);

    result *= result * cos(k * wp->delta_v_ij) / PI;

    return result;
}

double debuggin_power_spectrum(double k, double dv)
{
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

double Palanque_Delabrouille_etal_2013_fit(double k, double z, struct palanque_fit_params &params)
{
    const double k_0 = 0.009; // s/km
    const double z_0 = 3.0;

    double  lnk = log(k / k_0), \
            lnz = log((1.+z) / (1.+z_0)),\
            lnkP_pi = 0;

    lnkP_pi = log(params.A) \
            + (3. + params.n) * lnk \
            + params.alpha * lnk * lnk \
            + (params.B + params.beta * lnk) * lnz;

    return exp(lnkP_pi) * PI / k;
}
