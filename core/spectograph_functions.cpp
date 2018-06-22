#include "spectograph_functions.hpp"

#include <cmath>

double R_SPECTOGRAPH;

double sinc(double x)
{
    if (abs(x) < 1E-10)
    {
        return 1.;
    }

    return sin(x) / x;
}

void convert_lambda2v(double *lambda, int size)
{
    #define SPEED_OF_LIGHT 299792.458
    #define LYA_REST 1215.67

    double mean_lambda = 0.;

    for (int i = 0; i < size; i++)
    {
        mean_lambda += lambda[i] / size;
    }

    for (int i = 0; i < size; i++)
    {
        lambda[i] = 2. * SPEED_OF_LIGHT * (1. - sqrt(mean_lambda / lambda[i]));
    }
}

double spectral_response_window_fn(double k, void *params)
{
    struct spectograph_windowfn_params *wp = (struct spectograph_windowfn_params*) params;

    double  R = R_SPECTOGRAPH, \
            dv_kms = wp->pixel_width;

    return exp(-k*k * R*R / 2.) * sinc(k * dv_kms / 2.);
}


