#include "spectograph_functions.hpp"

#include <cmath>
#include <cstdio>

double R_SPECTOGRAPH;

double sinc(double x)
{
    if (fabs(x) < 1E-10)
    {
        return 1.;
    }

    return sin(x) / x;
}

bool check_linearly_spaced(double *v, int size)
{
    double dv, cv = v[1]-v[0], eps = 1e-3;

    printf("first 5 dv: %lf", cv);

    for (int i = 1; i < size-1; i++)
    {
        dv = v[i+1] - v[i];

        if (i < 5)
        {
            printf(" %lf", dv);
        }

        if (fabs(dv - cv) > cv * eps)
        {
            printf("\n");
            return false;
        }
    }
    printf("\n");
    return true;
}

void convert_flux2deltaf(double *flux, int size)
{
    double mean_f = 0;

    for (int i = 0; i < size; i++)
    {
        mean_f += flux[i] / size;
    }

    for (int i = 0; i < size; i++)
    {
        flux[i] = (flux[i] / mean_f) - 1.;
    }
}

void convert_lambda2v(double &mean_z, double *lambda, int size)
{
    #define SPEED_OF_LIGHT 299792.458
    #define LYA_REST 1215.67

    double mean_lambda = 0; //lambda[size/2];

    for (int i = 0; i < size; i++)
    {
        mean_lambda += lambda[i] / (1.0 * size);
    }

    mean_z = (mean_lambda / LYA_REST) - 1.;
    
    for (int i = 0; i < size; i++)
    {
        lambda[i] = 2. * SPEED_OF_LIGHT * (1. - sqrt(mean_lambda / lambda[i]));
    }

    if (!check_linearly_spaced(lambda, size))
        printf("WARNING: NOT Linearly spaced!\n");
}

double spectral_response_window_fn(double k, void *params)
{
    struct spectograph_windowfn_params *wp = (struct spectograph_windowfn_params*) params;

    double  R = R_SPECTOGRAPH, \
            dv_kms = wp->pixel_width;

    return exp(-k*k * R*R / 2.) * sinc(k * dv_kms / 2.);
}


