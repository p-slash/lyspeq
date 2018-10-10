#ifndef FIDUCIAL_COSMOLOGY_H
#define FIDUCIAL_COSMOLOGY_H

#include "spectrograph_functions.hpp"

struct palanque_fit_params
{
    double A;
    double n;
    double alpha;

    double B;
    double beta;
};

struct sq_integrand_params
{
    struct palanque_fit_params          *fiducial_pd_params;
    struct spectrograph_windowfn_params *spec_window_params;
};

double fiducial_power_spectrum(double k, double z, void *params);

double q_matrix_integrand(double k, void *params);

double signal_matrix_integrand(double k, void *params);

double debuggin_power_spectrum(double k);

double lnpoly2_power_spectrum(double lnk);

double Palanque_Delabrouille_etal_2013_fit(double k, double z, struct palanque_fit_params *params);

#endif
