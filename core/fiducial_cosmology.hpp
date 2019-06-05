#ifndef FIDUCIAL_COSMOLOGY_H
#define FIDUCIAL_COSMOLOGY_H

#include "core/spectrograph_functions.hpp"

// This function is defined in preprocessing
double fiducial_power_spectrum(double k, double z, void *params);

// Fitting function has the form in Palanque-Delabrouille et al. 2013
// Added a Lorentzian decay to suppress small scale power
typedef struct
{
    double A;
    double n;
    double alpha;

    double B;
    double beta;

    double lambda;
} pd13_fit_params;

double Palanque_Delabrouille_etal_2013_fit(double k, double z, pd13_fit_params *params);

// Data structures to integrate signal and derivative matrix expressions
struct sq_integrand_params
{
    pd13_fit_params                     *fiducial_pd_params;
    struct spectrograph_windowfn_params *spec_window_params;
};

double q_matrix_integrand(double k, void *params);

double signal_matrix_integrand(double k, void *params);

#endif
