#ifndef FIDUCIAL_COSMOLOGY_H
#define FIDUCIAL_COSMOLOGY_H

double fiducial_power_spectrum(double k, double z, void *params);

double q_matrix_integrand(double k, void *params);

double signal_matrix_integrand(double k, void *params);

double debuggin_power_spectrum(double k, double dv);

double lnpoly2_power_spectrum(double lnk);

struct palanque_fit_params
{
    double A;
    double n;
    double alpha;

    double B;
    double beta;
};

double Palanque_Delabrouille_etal_2013_fit(double k, double z, struct palanque_fit_params &params);

#endif
