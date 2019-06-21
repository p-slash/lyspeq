#ifndef FIDUCIAL_COSMOLOGY_H
#define FIDUCIAL_COSMOLOGY_H

#define SPEED_OF_LIGHT 299792.458
#define LYA_REST 1215.67
#define PI 3.14159265359

namespace pd13
{
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

    extern pd13_fit_params FIDUCIAL_PD13_PARAMS;
}

void convert_flux2deltaf_mean(double *flux, double *noise, int size);
// void convert_flux2deltaf_becker(const double *lambda, double *flux, double *noise, int size);
void convert_lambda2v(double &median_z, double *v_array, const double *lambda, int size);

// This function is defined in preprocessing
double fiducial_power_spectrum(double k, double z, void *params);
double fiducial_power_growth_factor(double z_ij, double k_kn, double z_zm, void *params);

struct spectrograph_windowfn_params
{
    double delta_v_ij;
    double z_ij;
    double pixel_width;
    double spectrograph_res;    // This is 1 sigma km/s resolution
};

double spectral_response_window_fn(double k, struct spectrograph_windowfn_params *spec_params);

// Data structures to integrate signal and derivative matrix expressions
struct sq_integrand_params
{
    pd13::pd13_fit_params               *fiducial_pd_params;
    struct spectrograph_windowfn_params *spec_window_params;
};

double q_matrix_integrand(double k, void *params);

double signal_matrix_integrand(double k, void *params);

#endif
