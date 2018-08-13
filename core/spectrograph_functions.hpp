#ifndef SPECTROGRAPH_FUNCTIONS_H
#define SPECTROGRAPH_FUNCTIONS_H

struct spectrograph_windowfn_params
{
    double delta_v_ij;
    double pixel_width;
    double spectrograph_res;
};

void convert_flux2deltaf(double *flux, int size);
void convert_lambda2v(double &median_z, double *v_array, const double *lambda, int size);

double spectral_response_window_fn(double k, void *params);

#endif
