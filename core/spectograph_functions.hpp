#ifndef SPECTOGRAPH_FUNCTIONS_H
#define SPECTOGRAPH_FUNCTIONS_H

extern double R_SPECTOGRAPH;

struct spectograph_windowfn_params
{
    double delta_v_ij;
    double pixel_width;
};

void convert_flux2deltaf(double *flux, int size);
void convert_lambda2v(double &mean_z, double *lambda, int size);

double spectral_response_window_fn(double k, void *params);

#endif
