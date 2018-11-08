/* TODO: mean flux(z) from a paper instead of averaging the observed flux
 */

#ifndef SPECTROGRAPH_FUNCTIONS_H
#define SPECTROGRAPH_FUNCTIONS_H

struct spectrograph_windowfn_params
{
    double delta_v_ij;
    double z_ij;
    double pixel_width;
    double spectrograph_res;
};

void convert_flux2deltaf(const double *lambda, double *flux, double *noise, int size);
void convert_lambda2v(double &median_z, double *v_array, const double *lambda, int size);

double spectral_response_window_fn(double k, struct spectrograph_windowfn_params *spec_params);

#endif
