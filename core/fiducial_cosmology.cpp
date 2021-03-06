#include "core/fiducial_cosmology.hpp"

#include <cmath>
#include <numeric> // std::accumulate
#include <algorithm> // std::for_each & transform
#include <iostream>
#include <vector>
#include <set>
#include <stdexcept>

#include "gsltools/interpolation.hpp"
#include "gsltools/interpolation_2d.hpp"
#include "io/io_helper_functions.hpp"

// Conversion functions
namespace conv
{
    bool USE_LOG_V = false, FLUX_TO_DELTAF_BY_CHUNKS = false, INPUT_IS_DELTA_FLUX = false;
    Interpolation *interp_mean_flux = NULL;

    void noConversion(const double *lambda, double *flux, double *noise, int size)
    {
        const double *l __attribute__((unused)) = lambda;
        double *f __attribute__((unused)) = flux;
        double *n __attribute__((unused)) = noise;
        int s __attribute__((unused)) = size;
    }

    void chunkMeanConversion(const double *lambda, double *flux, double *noise, int size)
    {
        const double *l __attribute__((unused)) = lambda;
        double chunk_mean = std::accumulate(flux, flux+size, 0.) / size;

        std::for_each(flux, flux+size, [&](double &f) { f = f/chunk_mean-1; });
        std::for_each(noise, noise+size, [&](double &n) { n /= chunk_mean; });
    }

    void fullConversion(const double *lambda, double *flux, double *noise, int size)
    {
        for (int i = 0; i < size; ++i)
        {
            double tmp_meanf = interp_mean_flux->evaluate(lambda[i]/LYA_REST-1);
            *(flux+i)   = *(flux+i)/tmp_meanf - 1;
            *(noise+i) /= tmp_meanf;
        }
    }

    void convertLambdaToVelocity(double &median_z, double *v_array, const double *lambda, int size)
    {
        double median_lambda = lambda[size / 2];
    
        median_z = median_lambda / LYA_REST - 1.;

        std::transform(lambda, lambda+size, v_array, 
            [&](const double &l) { return SPEED_OF_LIGHT * log(l/median_lambda); });
    }

    void convertLambdaToRedshift(double *lambda, int size)
    {
        std::for_each(lambda, lambda+size, [](double &ld) { ld = ld/LYA_REST-1; });
    }

    void (*convertFluxToDeltaF)(const double*, double*, double*, int) = &noConversion;

    void setMeanFlux(const char *fname)
    {
        if (fname == 0)
        {
            if (FLUX_TO_DELTAF_BY_CHUNKS)
                convertFluxToDeltaF = &chunkMeanConversion;

            return;
        }

        double *z_values, *f_values;
        int size;

        std::ifstream to_read_meanflux = ioh::open_fstream<std::ifstream>(fname, 'b');
        // Assume file starts with two integers 
        // Nk Nz
        to_read_meanflux.read((char *)&size, sizeof(int));

        z_values = new double[size];
        f_values = new double[size];

        // Redshift array as doubles
        to_read_meanflux.read((char *)z_values, size*sizeof(double));
        
        // Remaining is flux array
        to_read_meanflux.read((char *)f_values, size*sizeof(double));
        to_read_meanflux.close();

        interp_mean_flux = new Interpolation(GSL_CUBIC_INTERPOLATION, z_values, f_values, size);
        delete [] z_values;
        delete [] f_values;

        convertFluxToDeltaF = &fullConversion;
    }
}

// Fiducial Functions
namespace fidcosmo
{
    namespace pd13
    {
        double Palanque_Delabrouille_etal_2013_fit(double, double, void*);
        double Palanque_Delabrouille_etal_2013_fit_growth_factor(double, double, double, void*);
    }

    double (*fiducialPowerSpectrum)(double, double, void*)             = &pd13::Palanque_Delabrouille_etal_2013_fit;
    double (*fiducialPowerGrowthFactor)(double, double, double, void*) = &pd13::Palanque_Delabrouille_etal_2013_fit_growth_factor;
    
    double FID_LOWEST_K = 0, FID_HIGHEST_K = 10.;
    Interpolation2D *interp2d_fiducial_power = NULL;

    inline double interpolationFiducialPower(double k, double z, void *params)
    {
        void  *pp __attribute__((unused)) = params;

        return interp2d_fiducial_power->evaluate(k, z);
    }

    // Assume binary file starts with two integers, then redshift values as double, k values as double,
    // finally power values.
    // Power is ordered for each redshift bin
    void setFiducialPowerFromFile(const char * fname)
    {
        int n_k_points, n_z_points, size;
        double *fiducial_power_from_file, *k_values, *z_values;

        std::ifstream to_read_fidpow = ioh::open_fstream<std::ifstream>(fname, 'b');
        // Assume file starts with two integers 
        // Nk Nz
        to_read_fidpow.read((char *)&n_k_points, sizeof(int));
        to_read_fidpow.read((char *)&n_z_points, sizeof(int));
        size = n_k_points * n_z_points;

        k_values = new double[n_k_points];
        z_values = new double[n_z_points];
        fiducial_power_from_file = new double[size];

        // Redshift array, then k array as doubles
        to_read_fidpow.read((char *)z_values, n_z_points*sizeof(double));
        to_read_fidpow.read((char *)k_values, n_k_points*sizeof(double));

        FID_LOWEST_K  = k_values[0];
        FID_HIGHEST_K = k_values[n_k_points - 1];
        
        // Remaining is power array
        to_read_fidpow.read((char *)fiducial_power_from_file, size*sizeof(double));
        to_read_fidpow.close();
        
        interp2d_fiducial_power = new Interpolation2D(GSL_BICUBIC_INTERPOLATION, k_values, z_values, 
            fiducial_power_from_file, n_k_points, n_z_points);

        delete [] k_values;
        delete [] z_values;
        delete [] fiducial_power_from_file;

        fiducialPowerSpectrum = &interpolationFiducialPower;
    }

    // Palanque_Delabrouille et al. 2013 based functions. Extra Lorentzian decay
    namespace pd13
    {
    // Defined values for PD fit
    #define K_0 0.009 // s km^-1
    #define Z_0 3.0

        pd13_fit_params FIDUCIAL_PD13_PARAMS = {6.621420e-02, -2.685349e+00, -2.232763e-01, 3.591244e+00, -1.768045e-01, 3.598261e+02};

        double Palanque_Delabrouille_etal_2013_fit(double k, double z, void *params)
        {
            pd13_fit_params *pfp = (pd13::pd13_fit_params *) params;

            double  q0 = k / K_0 + 1E-10, x0 = (1. + z) / (1. + Z_0);

            return    (pfp->A * PI / K_0)
                    * pow(q0, 2. + pfp->n + pfp->alpha * log(q0) + pfp->beta  * log(x0)) 
                    * pow(x0,   pfp->B) / (1. + pfp->lambda * k * k);
        }

    #undef Z_0 // Do not need Z_0 after this point

        double Palanque_Delabrouille_etal_2013_fit_growth_factor(double z_ij, double k_kn, double z_zm, void *params)
        {
            pd13_fit_params *pfp = (pd13::pd13_fit_params *) params;

            double  q0 = k_kn / K_0 + 1E-10, xm = (1. + z_ij) / (1. + z_zm);

            return  pow(xm, pfp->B + pfp->beta  * log(q0));
        }

    #undef K_0 // Do not need K_0 either
    }
}

// Signal and Derivative Integrands and Window Function
inline double sinc(double x)
{
    if (fabs(x) < 1E-6)     return 1.;

    return sin(x) / x;
}

inline double spectral_response_window_fn(double k, struct spectrograph_windowfn_params *spec_params)
{
    double R = spec_params->spectrograph_res, dv_kms = spec_params->pixel_width;

    return sinc(k * dv_kms / 2.) * exp(-k*k * R*R / 2.);
}

double signal_matrix_integrand(double k, void *params)
{
    struct sq_integrand_params *sqip = (struct sq_integrand_params*) params;

    double result = spectral_response_window_fn(k, sqip->spec_window_params);

    return result * result * fidcosmo::fiducialPowerSpectrum(k, sqip->spec_window_params->z_ij, sqip->fiducial_pd_params) / PI;
}

double q_matrix_integrand(double k, void *params)
{
    struct sq_integrand_params *sqip = (struct sq_integrand_params*) params;

    double result = spectral_response_window_fn(k, sqip->spec_window_params);
    return result * result / PI;
}












