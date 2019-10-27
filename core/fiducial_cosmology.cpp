#include "core/fiducial_cosmology.hpp"

#include <cmath>
#include <iostream>
#include <vector>
#include <set>
#include <stdexcept>

#include "gsltools/interpolation_2d.hpp"
#include "io/io_helper_functions.hpp"

// Conversion functions
namespace conv
{
    bool USE_LOG_V = false, FLUX_TO_DELTAF_BY_CHUNKS = false; // , USE_FID_LEE12_MEAN_FLUX = false;
    
    void convertLambdaToVelocity(double &median_z, double *v_array, const double *lambda, int size)
    {
        double median_lambda = lambda[size / 2];
    
        median_z = median_lambda / LYA_REST - 1.;

        if (USE_LOG_V)
        {
            for (int i = 0; i < size; ++i)
                v_array[i] = SPEED_OF_LIGHT * log(lambda[i]/median_lambda);
        }
        else
        {
            for (int i = 0; i < size; ++i)
                v_array[i] = 2. * SPEED_OF_LIGHT * (1. - sqrt(median_lambda / lambda[i]));
        }
    }

    void convertLambdaToRedshift(double *lambda, int size)
    {
        for (int i = 0; i < size; ++i)
        {
            lambda[i] /= LYA_REST;
            lambda[i] -= 1.;
        }
    }

    // double meanFluxBecker13(double z)
    // {
    //     double tau = 0.751 * pow((1. + z) / 4.5, 2.90) - 0.132;
    //     return exp(-tau);
    // }

    // double meanFluxLee12(double z)
    // {
    //     double tau = 0.001845 * pow(1. + z, 3.924);
    //     return exp(-tau);
    // }

    void convertFluxToDeltaf(double *flux, double *noise, int size)
    {
        double mean_f = 0.;

        for (int i = 0; i < size; ++i)
            mean_f += flux[i];
        mean_f /= size;

        for (int i = 0; i < size; ++i)
        {
            flux[i]  /= mean_f;
            flux[i]  -= 1.;
            noise[i] /= mean_f;
        }
    }

    // void convertFluxToDeltafLee12(const double *lambda, double *flux, double *noise, int size)
    // {
    //     double mean_f;

    //     for (int i = 0; i < size; ++i)
    //     {
    //         mean_f    = meanFluxLee12(lambda[i]/LYA_REST-1);
    //         flux[i]  /= mean_f;
    //         flux[i]  -= 1.;
    //         noise[i] /= mean_f;
    //     }
    // }
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
    
    bool USE_INTERP_FIDUCIAL_POWER = false;
    Interpolation2D *interp2d_fiducial_power = NULL;

    inline double interpolationFiducialPower(double k, double z, void *params)
    {
        void  *pp __attribute__((unused)) = params;

        return interp2d_fiducial_power->evaluate(z, k);
    }

    // Assume file starts with two integers, then has three columns
    // Nk Nz
    // z k P
    // . . .
    // Power is ordered for each redshift bin
    void setFiducialPowerFromFile(const char * fname)
    {
        int n_k_points, n_z_points, size;
        std::vector<double> fiducial_power_vector_from_file, k_values, z_values;

        std::fstream to_read_fidpow = ioh::open_file(fname);
        // Assume file starts with two integers 
        // Nk Nz
        to_read_fidpow >> n_k_points >> n_z_points;
        size = n_k_points * n_z_points;

        fiducial_power_vector_from_file.reserve(size);
        k_values.reserve(n_k_points);
        z_values.reserve(n_z_points);

        // Assume file has three columns after
        // z k P
        // Power is ordered for each redshift bin
        for (int nline = 0; nline < size; ++nline)
        {
            double tmp_z, tmp_k, tmp_p;
            to_read_fidpow >> tmp_z >> tmp_k >> tmp_p;
            
            // First Nk values are k values
            if (nline < n_k_points)         k_values.push_back(tmp_k);
            // Every other Nk value form z values
            if (nline % n_k_points == 0)    z_values.push_back(tmp_z);

            fiducial_power_vector_from_file.push_back(tmp_p);
        }
        
        interp2d_fiducial_power = new Interpolation2D(GSL_BICUBIC_INTERPOLATION, z_values.data(), k_values.data(),
            fiducial_power_vector_from_file.data(), n_z_points, n_k_points);

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












