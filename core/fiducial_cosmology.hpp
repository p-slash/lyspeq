#ifndef FIDUCIAL_COSMOLOGY_H
#define FIDUCIAL_COSMOLOGY_H

#define SPEED_OF_LIGHT 299792.458
#define LYA_REST 1215.67
#define PI 3.14159265359
#define ONE_SIGMA_2_FWHM 2.35482004503

namespace conv
{
    extern bool FLUX_TO_DELTAF_BY_CHUNKS, INPUT_IS_DELTA_FLUX;

    void setMeanFlux(const char *fname=0);

    // void convertLambdaToVelocity(double &median_z, double *v_array, const double *lambda, int size);
    void convertLambdaToRedshift(double *lambda, int size);

    extern void (*convertFluxToDeltaF)(const double *lambda, double *flux, double *noise, int size);
    // void convertFluxToDeltafLee12(const double *lambda, double *flux, double *noise, int size);
}

namespace fidcosmo
{
    // Default fiducial power is Palanque Delabrouille fit.
    // Calling setFiducialPowerFromFile changes this pointer to interpolationFiducialPower
    extern double (*fiducialPowerSpectrum)(double k, double z, void *params);
    // Currently growth factor is always given Palanqu Delabrouille fit.
    extern double (*fiducialPowerGrowthFactor)(double z_ij, double k_kn, double z_zm, void *params);
    extern double FID_LOWEST_K, FID_HIGHEST_K;

    // Assume binary file starts with 
    // two integers, 
    // then redshift values as doubles, 
    // k values as doubles,
    // finally power values as doubles.
    // Power is ordered for each redshift bin
    void setFiducialPowerFromFile(const char *fname);

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
    
        // Default parameters are fit parameters if not specified in config file.
        extern pd13_fit_params FIDUCIAL_PD13_PARAMS;
    }
}

namespace fidpd13 = fidcosmo::pd13;

// Data structures to integrate signal and derivative matrix expressions
struct spectrograph_windowfn_params
{
    double delta_v_ij;
    double z_ij;
    double pixel_width;
    double spectrograph_res;    // This is 1 sigma km/s resolution
};

struct sq_integrand_params
{
    fidpd13::pd13_fit_params            *fiducial_pd_params;
    struct spectrograph_windowfn_params *spec_window_params;
};

double q_matrix_integrand(double k, void *params);
double signal_matrix_integrand(double k, void *params);

#endif
