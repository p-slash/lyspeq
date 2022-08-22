#ifndef FIDUCIAL_COSMOLOGY_H
#define FIDUCIAL_COSMOLOGY_H

#include <string>
#include "io/config_file.hpp"

namespace conv
{
    extern bool FLUX_TO_DELTAF_BY_CHUNKS, INPUT_IS_DELTA_FLUX;

    void setMeanFlux(const std::string &fname="");

    // void convertLambdaToVelocity(double &median_z, double *v_array, const double *lambda, int size);
    void convertLambdaToRedshift(double *lambda, int size);

    extern void (*convertFluxToDeltaF)(const double *lambda, double *flux, double *noise, int size);
    // void convertFluxToDeltafLee12(const double *lambda, double *flux, double *noise, int size);

    const config_map conversion_default_parameters ({
        {"UseChunksMeanFlux", "-1"}, {"InputIsDeltaFlux", "1"}, 
        {"MeanFluxFile", ""}});
    /* This function reads following keys from config file:
    UseChunksMeanFlux: int
        Forces the mean flux of each chunk to be zero when > 0. Off by default.
    InputIsDeltaFlux: int
        Assumes input is delta when > 0. True by default.
    MeanFluxFile: string
        Reads the mean flux from a file and get deltas using this mean flux.
        Off by default.
    */
    void readConversion(ConfigFile &config);
    void clearCache();
}

namespace fidcosmo
{
    // Default fiducial power is Palanque Delabrouille fit.
    // Calling setFiducialPowerFromFile changes this pointer to interpolationFiducialPower
    extern double (*fiducialPowerSpectrum)(double k, double z, void *params);
    // Currently growth factor is always given Palanqu Delabrouille fit.
    extern double (*fiducialPowerGrowthFactor)(double z_ij, double k_kn, double z_zm, void *params);
    extern double FID_LOWEST_K, FID_HIGHEST_K;

    /* This function reads following keys from config file:
    FiducialPowerFile: string
        File for the fiducial power spectrum. Off by default.
    FiducialAmplitude: double
    FiducialSlope: double
    FiducialCurvature: double
    FiducialRedshiftPower: double
    FiducialRedshiftCurvature: double
    FiducialLorentzianLambda: double
    */
    void readFiducialCosmo(const ConfigFile &config);
    void clearCache();

    // Assume binary file starts with 
    // two integers, 
    // then redshift values as doubles, 
    // k values as doubles,
    // finally power values as doubles.
    // Power is ordered for each redshift bin
    void setFiducialPowerFromFile(const std::string &fname);

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
