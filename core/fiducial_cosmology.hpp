#ifndef FIDUCIAL_COSMOLOGY_H
#define FIDUCIAL_COSMOLOGY_H

#include <cmath>
#include <string>
#include "io/config_file.hpp"

namespace conv
{
    void setMeanFlux(const std::string &fname="");

    // void convertLambdaToVelocity(double &median_z, double *v_array, const double *lambda, int size);
    void convertLambdaToRedshift(double *lambda, int size);

    extern void (*convertFluxToDeltaF)(const double *lambda, double *flux, double *ivar, int size);

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
}

namespace fidcosmo
{
    // Default fiducial power is Palanque Delabrouille fit.
    // Calling setFiducialPowerFromFile changes this pointer to interpolationFiducialPower
    extern double (*fiducialPowerSpectrum)(double k, double z, void *params);
    // Currently growth factor is always given Palanqu Delabrouille fit.
    extern double (*fiducialPowerGrowthFactor)(double z_ij, double k_kn, double z_zm, void *params);
    extern double FID_LOWEST_K, FID_HIGHEST_K;
    namespace pd13
    {
        // Defined values for PD fit
        // K_0 in s km^-1
        const static double K_0 = 0.009, Z_0 = 3.0;

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

    inline double guidePower(double k) {
        // constexpr double Z0 = 3.0;
        // return fiducialPowerSpectrum(k, Z0, &pd13::FIDUCIAL_PD13_PARAMS);
        constexpr double kF = 5e-4;
        return kF / (k * k + kF * kF);
    }
    inline
    double spectralWindowFunc2(double k, double dv_kms, double R_kms) {
        if (k == 0)  return 1.;
        double kr = k * R_kms, r = exp(-kr * kr), x = k * dv_kms / 2.;
        if (dv_kms == 0)  return r;
        x = sin(x) / x;
        return x * x * r;
    }

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
    void readFiducialCosmo(ConfigFile &config);

    // Assume binary file starts with 
    // two integers, 
    // then redshift values as doubles, 
    // k values as doubles,
    // finally power values as doubles.
    // Power is ordered for each redshift bin
    void setFiducialPowerFromFile(const std::string &fname);
}

namespace fidpd13 = fidcosmo::pd13;

#endif
