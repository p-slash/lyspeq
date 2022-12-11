#include "core/fiducial_cosmology.hpp"
#include "core/global_numbers.hpp"
#include "mathtools/interpolation.hpp"
#include "mathtools/interpolation_2d.hpp"
#include "io/io_helper_functions.hpp"
#include "io/logger.hpp"

#include <cmath>
#include <numeric> // std::accumulate
#include <algorithm> // std::for_each & transform
#include <iostream>
#include <vector>
#include <set>
#include <stdexcept>
#include <memory>

// Conversion functions
namespace conv
{
    bool FLUX_TO_DELTAF_BY_CHUNKS = false, INPUT_IS_DELTA_FLUX = false;
    std::unique_ptr<Interpolation> interp_mean_flux;

    /* This function reads following keys from config file:
    UseChunksMeanFlux: int
        Forces the mean flux of each chunk to be zero when > 0. Off by default.
    InputIsDeltaFlux: int
        Assumes input is delta when > 0. True by default.
    MeanFluxFile: string
        Reads the mean flux from a file and get deltas using this mean flux.
        Off by default.
    */
    void readConversion(ConfigFile &config)
    {
        LOG::LOGGER.STD("###############################################\n");
        LOG::LOGGER.STD("Reading conversion parameters from config.\n");

        config.addDefaults(conversion_default_parameters);

        // If 1, uses mean of each chunk as F-bar
        int uchunkmean = config.getInteger("UseChunksMeanFlux");
        // If 1, input is delta_f
        int udeltaf = config.getInteger("InputIsDeltaFlux");

        FLUX_TO_DELTAF_BY_CHUNKS  = uchunkmean > 0;
        INPUT_IS_DELTA_FLUX       = udeltaf > 0;

        // File to interpolate for F-bar
        std::string FNAME_MEAN_FLUX = config.get("MeanFluxFile");

        // resolve conflict: Input delta flux overrides all
        // Then, chunk means.
        if (INPUT_IS_DELTA_FLUX && FLUX_TO_DELTAF_BY_CHUNKS)
        {
            LOG::LOGGER.ERR("Both input delta flux and conversion"
                " using chunk's mean flux is turned on. "
                "Assuming input is flux fluctuations delta_f.\n");
            FLUX_TO_DELTAF_BY_CHUNKS = false;
        }

        setMeanFlux();

        if (!FNAME_MEAN_FLUX.empty())
        {
            if (FLUX_TO_DELTAF_BY_CHUNKS)
            {
                LOG::LOGGER.ERR("Both mean flux file and using"
                    " chunk's mean flux is turned on. "
                    "Using chunk's mean flux.\n");
            }
            else if (INPUT_IS_DELTA_FLUX)
            {
                LOG::LOGGER.ERR("Both input delta flux and conversion using "
                    "mean flux file is turned on. "
                    "Assuming input is flux fluctuations delta_f.\n");
            }
            else
                setMeanFlux(FNAME_MEAN_FLUX);
        }
        else if (!(INPUT_IS_DELTA_FLUX || FLUX_TO_DELTAF_BY_CHUNKS))
            INPUT_IS_DELTA_FLUX = true;

        #define booltostr(x) x ? "true" : "false"
        LOG::LOGGER.STD("UseChunksMeanFlux is set to %s.\n",
            booltostr(FLUX_TO_DELTAF_BY_CHUNKS));
        LOG::LOGGER.STD("InputIsDeltaFlux is set to %s.\n",
            booltostr(INPUT_IS_DELTA_FLUX));
        LOG::LOGGER.STD("MeanFluxFile is set to %s.\n\n",
            FNAME_MEAN_FLUX.c_str());
        #undef booltostr
    }

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

        std::for_each(flux, flux+size, [chunk_mean](double &f) { f = f/chunk_mean-1; });
        std::for_each(noise, noise+size, [chunk_mean](double &n) { n /= chunk_mean; });
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
            [median_lambda](const double &l) { 
                return SPEED_OF_LIGHT * log(l/median_lambda); 
            });
    }

    void convertLambdaToRedshift(double *lambda, int size)
    {
        std::for_each(lambda, lambda+size, [](double &ld) { ld = ld/LYA_REST-1; });
    }

    void (*convertFluxToDeltaF)(const double*, double*, double*, int) = &noConversion;

    void setMeanFlux(const std::string &fname)
    {
        if (fname.empty())
        {
            if (FLUX_TO_DELTAF_BY_CHUNKS)
                convertFluxToDeltaF = &chunkMeanConversion;

            return;
        }

        int size;

        std::ifstream to_read_meanflux = ioh::open_fstream<std::ifstream>(fname, 'b');
        // Assume file starts with two integers 
        // Nk Nz
        to_read_meanflux.read((char *)&size, sizeof(int));

        double *z_values = new double[size],
               *f_values = new double[size];

        // Redshift array as doubles
        to_read_meanflux.read((char *)z_values, size*sizeof(double));
        
        // Remaining is flux array
        to_read_meanflux.read((char *)f_values, size*sizeof(double));
        to_read_meanflux.close();

        interp_mean_flux = std::make_unique<Interpolation>(GSL_CUBIC_INTERPOLATION,
            z_values, f_values, size);

        convertFluxToDeltaF = &fullConversion;

        delete [] z_values;
        delete [] f_values;
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
    std::unique_ptr<Interpolation2D> interp2d_fiducial_power;

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
    void readFiducialCosmo(ConfigFile &config)
    {
        // Baseline Power Spectrum
        std::string FNAME_FID_POWER = config.get("FiducialPowerFile");
        // Fiducial Palanque fit function parameters
        pd13::FIDUCIAL_PD13_PARAMS.A      = config.getDouble("FiducialAmplitude");
        pd13::FIDUCIAL_PD13_PARAMS.n      = config.getDouble("FiducialSlope");
        pd13::FIDUCIAL_PD13_PARAMS.alpha  = config.getDouble("FiducialCurvature");
        pd13::FIDUCIAL_PD13_PARAMS.B      = config.getDouble("FiducialRedshiftPower");
        pd13::FIDUCIAL_PD13_PARAMS.beta   = config.getDouble("FiducialRedshiftCurvature");
        pd13::FIDUCIAL_PD13_PARAMS.lambda = config.getDouble("FiducialLorentzianLambda");

        if (!FNAME_FID_POWER.empty())
            setFiducialPowerFromFile(FNAME_FID_POWER);
        else if (pd13::FIDUCIAL_PD13_PARAMS.A <= 0)
            throw std::invalid_argument("FiducialAmplitude must be > 0.");
    }

    inline double interpolationFiducialPower(double k, double z, void *params)
    {
        void  *pp __attribute__((unused)) = params;

        return interp2d_fiducial_power->evaluate(k, z);
    }

    // Assume binary file starts with two integers, then redshift values as double, k values as double,
    // finally power values.
    // Power is ordered for each redshift bin
    void setFiducialPowerFromFile(const std::string &fname)
    {
        int n_k_points, n_z_points, size;

        std::ifstream to_read_fidpow = ioh::open_fstream<std::ifstream>(fname, 'b');
        // Assume file starts with two integers 
        // Nk Nz
        to_read_fidpow.read((char *)&n_k_points, sizeof(int));
        to_read_fidpow.read((char *)&n_z_points, sizeof(int));
        size = n_k_points * n_z_points;

        double *k_values = new double[n_k_points],
               *z_values = new double[n_z_points],
            *fiducial_power_from_file = new double[size];

        // Redshift array, then k array as doubles
        to_read_fidpow.read((char *)z_values, n_z_points*sizeof(double));
        to_read_fidpow.read((char *)k_values, n_k_points*sizeof(double));

        FID_LOWEST_K  = k_values[0];
        FID_HIGHEST_K = k_values[n_k_points - 1];
        
        // Remaining is power array
        to_read_fidpow.read((char *)fiducial_power_from_file, size*sizeof(double));
        to_read_fidpow.close();
        
        interp2d_fiducial_power = std::make_unique<Interpolation2D>(GSL_BICUBIC_INTERPOLATION,
            k_values, z_values, fiducial_power_from_file,
            n_k_points, n_z_points);

        fiducialPowerSpectrum = &interpolationFiducialPower;

        delete [] k_values;
        delete [] z_values;
        delete [] fiducial_power_from_file;
    }

    // Palanque_Delabrouille et al. 2013 based functions. Extra Lorentzian decay
    namespace pd13
    {
        // Defined values for PD fit
        // K_0 in s km^-1
        const double K_0 = 0.009, Z_0 = 3.0;

        pd13_fit_params FIDUCIAL_PD13_PARAMS = {6.621420e-02, -2.685349e+00, -2.232763e-01, 3.591244e+00, -1.768045e-01, 3.598261e+02};

        double Palanque_Delabrouille_etal_2013_fit(double k, double z, void *params)
        {
            pd13_fit_params *pfp = (pd13::pd13_fit_params *) params;

            double  q0 = k / K_0 + 1E-10, x0 = (1. + z) / (1. + Z_0);

            return    (pfp->A * MY_PI / K_0)
                    * pow(q0, 2. + pfp->n + pfp->alpha * log(q0) + pfp->beta  * log(x0)) 
                    * pow(x0,   pfp->B) / (1. + pfp->lambda * k * k);
        }

        double Palanque_Delabrouille_etal_2013_fit_growth_factor(double z_ij, double k_kn, double z_zm, void *params)
        {
            pd13_fit_params *pfp = (pd13::pd13_fit_params *) params;

            double  q0 = k_kn / K_0 + 1E-10, xm = (1. + z_ij) / (1. + z_zm);

            return  pow(xm, pfp->B + pfp->beta  * log(q0));
        }
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

    return result * result * fidcosmo::fiducialPowerSpectrum(k, sqip->spec_window_params->z_ij, sqip->fiducial_pd_params) / MY_PI;
}

double q_matrix_integrand(double k, void *params)
{
    struct sq_integrand_params *sqip = (struct sq_integrand_params*) params;

    double result = spectral_response_window_fn(k, sqip->spec_window_params);
    return result * result / MY_PI;
}












