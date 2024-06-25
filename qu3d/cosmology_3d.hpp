#ifndef COSMOLOGY_3D_H
#define COSMOLOGY_3D_H

#include <cmath>
#include <memory>
#include <string>

#include "core/global_numbers.hpp"
#include "mathtools/discrete_interpolation.hpp"
#include "io/config_file.hpp"
#include "io/io_helper_functions.hpp"


inline double trapz(const double *y, int N, double dx=1.0) {
    double result = y[N - 1] / 2;
    for (int i = N - 2; i > 0; --i)
        result += y[i];
    result += y[0] / 2;
    return result * dx;
}


namespace fidcosmo {
    const config_map planck18_default_parameters ({
        {"OmegaMatter", "0.30966"}, {"OmegaRadiation", "5.402015137139352e-05"},
        {"Hubble", "67.66"}
    });

    class FlatLCDM {
        double Omega_m, Omega_r, H0, Omega_L;
        std::unique_ptr<DiscreteCubicInterpolation1D>
            interp_comov_dist, hubble_z;
    public:
        /* This function reads following keys from config file:
        OmegaMatter: double
        OmegaRadiation: double
        Hubble: double
        */
        FlatLCDM(ConfigFile &config) {
            config.addDefaults(planck18_default_parameters);
            Omega_m = config.getDouble("OmegaMatter");
            Omega_r = config.getDouble("OmegaRadiation");
            H0 = config.getDouble("Hubble");
            Omega_L = 1.0 - Omega_m - Omega_r * (1 + _nu_relative_density(1));

            // Cache
            const int nz = 300;
            const double dz = 0.02;
            double z1arr[nz], Hz[nz], cDist[nz];
            for (int i = 0; i < nz; ++i) {
                z1arr[i] = 1.0 + dz * i;
                Hz[i] = _calcHubble(z1arr[i]);
            }

            hubble_z = std::make_unique<DiscreteCubicInterpolation1D>(
                z1arr[0], dz, nz, &Hz[0]);

            const int nz2 = 3100;
            const double dz2 = 0.002;
            double invHz[nz2];
            for (int i = 0; i < nz2; ++i)
                invHz[i] = getInvHubble(1 + i * dz2);

            for (int i = 0; i < nz; ++i) {
                int N = (z1arr[i] - 1) / dz2 + 1.01;
                cDist[i] = SPEED_OF_LIGHT * trapz(&invHz[0], N, dz2);
            }

            interp_comov_dist = std::make_unique<DiscreteCubicInterpolation1D>(
                z1arr[0], dz, nz, &cDist[0]);
        }
    
        /* Fitted to astropy Planck18.nu_relative_density function based on
           Komatsu et al. 2011, eq 26
        */
        double _nu_relative_density(
                double z1, double A=0.3173, double nu_y0=357.91212097,
                double p=1.83, double invp=0.54644808743,
                double nmassless=2, double B=0.23058962986246165
        ) {
            return B * (nmassless + pow(1 + pow(A * nu_y0 / z1, p), invp));
        }

        /* in km/s/Mpc */
        double _calcHubble(double z1) {
            double z3 = z1 * z1 * z1;
            double nu = 1.0 + _nu_relative_density(z1);
            return H0 * sqrt(Omega_L + (Omega_m + Omega_r * nu * z1) * z3);
        }

        /* in km/s/Mpc */
        double getHubble(double z1) const {
            return hubble_z->evaluate(z1);
        }

        double getInvHubble(double z1) const {
            return 1 / hubble_z->evaluate(z1);
        }

        double getComovingDist(double z1) const {
            return interp_comov_dist->evaluate(z1);
        }
    };

    class LinearPowerInterpolator {
        /* Power spectrum interpolator in Mpc units */
        std::unique_ptr<DiscreteCubicInterpolation1D> interp_lnp;

        std::unique_ptr<double[]> _appendLinearExtrapolation(
                double lnk1, double lnk2, double dlnk, int N,
                const std::vector<double> &lnP, double &newlnk1,
                double lnkmin=log(1e-5), double lnkmax=log(5.0)
        ) {
            double m1 = lnP[1] - lnP[0],
                   m2 = lnP[N - 1] - lnP[N - 2];

            int n1 = std::max(0, int((lnk1 - lnkmin) / dlnk)),
                n2 = std::max(0, int((lnkmax - lnk2) / dlnk));

            newlnk1 = lnk1 - n1 * dlnk;
            auto result = std::make_unique<double[]>(n1 + N + n2);

            for (int i = 0; i < n1; ++i)
                result[i] = lnP[0] + m1 * (i - n1);

            std::copy(lnP.begin(), lnP.end(), result.get() + n1);

            for (int i = 0; i < n2; ++i)
                result[n1 + N + i] = lnP.back() + m2 * (i + 1);

            return result;
        }

        void _readFile(const std::string &fname) {
            std::ifstream toRead = ioh::open_fstream<std::ifstream>(fname);
            std::vector<double> lnk, lnP;
            double lnk1, lnP1;

            while (toRead >> lnk1 >> lnP1) {
                lnk.push_back(lnk1);
                lnP.push_back(lnP1);
            }

            toRead.close();

            int N = lnk.size();
            double dlnk = lnk[1] - lnk[0];

            for (size_t i = 1; i < N - 1; ++i)
                if (fabs(lnk[i + 1] - lnk[i] - dlnk) > 1e-8)
                    throw std::runtime_error(
                        "Input PlinearFilename does not have equal ln k spacing.");

            auto appendLnp = _appendLinearExtrapolation(
                lnk[0], lnk.back(), dlnk, N, lnP, lnk1);
            interp_lnp = std::make_unique<DiscreteCubicInterpolation1D>(
                lnk1, dlnk, N, appendLnp.get());
        }
    public:
        double z_pivot;

        /* This function reads following keys from config file:
        PlinearFilename: str
            Linear power spectrum file. First column is ln k, second column is
            ln P. ln k must be equally spaced.
        PlinearPivotRedshift: double
        */
        LinearPowerInterpolator(ConfigFile &config) {
            std::string fname = config.get("PlinearFilename");
            if (fname.empty())
                throw std::invalid_argument("Must pass PlinearFilename.");

            z_pivot = config.getDouble("PlinearPivotRedshift");
            _readFile(fname);
        }

        double evaluate(double k) {
            if (k == 0)
                return 0;

            return exp(interp_lnp->evaluate(log(k)));
        }
    };

    const config_map arinyo_default_parameters ({
        // Some parameters from vega and DESI Y1 fits.
        // Need to confirm if k_p and k_nu are in Mpc units or in Mpc/h units.
        {"b_F", "0.11"}, {"beta_F", "1.74"}, {"k_p", "19.47"},
        {"q_1", "0.8558"}, {"nu_0", "1.07"}, {"nu_1", "1.61"},
        {"k_nu", "1.11"}
    });

    class ArinyoP3DModel {
        double _varlss;
        double b_F, beta_F, k_p, q_1, nu_0, nu_1, k_nu, rscale_long;
        std::unique_ptr<LinearPowerInterpolator> interp_p;

        void _calcVarLss() {

        }
    public:
        /* This function reads following keys from config file:
        b_F: double
        beta_F: double
        k_p: double
        q_1: double
        nu_0: double  == b_nu - a_nu of arinyo
        nu_1: double  == b_nu of arinyo
        k_nu: double
        */
        ArinyoP3DModel(ConfigFile &config) : _varlss(0) {
            config.addDefaults(arinyo_default_parameters);
            b_F = config.getDouble("b_F");
            beta_F = config.getDouble("beta_F");
            k_p = config.getDouble("k_p");
            q_1 = config.getDouble("q_1");
            nu_0 = config.getDouble("nu_0");
            nu_1 = config.getDouble("nu_1");
            k_nu = config.getDouble("k_nu");
            rscale_long = config.getDouble("LongScale");

            interp_p = std::make_unique<LinearPowerInterpolator>(config);
            _calcVarLss();
        }

        double evaluate(double k, double kz) {
            if (k == 0)
                return 0;

            double
            plin = interp_p->evaluate(k),
            delta2_L = plin * k * k * k / 2 / MY_PI * MY_PI,
            k_kp = k / k_p, k_rL = k * rscale_long,
            mu = kz / k,
            result, lnD;

            result = b_F * (1 + beta_F * mu * mu);
            result *= result;

            lnD = (q_1 * delta2_L) * (
                    1 - pow(kz / k_nu, nu_1) * pow(k / k_nu, -nu_0)
            ) - k_kp * k_kp - k_rL * k_rL;

            return result * plin * exp(lnD);
        }

        double getVarLss() {

        }
    };
}

#endif
