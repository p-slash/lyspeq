#ifndef COSMOLOGY_3D_H
#define COSMOLOGY_3D_H

#include <cmath>
#include <memory>
#include <string>

#include "core/global_numbers.hpp"
#include "mathtools/discrete_interpolation.hpp"
#include "io/config_file.hpp"


double trapz(double *y, double dx, int N) {
    double result = (y[0] + y[N - 1]) / 2;
    for (int i = 1; i < N - 1; ++i)
        result += y[i];
    return result * dx;
}


namespace fidcosmo {
    const config_map planck18_default_parameters ({
        {"OmegaMatter", "0.30966"}, {"OmegaRadiation", "5.402015137139352e-05"},
        {"Hubble", "67.66"}
    });

    class FlatLCDM {
        double Omega_m, Omega_r, H0, Omega_L;
        std::unique_ptr<DiscreteInterpolation1D>
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
            Omega_L = 1 - Omega_m - Omega_r * (1 + _nu_relative_density(1));

            // Cache
            const int nz = 2000, nz2 = 2900;
            const double dz = 0.002;
            double z1arr[nz], Hz[nz], cDist[nz];
            for (int i = 0; i < nz; ++i) {
                z1arr[i] = 2.8 + dz * i;
                Hz[i] = _calcHubble(z1arr[i]);
            }

            hubble_z = std::make_unique<DiscreteInterpolation1D>(
                z1arr[0], dz, nz, &Hz[0]);

            double invHz[nz2];
            for (int i = 0; i < nz2; ++i)
                invHz[i] = getInvHubble(1 + i * dz);

            for (int i = 0; i < nz; ++i) {
                int N = (z1arr[i] - 1) / dz + 1.01;
                cDist[i] = SPEED_OF_LIGHT * trapz(&invHz[0], dz, N);
            }

            interp_comov_dist = std::make_unique<DiscreteInterpolation1D>(
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
            double z4 = z3 * z1;
            double nu = _nu_relative_density(z1);
            return H0 * sqrt(Omega_L + Omega_m * z3 + Omega_r * (1 + nu) * z4);
        }

        /* in km/s/Mpc */
        double getHubble(double z1) {
            return hubble_z->evaluate(z1);
        }

        double getInvHubble(double z1) {
            return 1 / hubble_z->evaluate(z1);
        }

        double getComovingDist(double z1) {
            return interp_comov_dist->evaluate(z1);
        }
    };

    class LinearPowerInterpolator {
        int N;
        double z_pivot;
        std::unique_ptr<double[]> lnk, lnP;
    public:
        /* This function reads following keys from config file:
        PlinearFilename: str
        PlinearPivotRedshift: double
        */
        LinearPowerInterpolator(ConfigFile &config) {
            std::string fname = config.get("PlinearFilename");
            z_pivot = config.getDouble("PlinearPivotRedshift");
        }
    }
}

#endif
