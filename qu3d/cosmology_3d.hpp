#ifndef COSMOLOGY_3D_H
#define COSMOLOGY_3D_H

#include <cmath>
#include <memory>
#include <string>

#include "mathtools/discrete_interpolation.hpp"
#include "io/config_file.hpp"
#include "qu3d/qu3d_file.hpp"


namespace fidcosmo {
    const config_map planck18_default_parameters ({
        {"OmegaMatter", "0.30966"}, {"OmegaRadiation", "5.402015137139352e-05"},
        {"Hubble", "67.66"}
    });

    struct CosmoParams {
        double H0, Omega_m, Omega_L, Omega_r;
    };

    class FlatLCDM {
        struct CosmoParams cosmo_params;
        std::unique_ptr<DiscreteCubicInterpolation1D>
            interp_comov_dist, hubble_z, linear_growth_unnorm;

        void _integrateComovingDist(int nz, const double *z1arr, double *cDist);
        void _integrateLinearGrowth(int nz, const double *z1arr, double *linD);

    public:
        /* This function reads following keys from config file:
        OmegaMatter: double
        OmegaRadiation: double
        Hubble: double
        */
        FlatLCDM(ConfigFile &config);

        /* in km/s/Mpc */
        double getHubble(double z1) const { return hubble_z->evaluate(z1); }

        double getInvHubble(double z1) const {
            return 1 / hubble_z->evaluate(z1);
        }

        double getComovingDist(double z1) const {
            return interp_comov_dist->evaluate(z1);
        }

        double getUnnormLinearGrowth(double z1) const {
            return linear_growth_unnorm->evaluate(z1);
        }
    };

    class LinearPowerInterpolator {
        /* Power spectrum interpolator in Mpc units */
        std::unique_ptr<DiscreteCubicInterpolation1D> interp_lnp;

        std::unique_ptr<double[]> _appendLinearExtrapolation(
            double lnk1, double lnk2, double dlnk, int N,
            const std::vector<double> &lnP, double &newlnk1,
            double lnkmin=log(1e-5), double lnkmax=log(5.0)
        );

        void _readFile(const std::string &fname);
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
            if (k == 0)  return 0;
            return exp(interp_lnp->evaluate(log(k)));
        }
    };

    const config_map arinyo_default_parameters ({
        /* These defaults are obtained from Chabanier+24 Table A3.
           b_F fitted to all redshift ranges accounting for errors.
           Two redshift bins are averaged for others. */
        {"b_F", "0.1195977"}, {"alpha_F", "3.37681"},
        {"beta_F", "1.69"}, {"k_p", "17.625"},
        {"q_1", "0.7935"}, {"nu_0", "1.253"}, {"nu_1", "1.625"},
        {"k_nu", "0.3701"}
    });

    class ArinyoP3DModel {
        double _varlss, _D_pivot, _z1_pivot;
        double b_F, alpha_F, beta_F, k_p, q_1, nu_0, nu_1, k_nu, rscale_long;

        std::unique_ptr<LinearPowerInterpolator> interp_p;
        std::unique_ptr<DiscreteCubicInterpolation1D> interp_growth;

        std::unique_ptr<DiscreteInterpolation2D>
            interp2d_pL, interp2d_pS, interp2d_cfS;

        std::unique_ptr<DiscreteInterpolation1D>
            interp_kp_pL, interp_kz_pL, interp_kp_pS, interp_kz_pS,
            interp_p1d;

        std::unique_ptr<fidcosmo::FlatLCDM> cosmo;

        void _calcVarLss();
        void _cacheInterp2D();
        void _construcP1D();
        void _getCorrFunc2dS();

    public:
        /* This function reads following keys from config file:
        b_F: double
        alpha_F (double): Redshift growth power of b_F.
        beta_F: double
        k_p: double
        q_1: double
        nu_0 (double): b_nu - a_nu of arinyo
        nu_1 (double): b_nu of arinyo
        k_nu: double
        */
        ArinyoP3DModel(ConfigFile &config);

        const fidcosmo::FlatLCDM* getCosmoPtr() const { return cosmo.get(); }

        double getRedshiftEvolution(double z1) {
            return interp_growth->evaluate(z1);
        }

        double evalExplicit(double k, double kz);

        double evaluate(double kperp, double kz) const {
            /* Evaluate large-scale P3D using interpolation. */
            if ((kz == 0) && (kperp == 0))
                return 0;
            else if (kz == 0)
                return exp(interp_kp_pL->evaluate(log(kperp)));
            else if (kperp == 0)
                return exp(interp_kz_pL->evaluate(log(kz)));

            return exp(interp2d_pL->evaluate(log(kz), log(kperp)));
        }

        double evaluateSS(double kperp, double kz) const {
            /* Evaluate small-scale P3D using interpolation. */
            if ((kz == 0) && (kperp == 0))
                return 0;
            else if (kz == 0)
                return exp(interp_kp_pS->evaluate(log(kperp)));
            else if (kperp == 0)
                return exp(interp_kz_pS->evaluate(log(kz)));

            return exp(interp2d_pS->evaluate(log(kz), log(kperp)));
        }

        double evalP1d(double kz) const {
            if (kz < 1e-6) return exp(interp_p1d->evaluate(-13.8155));
            return exp(interp_p1d->evaluate(log(kz)));
        }

        double evalCorrFunc2dS(double rperp, double rz) const {
            /* Evaluate small-scale CF using interpolation. */
            return interp2d_cfS->evaluate(rz, rperp);
        }

        double getVarLss() const { return _varlss; }

        void write(ioh::Qu3dFile *out);
    };
}

#endif
