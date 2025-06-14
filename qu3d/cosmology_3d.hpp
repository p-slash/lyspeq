#ifndef COSMOLOGY_3D_H
#define COSMOLOGY_3D_H

#include <cmath>
#include <memory>
#include <string>

#include "core/global_numbers.hpp"
#include "mathtools/discrete_interpolation.hpp"
#include "mathtools/multipole_interpolation.hpp"
#include "mathtools/mathutils.hpp"
#include "io/config_file.hpp"
#include "qu3d/qu3d_file.hpp"


#ifndef INTERP_COSMO_2D
#define INTERP_COSMO_2D DiscreteBicubicSpline
#endif

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
            interp_comov_dist, hubble_z, linear_growth, interp_invcomov_dist;

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

        double getZ1FromComovingDist(double chi) const {
            return interp_invcomov_dist->evaluate(chi);
        }

        double getLinearGrowth(double z1) const {
            return linear_growth->evaluate(z1);
        }
    };

    class LinearPowerInterpolator {
        /* Power spectrum interpolator in Mpc units */
        std::unique_ptr<DiscreteCubicInterpolation1D> interp_lnp;

        std::vector<double> _appendLinearExtrapolation(
            double lnk1, double lnk2, double dlnk, int N,
            const std::vector<double> &lnP, double &newlnk1
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

        void write(ioh::Qu3dFile *out);
    };

    const config_map arinyo_default_parameters ({
        /* These defaults are obtained from Chabanier+24 Table A3 for z=2.4.
           b_F fitted to all redshift ranges accounting for errors.
           Two redshift bins are averaged for others. */
        {"b_F", "0.1195977"}, {"alpha_F", "3.37681"},
        {"beta_F", "1.6633"}, {"k_p", "0.4"}, // {"k_p", "16.802"},
        {"q_1", "0.796"}, {"a_nu", "0.383"}, {"b_nu", "1.65"},
        {"k_nu", "0.3922"},
        {"b_HCD", "0.05"}, {"beta_HCD", "0.7"}, {"L_HCD", "14.8"}
    });

    const config_map metals_default_parameters ({
        {"b_SiIII-1207", "9.8e-3"}, {"b_SiII-1190", "4.5e-3"},
        {"b_SiII-1193", "3.05e-3"}, {"b_SiII-1260", "4.0e-3"},
        {"beta_metal", "0.5"}, {"sigma_v", "5.0"}
    });

    class ArinyoP3DModel {
    public:
        static constexpr double MAX_R_FACTOR = 20.0;
    private:
        double _varlss, _D_pivot, _z1_pivot, _sigma_mpc, _deltar_mpc;
        double b_F, alpha_F, beta_F, k_p, q_1, a_nu, b_nu, k_nu, rscale_long, rmax;
        double b_HCD, beta_HCD, L_HCD, beta_metal, sigma_v;
        double KMAX_HALO;

        std::vector<std::pair<double, double>> b_dr_pair_metals;
        std::unique_ptr<LinearPowerInterpolator> interp_p;
        std::unique_ptr<DiscreteCubicInterpolation1D> interp_growth;

        std::unique_ptr<INTERP_COSMO_2D> interp2d_cfS;
        std::unique_ptr<DiscreteCubicInterpolation1D>
            interp1d_pT, interp1d_cfS, interp1d_cfT;
        MultipoleInterpolation xi_ell_T, p3d_ell_T;

        std::unique_ptr<fidcosmo::FlatLCDM> cosmo;

        void _cacheInterp2D();
        void _construcP1D();
        void _getCorrFunc2dS();
        void _calcMultipoles();
        double apodize(double r) const {
            // const static double _rmax_half = rmax / 2.0;
            const static double _r1 = 0.75 * rmax, _dr = rmax - _r1;
            // if (r > rmax)   return 0.0;

            if (r < _r1) return 1.0;
            // double y = cos((r / _rmax_half - 1.0) * MY_PI / 2.0);
            double y = cos((r - _r1) / _dr * MY_PI / 2.0);
            y *= y;
            return y;
        }

        double tophat2(double r2) const {
            const static double _rmax_half = rmax / 2.0;
            double r = sqrt(r2) / _rmax_half;
            if (r > 2)  return 0;
            return 1.0 - 0.75 * r + r * r * r / 16.0;
        }

        double getMetalTerm(
            double kz, double mu2, double bbeta_lya, double lnD) const;

    public:
        DiscreteLogLogInterpolation2D<
            DiscreteCubicInterpolation1D, INTERP_COSMO_2D
        > interp2d_pL, interp2d_pS, interp2d_pT;
        /* This function reads following keys from config file:
        b_F: double
        alpha_F (double): Redshift growth power of b_F.
        beta_F: double
        k_p: double
        q_1: double
        a_nu: double
        b_nu: double
        k_nu: double
        */
        ArinyoP3DModel(ConfigFile &config);
        void setSpectroParams(double sigma, double delta_r) {
            _sigma_mpc = sigma;  _deltar_mpc = delta_r;
        }
        void construct();
        void calcVarLss(bool pp_enabled);
        double getSpectroWindow2(double kz) const;
        const fidcosmo::FlatLCDM* getCosmoPtr() const { return cosmo.get(); }
        double getRedshiftEvolution(double z1) const {
            return interp_growth->evaluate(z1);
        }
        double evalExplicit(double k, double kz) const;
        double evalP1d(double kz) const;
        double evalP3dL(double k, int l) const {
            return p3d_ell_T.evaluateEll(l, log(k + 1e-300));
        };

        double evalCorrFunc1dT(float rz) const {
            /* Evaluate total (L + S) 1D CF using interpolation.
               rzmin, rzmax boundaries cannot be hit.
            */
            rz = fastlog2(rz);
            return interp1d_cfT->evaluate(rz);
        }

        double evalCorrFunc1dS(float rz) const {
            /* Evaluate small-scale CF using interpolation.
               rzmin boundary cannot be hit.
            */
            // const static float rmaxf = rmax;
            // if (rz > rmaxf)  return 0;
            return interp1d_cfS->evaluate(fastlog2(rz));  // * apodize(rz);
        }

        double getVar1dS() const { return interp1d_cfS->get()[0]; }
        double getVar1dT() const { return interp1d_cfT->get()[0]; }

        #ifndef NUSE_LOGR_INTERP
            double evalCorrFunc2dS(float rperp, float rz) const {
                /* Evaluate small-scale CF using interpolation. */
                const static float
                    rz_min = exp2f(interp2d_cfS->getX1()),
                    rperp_min = exp2f(interp2d_cfS->getY1()),
                    rmax2f = rmax * rmax;

                float r2 = rperp * rperp + rz * rz;
                if (r2 > rmax2f)
                    return 0;

                double rzin, rperpin;
                if (rz < rz_min)
                    rzin = interp2d_cfS->getX1();
                else
                    rzin = fastlog2(rz);

                if (rperp < rperp_min)
                    rperpin = interp2d_cfS->getY1();
                else
                    rperpin = fastlog2(rperp);

                return interp2d_cfS->evaluateHermite2(rzin, rperpin)
                       * apodize(sqrt(r2));
            }
        #else
            double evalCorrFunc2dS(float rperp, float rz) const {
                /* Evaluate small-scale CF using interpolation. */
                if (rz > rmax || rperp > rmax)
                    return 0;
                return interp2d_cfS->evaluate(rz, rperp);
            }
        #endif

        double getVarLss() const { return _varlss; }
        void write(ioh::Qu3dFile *out);
    };
}

#endif
