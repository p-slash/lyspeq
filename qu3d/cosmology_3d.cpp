#include <functional>

#include <gsl/gsl_integration.h>

#include "qu3d/cosmology_3d.hpp"
#include "io/io_helper_functions.hpp"
#include "mathtools/interpolation.hpp"
#include "qu3d/ps2cf_2d.hpp"


constexpr double SAFE_ZERO = 1E-300;
constexpr double TWO_PI2 = 2 * MY_PI * MY_PI;
constexpr double KMIN = 1E-6, KMAX = 2E2;
constexpr int Nhankel = 2048, ltrunc = 512;
const double LNKMIN = log(KMIN), LNKMAX = log(KMAX);

using namespace fidcosmo;

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
double _calcHubble(double z1, struct CosmoParams *params) {
    double z3 = z1 * z1 * z1;
    double nu = 1.0 + _nu_relative_density(z1);
    return params->H0 * sqrt(
            params->Omega_L + (params->Omega_m + params->Omega_r * nu * z1) * z3
    );
}


double _calcInvHubble(double z1, void *params) {
    struct CosmoParams *cosmo_params = (struct CosmoParams *) params;
    return 1.0 / _calcHubble(z1, cosmo_params);
}


double _integrand_LinearGrowth(double z1, void *params)
{
    struct CosmoParams *cosmo_params = (struct CosmoParams *) params;

    double hub = _calcHubble(z1, cosmo_params);
    hub *= hub * hub;

    return z1 / hub;
}


#define ABS_ERROR 0
#define REL_ERROR 1E-8
#define WORKSPACE_SIZE 3000

void FlatLCDM::_integrateComovingDist(
        int nz, const double *z1arr, double *cDist
) {
    gsl_function F;
    F.function = _calcInvHubble;
    F.params = &cosmo_params;

    gsl_integration_workspace *w =
        gsl_integration_workspace_alloc(WORKSPACE_SIZE);

    if (w == NULL)
        throw std::bad_alloc();

    for (int i = 0; i < nz; ++i) {
        double error = 0;

        gsl_integration_qag(
            &F, 1.0, z1arr[i],
            ABS_ERROR, REL_ERROR,
            WORKSPACE_SIZE, GSL_INTEG_GAUSS31, w,
            cDist + i, &error);

        cDist[i] *= SPEED_OF_LIGHT;
    }

    gsl_integration_workspace_free(w);
}


void FlatLCDM::_integrateLinearGrowth(
        int nz, const double *z1arr, double *linD
) {
    double error = 0, linD0 = 0;
    gsl_function F;
    F.function = _integrand_LinearGrowth;
    F.params = &cosmo_params;

    gsl_integration_workspace *w =
        gsl_integration_workspace_alloc(WORKSPACE_SIZE);

    if (w == NULL)
        throw std::bad_alloc();

    gsl_integration_qagiu(
        &F, 1.0,
        ABS_ERROR, REL_ERROR,
        WORKSPACE_SIZE, w,
        &linD0, &error);

    linD0 *= getHubble(1.0);

    for (int i = 0; i < nz; ++i) {
        gsl_integration_qagiu(
            &F, z1arr[i],
            ABS_ERROR, REL_ERROR,
            WORKSPACE_SIZE, w,
            linD + i, &error);

        linD[i] *= getHubble(z1arr[i]) / linD0;
    }

    gsl_integration_workspace_free(w);
}

#undef ABS_ERROR
#undef REL_ERROR
#undef WORKSPACE_SIZE


FlatLCDM::FlatLCDM(ConfigFile &config) {
    config.addDefaults(planck18_default_parameters);
    cosmo_params.H0 = config.getDouble("Hubble");
    cosmo_params.Omega_m = config.getDouble("OmegaMatter");
    cosmo_params.Omega_r = config.getDouble("OmegaRadiation");
    cosmo_params.Omega_L =
        1.0 - cosmo_params.Omega_m
        - cosmo_params.Omega_r * (1 + _nu_relative_density(1));

    // Cache
    constexpr int nz = 601;
    constexpr double dz = 0.01;
    double z1arr[nz], temparr[nz];
    for (int i = 0; i < nz; ++i) {
        z1arr[i] = 1.0 + dz * i;
        temparr[i] = _calcHubble(z1arr[i], &cosmo_params);
    }

    hubble_z = std::make_unique<DiscreteCubicInterpolation1D>(
        z1arr[0], dz, nz, temparr);

    _integrateComovingDist(nz, z1arr, temparr);
    interp_comov_dist = std::make_unique<DiscreteCubicInterpolation1D>(
        z1arr[0], dz, nz, temparr);

    double dchi = (temparr[nz - 1] - temparr[0]) / (nz - 1), chi0 = temparr[0];
    Interpolation _inv_comoving_interp(
        GSL_CUBIC_INTERPOLATION, temparr, z1arr, nz);

    for (int i = 0; i < nz; ++i)
        temparr[i] = _inv_comoving_interp.evaluate(chi0 + i * dchi);

    interp_invcomov_dist = std::make_unique<DiscreteCubicInterpolation1D>(
        chi0, dchi, nz, temparr);

    _integrateLinearGrowth(nz, z1arr, temparr);
    linear_growth = std::make_unique<DiscreteCubicInterpolation1D>(
        z1arr[0], dz, nz, temparr);
}


std::vector<double> LinearPowerInterpolator::_appendLinearExtrapolation(
        double lnk1, double lnk2, double dlnk, int N,
        const std::vector<double> &lnP, double &newlnk1
) {
    double m1 = lnP[1] - lnP[0],
           m2 = lnP[N - 1] - lnP[N - 2];

    int n1 = std::max(0, int((lnk1 - LNKMIN) / dlnk)),
        n2 = std::max(0, int((LNKMAX - lnk2) / dlnk));

    newlnk1 = lnk1 - n1 * dlnk;
    std::vector<double> result(n1 + N + n2);

    for (int i = 0; i < n1; ++i)
        result[i] = lnP[0] + m1 * (i - n1);

    std::copy(lnP.begin(), lnP.end(), result.begin() + n1);

    for (int i = 0; i < n2; ++i)
        result[n1 + N + i] = lnP.back() + m2 * (i + 1);

    return result;
}


void LinearPowerInterpolator::_readFile(const std::string &fname) {
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

    for (int i = 1; i < N - 1; ++i)
        if (fabs(lnk[i + 1] - lnk[i] - dlnk) > 1e-8)
            throw std::runtime_error(
                "Input PlinearFilename does not have equal ln k spacing.");

    auto appendLnp = _appendLinearExtrapolation(
        lnk[0], lnk.back(), dlnk, N, lnP, lnk1);
    interp_lnp = std::make_unique<DiscreteCubicInterpolation1D>(
        lnk1, dlnk, appendLnp.size(), appendLnp.data());
}


void LinearPowerInterpolator::write(ioh::Qu3dFile *out) {
    out->write(interp_lnp->get(), interp_lnp->size(), "PLIN_APPD");
}


const std::unordered_map<std::string, double> metal_line_map ({
    {"b_SiIII-1207", 1206.52}, {"b_SiII-1190", 1190.42},
    {"b_SiII-1193", 1193.28}, {"b_SiII-1260", 1260.42},
});

ArinyoP3DModel::ArinyoP3DModel(ConfigFile &config) : _varlss(0) {
    config.addDefaults(arinyo_default_parameters);
    config.addDefaults(metals_default_parameters);
    b_F = config.getDouble("b_F");
    alpha_F = config.getDouble("alpha_F");
    beta_F = config.getDouble("beta_F");
    k_p = config.getDouble("k_p");
    q_1 = config.getDouble("q_1");
    a_nu = config.getDouble("a_nu");
    b_nu = config.getDouble("b_nu");
    k_nu = config.getDouble("k_nu");
    b_HCD = config.getDouble("b_HCD");
    beta_HCD = config.getDouble("beta_HCD");
    L_HCD = config.getDouble("L_HCD");
    beta_metal = config.getDouble("beta_metal");
    sigma_v = config.getDouble("sigma_v");

    KMAX_HALO = std::min(1.5, k_p);
    interp_p = std::make_unique<LinearPowerInterpolator>(config);
    cosmo = std::make_unique<fidcosmo::FlatLCDM>(config);
    rscale_long = config.getDouble("LongScale");
    rmax = rscale_long * config.getDouble("ScaleFactor");
    _z1_pivot = 1.0 + interp_p->z_pivot;
    _sigma_mpc = 0;
    _deltar_mpc = 0;

    for (const auto &[key, wave_m] : metal_line_map) {
        double b = config.getDouble(key);
        if (b == 0)
            continue;

        double d = fabs(cosmo->getComovingDist(_z1_pivot * LYA_REST / wave_m)
                        - cosmo->getComovingDist(_z1_pivot));
        b_dr_pair_metals.push_back(std::make_pair(b, d));
    }

    // This is tiny ~2-3 Mpc
    // L_metal = (cosmo->getComovingDist((1.0 + bins::Z_LOWER_EDGE) * alpha_si)
    //            - cosmo->getComovingDist((1.0 + bins::Z_UPPER_EDGE) * alpha_si))
    //           - (cosmo->getComovingDist(1.0 + bins::Z_LOWER_EDGE)
    //              - cosmo->getComovingDist(1.0 + bins::Z_UPPER_EDGE));
    // L_metal = fabs(L_metal);
}


void ArinyoP3DModel::construct() {
    _construcP1D();
    _cacheInterp2D();
    _getCorrFunc2dS();
    _calcMultipoles();

    _D_pivot = cosmo->getLinearGrowth(_z1_pivot);
    constexpr int nz = 401;
    constexpr double dz = 0.01, z1_i = 2.9;
    double growth[nz];

    for (int i = 0; i < nz; ++i) {
        double z1 = z1_i + dz * i;
        growth[i] = cosmo->getLinearGrowth(z1) / _D_pivot
                    * pow(z1 / 3.4, alpha_F);
    }
    interp_growth = std::make_unique<DiscreteCubicInterpolation1D>(
        z1_i, dz, nz, &growth[0]);
}


double ArinyoP3DModel::getSpectroWindow2(double kz) const {
    #ifdef TURN_OFF_SPECTRO_WINDOW
    return 1.0;

    #else
    if (kz == 0)  return 1;
    double kr = kz * _sigma_mpc, kv = kz * _deltar_mpc / 2.0;
    kr *= kr;
    kv = sin(kv) / kv;
    return exp(-kr) * kv * kv;

    #endif
}


void ArinyoP3DModel::calcVarLss(bool pp_enabled) {
    constexpr int nlnk = 6001;
    const double dlnk = (LNKMAX - LNKMIN) / (nlnk - 1);
    double powers_kz[nlnk], powers_kperp[nlnk];

    std::function<double(double)> eval_ls_supp;
    if (pp_enabled)
        eval_ls_supp = [](double k) { return 1; };
    else {
        double rl = rscale_long;
        eval_ls_supp = [rl](double k) {
            double k_rL = k * rl;
            k_rL *= -k_rL;
            return exp(k_rL);
        };
    }

    for (int i = 0; i < nlnk; ++i) {
        double kperp2 = exp(LNKMIN + i * dlnk);
        kperp2 *= kperp2;
        for (int j = 0; j < nlnk; ++j) {
            double kz = exp(LNKMIN + j * dlnk), k = sqrt(kperp2 + kz * kz);
            powers_kz[j] = kz * evalExplicit(k, kz);
            powers_kz[j] *= eval_ls_supp(k);
        }

        powers_kperp[i] = kperp2 * trapz(powers_kz, nlnk, dlnk);
    }

    _varlss = trapz(powers_kperp, nlnk, dlnk) / TWO_PI2;
}


void ArinyoP3DModel::_cacheInterp2D() {
    constexpr double dlnk = 0.02;
    const int N = ceil((LNKMAX - LNKMIN) / dlnk);
    auto lnP_L = std::make_unique<double[]>(N * N),
         lnP_S = std::make_unique<double[]>(N * N),
         lnP_T = std::make_unique<double[]>(N * N);

    /* Large-scale and small-scale 2Ds */
    for (int iperp = 0; iperp < N; ++iperp) {
        double kperp = exp(LNKMIN + iperp * dlnk);

        for (int iz = 0; iz < N; ++iz) {
            double kz = exp(LNKMIN + iz * dlnk),
                   k = sqrt(kperp * kperp + kz * kz),
                   k_rL = k * rscale_long;

            double ptot = log(evalExplicit(k, kz) + SAFE_ZERO);

            /* Large-scale 2D */
            k_rL *= -k_rL;
            lnP_L[iz + N * iperp] = ptot + k_rL;
            /* Small-scale 2D */
            k_rL = log(1.0 - exp(k_rL) + SAFE_ZERO);
            lnP_S[iz + N * iperp] = ptot + k_rL;
            /* Total 2D */
            lnP_T[iz + N * iperp] = ptot;
        }
    }

    interp2d_pL.setInterp2D(LNKMIN, dlnk, LNKMIN, dlnk, lnP_L.get(), N, N);
    interp2d_pS.setInterp2D(LNKMIN, dlnk, LNKMIN, dlnk, lnP_S.get(), N, N);
    interp2d_pT.setInterp2D(LNKMIN, dlnk, LNKMIN, dlnk, lnP_T.get(), N, N);

    /* Large-scale and small-scale 1Ds */
    for (int i = 0; i < N; ++i) {
        double k = exp(LNKMIN + i * dlnk), k_rL = k * rscale_long;
        k_rL *= -k_rL;

        double pperp = log(evalExplicit(k, 0) + SAFE_ZERO),
               pz = log(evalExplicit(k, k) + SAFE_ZERO);

        /* Large-scale 1D */
        lnP_L[i] = pperp + k_rL;
        lnP_L[i + N] = pz + k_rL;

        /* Small-scale 1D */
        k_rL = log(1.0 - exp(k_rL) + SAFE_ZERO);
        lnP_L[i + 2 * N] = pperp + k_rL;
        lnP_L[i + 3 * N] = pz + k_rL;

        /* Total 1D */
        lnP_L[i + 4 * N] = pperp;
        lnP_L[i + 5 * N] = pz;
    }

    interp2d_pL.setInterpX(LNKMIN, dlnk, N, lnP_L.get());
    interp2d_pL.setInterpY(LNKMIN, dlnk, N, lnP_L.get() + N);
    interp2d_pS.setInterpX(LNKMIN, dlnk, N, lnP_L.get() + 2 * N);
    interp2d_pS.setInterpY(LNKMIN, dlnk, N, lnP_L.get() + 3 * N);
    interp2d_pT.setInterpX(LNKMIN, dlnk, N, lnP_L.get() + 4 * N);
    interp2d_pT.setInterpY(LNKMIN, dlnk, N, lnP_L.get() + 5 * N);
}


double ArinyoP3DModel::evalP1d(double kz) const {
    const static double p1d0 = exp(interp1d_pT->evaluate(LNKMIN));
    if (kz < KMIN)  return p1d0;
    if (kz > KMAX)  return 0;
    double p1d = exp(interp1d_pT->evaluate(log(kz))) - SAFE_ZERO;
    if (p1d < 0)  return 0;
    return p1d;
}


void ArinyoP3DModel::_construcP1D() {
    constexpr int nlnk = 10001;
    const double dlnk = (LNKMAX - LNKMIN) / (nlnk - 1), dlnk2 = 0.02;
    const int nlnk2 = ceil((LNKMAX - LNKMIN) / dlnk2);
    double p1d_integrand[nlnk], p1d[nlnk2];

    for (int i = 0; i < nlnk2; ++i) {
        double kz = exp(LNKMIN + i * dlnk2);

        for (int j = 0; j < nlnk; ++j) {
            double kperp2 = exp(LNKMIN + j * dlnk);
            kperp2 *= kperp2;
            double k = sqrt(kperp2 + kz * kz);

            p1d_integrand[j] = kperp2 * evalExplicit(k, kz);
        }
        p1d[i] = log(trapz(p1d_integrand, nlnk, dlnk) / (2.0 * MY_PI) + SAFE_ZERO);
    }

    interp1d_pT = std::make_unique<DiscreteCubicInterpolation1D>(
        LNKMIN, dlnk2, nlnk2, &p1d[0]);

    /* Quasar forest length can be a maximum of 650 Mpc.
       Truncating 1e-6--1e6 logspaced array of 1536 points by 380 on each end,
       Truncating 1e-6--1e6 logspaced array of 2048 points by 512 on each end,
       Truncating 1e-4--1e4 logspaced array of 1536 points by 190 on each end,
       gives approximately 1e-3--1e3 Mpc span. */
    const double log2_e = log2(exp(1.0)), SQRT_2PI = sqrt(2.0 * MY_PI);

    FFTLog fht_z(Nhankel);
    fht_z.construct(-0.5, KMIN, 1 / KMIN, 0, 0);

    for (int iz = 0; iz < Nhankel; ++iz)
        fht_z.field[iz] = evalP1d(fht_z.r[iz]) * sqrt(fht_z.r[iz]);

    fht_z.transform();

    for (int iz = 0; iz < Nhankel; ++iz)
        fht_z.field[iz] /= SQRT_2PI * sqrt(fht_z.k[iz]);

    // Smoother smoother(1);
    // smoother.smooth1D(fht_z.field + truncate, Nhankel - 2 * truncate, 1, true);

    interp1d_cfT = std::make_unique<DiscreteCubicInterpolation1D>(
        log2(fht_z.k[ltrunc]), log2_e * fht_z.getDLn(),
        Nhankel - (2 * ltrunc), fht_z.field + ltrunc);
}


void ArinyoP3DModel::_getCorrFunc2dS() {
    Ps2Cf_2D hankel{Nhankel, KMIN, 1 / KMIN};

    auto psarr = std::make_unique<double[]>(Nhankel * Nhankel);
    const double *kperparr = hankel.getKperp(), *kzarr = hankel.getKz();

    for (int iperp = 0; iperp < Nhankel; ++iperp) {
        double kperp2 = kperparr[iperp] * kperparr[iperp];

        for (int iz = 0; iz < Nhankel; ++iz) {
            double k = sqrt(kperp2 + kzarr[iz] * kzarr[iz]),
                   k_rL = k * rscale_long;
            psarr[iz + Nhankel * iperp] = evalExplicit(k, kzarr[iz])
                                          * (1.0 - exp(-k_rL * k_rL));
        }
    }

    #ifndef NUSE_LOGR_INTERP
        interp2d_cfS = hankel.transform<INTERP_COSMO_2D>(
            psarr.get(), ltrunc, ltrunc, 0, true);
    #else
        interp2d_cfS = hankel.transform<INTERP_COSMO_2D>(
            psarr.get(), ltrunc, ArinyoP3DModel::MAX_R_FACTOR * rscale_long);
    #endif

    // Apodize interp2d_cfS only
    // interp2d_cfS->applyFunction(
    //     [this](double log2rz, double log2rperp) {
    //         double r2 = exp2(2.0 * log2rz) + exp2(2.0 * log2rperp);
    //         return tophat2(r2);
    // });

    interp1d_cfS = interp2d_cfS->get1dSliceX<DiscreteCubicInterpolation1D>(
        interp2d_cfS->getY1());
}


void ArinyoP3DModel::_calcMultipoles() {
    constexpr int nmu = 501;
    constexpr double dmu = 1.0 / (nmu - 1), dlnk = 0.02;
    const int nlnk = ceil((LNKMAX - LNKMIN) / dlnk);
    double p3d_l_integrand[nmu], p3d_l[nlnk];

    for (int l = 0; l < ArinyoP3DModel::MAX_NUM_L; ++l) {
        for (int i = 0; i < nlnk; ++i) {
            double k = exp(LNKMIN + i * dlnk);

            for (int j = 0; j < nmu; ++j) {
                double mu = j * dmu;

                p3d_l_integrand[j] = legendre(2 * l, mu) * evalExplicit(k, k * mu);
            }

            p3d_l[i] = trapz(p3d_l_integrand, nmu, dmu) * (4 * l + 1);
        }

        p3d_ell_T.setInterpEll(l, LNKMIN, dlnk, nlnk, p3d_l);
    }

    FFTLog fht(Nhankel);
    int trim = 128, Nres = Nhankel - 2 * trim;
    double MY_2PI_CUBED = MY_PI * 2.0, window_xi = 5e3;
    MY_2PI_CUBED *= MY_2PI_CUBED * MY_2PI_CUBED;
    MY_2PI_CUBED = sqrt(MY_2PI_CUBED);

    for (int l = 0; l < ArinyoP3DModel::MAX_NUM_L; ++l) {
        double mu = 2 * l + 0.5;
        fht.construct(mu, KMIN, 1 / KMIN, 0, 0);

        for (int i = 0; i < Nhankel; ++i)
            fht.field[i] = p3d_ell_T.evaluateEll(l, log(fht.k[i]))
                            * fht.k[i] * sqrt(fht.k[i]);

        fht.transform();
        int sgn = (l % 2 == 0) ? 1 : -1;
        for (int i = 0; i < Nhankel; ++i) {
            fht.field[i] /= sgn * MY_2PI_CUBED * sqrt(fht.r[i]) * fht.r[i];
            if (fht.r[i] > window_xi) {
                double supp = (fht.r[i] - window_xi) / (window_xi / 2.0);
                fht.field[i] *= exp(-supp * supp);
            }
        }

        xi_ell_T.setInterpEll(
            l, log(fht.k[trim]), fht.getDLn(), Nres, fht.field + trim);
    }
}


double ArinyoP3DModel::getMetalTerm(
        double kz, double mu2, double bbeta_lya, double lnD
) const {
    if (b_dr_pair_metals.empty())
        return 0;

    double result = 0.0, dfog = kz * sigma_v, dfogxlya;
    dfog = 1.0 / (1.0 + dfog * dfog);
    dfogxlya = sqrt(lnD * dfog);

    std::vector<std::pair<double, double>> bbeta_dr_pair_metals;

    for (const auto &[b_m, dr_m] : b_dr_pair_metals) {
        bbeta_dr_pair_metals.push_back(std::make_pair(
            b_m * (1.0 + beta_metal * mu2), dr_m));
    }

    // xLya & auto metal terms
    for (const auto &[bbeta_m, dr_m] : bbeta_dr_pair_metals)
        result += 2.0 * bbeta_lya * bbeta_m * cos(kz * dr_m) * dfogxlya
                  + bbeta_m * bbeta_m * dfog;

    for (const auto &[bbeta_m1, dr_m1] : bbeta_dr_pair_metals)
        for (const auto &[bbeta_m2, dr_m2] : bbeta_dr_pair_metals)
            result += 2.0 * bbeta_m1 * bbeta_m2
                      * cos(kz * (dr_m1 - dr_m2)) * dfog;

    return result;
}

double ArinyoP3DModel::evalExplicit(double k, double kz) const {
    if (k == 0)
        return 0;

    double
    plin = interp_p->evaluate(k),
    delta2_L = plin * k * k * k / TWO_PI2,
    k_kp = k / k_p,
    mu = kz / k, mu2 = mu * mu,
    bbeta_lya = b_F * (1.0 + beta_F * mu2),
    bbeta_hcd_kz = b_HCD * (1 + beta_HCD * mu2) * exp(-L_HCD * kz),
    result, lnD, apod_halo;

    double u1;
    if (b_nu == 0)  u1 = 1.0;
    else  u1 = pow(mu, b_nu);

    if (a_nu != 0)  u1 *= pow(k / k_nu, a_nu);

    lnD = exp((q_1 * delta2_L) * (1.0 - u1) - k_kp * k_kp);

    if (k > KMAX_HALO)
        apod_halo = 0;
    else if (k < (KMAX_HALO / 2.0))
        apod_halo = 1.0;
    else {
        apod_halo = cos((2.0 * k / KMAX_HALO - 1.0) * MY_PI / 2.0);
        apod_halo *= apod_halo;
    }

    result = plin * (
        bbeta_lya * bbeta_lya * lnD + apod_halo * (
            + 2.0 * bbeta_lya * bbeta_hcd_kz
            + bbeta_hcd_kz * bbeta_hcd_kz
            + getMetalTerm(kz, mu2, bbeta_lya, lnD)));

    #ifdef KILL_15MPC_POWER
    double kill_cut_ = 15.0 * k;
    result *= exp(-kill_cut_ * kill_cut_);
    #endif

    #ifdef TURN_OFF_SPECTRO_WINDOW
    return result;
    #else
    if ((_sigma_mpc == 0) && (_deltar_mpc == 0))
        return result;

    return result * getSpectroWindow2(kz);
    #endif
}


void ArinyoP3DModel::write(ioh::Qu3dFile *out) {
    constexpr int nlnk = 502, nlnk2 = nlnk * nlnk;
    const double dlnk = (LNKMAX - LNKMIN) / (nlnk - 2);
    double karr[nlnk], pmarr[nlnk2];

    interp_p->write(out);

    for (int i = 0; i < nlnk; ++i) {
        karr[i] = 2.9 + i * 0.008;
        pmarr[i] = getRedshiftEvolution(karr[i]);
    }
    out->write(karr, nlnk, "ZMODEL");
    out->write(pmarr, nlnk, "G12");

    karr[0] = 0;
    for (int i = 1; i < nlnk; ++i)
        karr[i] = exp(LNKMIN + (i - 1) * dlnk);

    out->write(karr, nlnk, "KMODEL");

    for (int iperp = 0; iperp < nlnk; ++iperp)
        for (int iz = 0; iz < nlnk; ++iz)
            pmarr[iz + nlnk * iperp] = interp2d_pL.evaluate(karr[iperp], karr[iz]);

    out->write(pmarr, nlnk2, "PMODEL_L");
    out->flush();

    for (int iperp = 0; iperp < nlnk; ++iperp)
        for (int iz = 0; iz < nlnk; ++iz)
            pmarr[iz + nlnk * iperp] = interp2d_pS.evaluate(karr[iperp], karr[iz]);

    out->write(pmarr, nlnk2, "PMODEL_S");
    out->flush();

    for (int iperp = 0; iperp < nlnk; ++iperp)
        for (int iz = 0; iz < nlnk; ++iz)
            pmarr[iz + nlnk * iperp] = interp2d_pT.evaluate(karr[iperp], karr[iz]);

    out->write(pmarr, nlnk2, "PMODEL_T");
    out->flush();

    for (int iz = 0; iz < nlnk; ++iz)
        pmarr[iz] = evalP1d(karr[iz]);

    out->write(pmarr, nlnk, "PMODEL_1D");

    for (int l = 0; l < ArinyoP3DModel::MAX_NUM_L; ++ l)
        for (int iz = 0; iz < nlnk; ++iz)
            pmarr[iz + l * nlnk] = evalP3dL(karr[iz], l);

    for (int l = 0; l < ArinyoP3DModel::MAX_NUM_L; ++ l)
        out->write(pmarr + l * nlnk, nlnk,
                   std::string("P3D_L") + std::to_string(2 * l));
    out->flush();

    constexpr int nr = 512, nr2 = nr * nr;
    double rarr[nr], cfsarr[nr2];

    #ifndef NUSE_LOGR_INTERP
        const double r1 = exp2(interp2d_cfS->getX1()), r2 = 1e4,
                     dlnr = log(r2 / r1) / (nr - 1);
        for (int i = 0; i < nr; ++i)
            rarr[i] = r1 * exp(i * dlnr);
    #else
        const double r2 = ArinyoP3DModel::MAX_R_FACTOR * rscale_long, dr = r2 / nr;
        for (int i = 0; i < nr; ++i)
            rarr[i] = i * dr;
    #endif

    out->write(rarr, nr, "RMODEL");

    for (int iperp = 0; iperp < nr; ++iperp)
        for (int iz = 0; iz < nr; ++iz)
            cfsarr[iz + nr * iperp] = evalCorrFunc2dS(rarr[iperp], rarr[iz]);

    out->write(cfsarr, nr2, "CFMODEL_S_2D");
    for (int iz = 0; iz < nr; ++iz)
        cfsarr[iz] = evalCorrFunc1dT(rarr[iz]);

    out->write(cfsarr, nr, "CFMODEL_T_1D");
    out->flush();

    for (int i = 0; i < nr; ++i) {
        for (int l = 0; l < ArinyoP3DModel::MAX_NUM_L; ++l)
            cfsarr[i + l * nr] = xi_ell_T.evaluateEll(l, log(rarr[i]));
    }

    for (int l = 0; l < ArinyoP3DModel::MAX_NUM_L; ++ l)
        out->write(cfsarr + l * nr, nr,
                   std::string("Xiell_L") + std::to_string(2 * l));
    out->flush();
}

#if 0
void ArinyoP3DModel::_cacheInterp2D_total() {
    constexpr double dlnk = 0.02;
    const int N = ceil((LNKMAX - LNKMIN) / dlnk);
    auto lnP_T = std::make_unique<double[]>(N * N);

    /* Large-scale and small-scale 2Ds */
    for (int iperp = 0; iperp < N; ++iperp) {
        double kperp = exp(LNKMIN + iperp * dlnk);

        for (int iz = 0; iz < N; ++iz) {
            double kz = exp(LNKMIN + iz * dlnk),
                   k = sqrt(kperp * kperp + kz * kz);

            lnP_T[iz + N * iperp] = log(evalExplicit(k, kz) + SAFE_ZERO);
        }
    }

    interp2d_pT.setInterp2D(LNKMIN, dlnk, LNKMIN, dlnk, lnP_T.get(), N, N);

    /* Large-scale and small-scale 1Ds */
    for (int i = 0; i < N; ++i) {
        double k = exp(LNKMIN + i * dlnk);

        lnP_T[i] = log(evalExplicit(k, 0) + SAFE_ZERO);
        lnP_T[i + N] = log(evalExplicit(k, k) + SAFE_ZERO);
    }

    interp2d_pT.setInterpX(LNKMIN, dlnk, N, lnP_T.get());
    interp2d_pT.setInterpY(LNKMIN, dlnk, N, lnP_T.get() + N);
}


void ArinyoP3DModel::_cacheInterp2D_multipoles() {
    constexpr double dlnk = 0.02;
    const int N = ceil((LNKMAX - LNKMIN) / dlnk);

    MultipoleInterpolation Pell_S;
    FFTLog fht(Nhankel);

    int Nres = Nhankel - 2 * ltrunc;
    double MY_2PI_CUBED = MY_PI * 2.0;
    MY_2PI_CUBED *= MY_2PI_CUBED * MY_2PI_CUBED;
    MY_2PI_CUBED = sqrt(MY_2PI_CUBED);

    for (int l = 0; l < ArinyoP3DModel::MAX_NUM_L; ++l) {
        double mu = 2 * l + 0.5;
        fht.construct(mu, KMIN, 1 / KMIN, 0, 0);

        for (int i = 0; i < Nhankel; ++i) {
            double r2 = fht.k[i] * fht.k[i];
            fht.field[i] = xi_ell_T.evaluateEll(l, log(fht.k[i]))
                            * fht.k[i] * sqrt(fht.k[i]) * tophat2(r2);
        }

        fht.transform();
        int sgn = (l % 2 == 0) ? 1 : -1;
        for (int i = 0; i < Nhankel; ++i)
            fht.field[i] *= sgn * MY_2PI_CUBED / (sqrt(fht.r[i]) * fht.r[i]);

        Pell_S.setInterpEll(
            l, log(fht.k[ltrunc]), fht.getDLn(), Nres, fht.field + ltrunc);
    }

    interp2d_pS = Pell_S.toDiscreteLogInterpolation2D(LNKMIN, dlnk, N);

    // Large scale
    auto lnP_T = std::make_unique<double[]>(N * N);

    /* Large-scale and small-scale 2Ds */
    for (int iperp = 0; iperp < N; ++iperp) {
        double kperp = exp(LNKMIN + iperp * dlnk);

        for (int iz = 0; iz < N; ++iz) {
            double kz = exp(LNKMIN + iz * dlnk);

            lnP_T[iz + N * iperp] =    (interp2d_pT.evaluate(kperp, kz)
                                        - interp2d_pS.evaluate(kperp, kz)
                                        + SAFE_ZERO);
        }
    }

    interp2d_pL.setInterp2D(LNKMIN, dlnk, LNKMIN, dlnk, lnP_T.get(), N, N);

    /* Large-scale and small-scale 1Ds */
    for (int i = 0; i < N; ++i) {
        double k = exp(LNKMIN + i * dlnk);

        lnP_T[i] = (interp2d_pT.evaluate(k, 0) - interp2d_pS.evaluate(k, 0)
                       + SAFE_ZERO);
        lnP_T[i + N] = (interp2d_pT.evaluate(k, k)
                           - interp2d_pS.evaluate(k, k) + SAFE_ZERO);
    }

    interp2d_pL.setInterpX(LNKMIN, dlnk, N, lnP_T.get());
    interp2d_pL.setInterpY(LNKMIN, dlnk, N, lnP_T.get() + N);
}
#endif
