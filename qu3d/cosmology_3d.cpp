#include <functional>

#include <gsl/gsl_integration.h>

#include "qu3d/cosmology_3d.hpp"
#include "core/global_numbers.hpp"
#include "io/io_helper_functions.hpp"
#include "mathtools/interpolation.hpp"
#include "qu3d/ps2cf_2d.hpp"


constexpr double SAFE_ZERO = 1E-36;
constexpr double TWO_PI2 = 2 * MY_PI * MY_PI;
constexpr double KMIN = 1E-6, KMAX = 2E2,
                 LNKMIN = log(KMIN), LNKMAX = log(KMAX);

using namespace fidcosmo;

/* Fitted to astropy Planck18.nu_relative_density function based on
   Komatsu et al. 2011, eq 26
*/
constexpr double _nu_relative_density(
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


ArinyoP3DModel::ArinyoP3DModel(ConfigFile &config) : _varlss(0) {
    config.addDefaults(arinyo_default_parameters);
    b_F = config.getDouble("b_F");
    alpha_F = config.getDouble("alpha_F");
    beta_F = config.getDouble("beta_F");
    k_p = config.getDouble("k_p");
    q_1 = config.getDouble("q_1");
    nu_0 = config.getDouble("nu_0");
    nu_1 = config.getDouble("nu_1");
    k_nu = config.getDouble("k_nu");
    b_HCD = config.getDouble("b_HCD");
    beta_HCD = config.getDouble("beta_HCD");
    L_HCD = config.getDouble("L_HCD");

    interp_p = std::make_unique<LinearPowerInterpolator>(config);
    cosmo = std::make_unique<fidcosmo::FlatLCDM>(config);
    rscale_long = config.getDouble("LongScale");
    _z1_pivot = 1.0 + interp_p->z_pivot;
    _sigma_mpc = 0;
    _deltar_mpc = 0;
}


void ArinyoP3DModel::construct() {
    _construcP1D();
    _cacheInterp2D();
    _getCorrFunc2dS();

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
    if (kz == 0)  return 1;
    double kr = kz * _sigma_mpc, kv = kz * _deltar_mpc / 2.0;
    kr *= kr;
    kv = sin(kv) / kv;
    return exp(-kr) * kv * kv;
}


void ArinyoP3DModel::calcVarLss(bool pp_enabled) {
    constexpr int nlnk = 6001;
    constexpr double dlnk = (LNKMAX - LNKMIN) / (nlnk - 1);
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
    constexpr int N = ceil((LNKMAX - LNKMIN) / dlnk);
    auto lnP = std::make_unique<double[]>(N * N);

    /* Large-scale 2D */
    for (int iperp = 0; iperp < N; ++iperp) {
        double kperp = exp(LNKMIN + iperp * dlnk);

        for (int iz = 0; iz < N; ++iz) {
            double kz = exp(LNKMIN + iz * dlnk),
                   k = sqrt(kperp * kperp + kz * kz),
                   k_rL = k * rscale_long;

            k_rL *= -k_rL;
            lnP[iz + N * iperp] = log(evalExplicit(k, kz) + SAFE_ZERO) + k_rL;
        }
    }
    interp2d_pL.interp_2d = std::make_unique<DiscreteInterpolation2D>(
        LNKMIN, dlnk, LNKMIN, dlnk, lnP.get(), N, N);

    /* Small-scale 2D */
    for (int iperp = 0; iperp < N; ++iperp) {
        double kperp = exp(LNKMIN + iperp * dlnk);

        for (int iz = 0; iz < N; ++iz) {
            double kz = exp(LNKMIN + iz * dlnk),
                   k = sqrt(kperp * kperp + kz * kz),
                   k_rL = k * rscale_long;

            k_rL = 1.0 - exp(-k_rL * k_rL);
            lnP[iz + N * iperp] = log(evalExplicit(k, kz) * k_rL + SAFE_ZERO);
        }
    }
    interp2d_pS.interp_2d = std::make_unique<DiscreteInterpolation2D>(
        LNKMIN, dlnk, LNKMIN, dlnk, lnP.get(), N, N);

    /* Large-scale 1Ds */
    for (int iperp = 0; iperp < N; ++iperp) {
        double k = exp(LNKMIN + iperp * dlnk), k_rL = k * rscale_long;
        k_rL *= -k_rL;

        lnP[iperp] = log(evalExplicit(k, 0) + SAFE_ZERO) + k_rL;
    }

    interp2d_pL.interp_x = std::make_unique<DiscreteInterpolation1D>(
        LNKMIN, dlnk, N, lnP.get());

    for (int iz = 0; iz < N; ++iz) {
        double kz = exp(LNKMIN + iz * dlnk), k_rL = kz * rscale_long;
        k_rL *= -k_rL;

        lnP[iz] = log(evalExplicit(kz, kz) + SAFE_ZERO) + k_rL;
    }
    interp2d_pL.interp_y = std::make_unique<DiscreteInterpolation1D>(
        LNKMIN, dlnk, N, lnP.get());


    /* Small-scale 1Ds */
    for (int iperp = 0; iperp < N; ++iperp) {
        double k = exp(LNKMIN + iperp * dlnk), k_rL = k * rscale_long;
        k_rL = 1.0 - exp(-k_rL * k_rL);

        lnP[iperp] = log(evalExplicit(k, 0) * k_rL + SAFE_ZERO);
    }

    interp2d_pS.interp_x = std::make_unique<DiscreteInterpolation1D>(
        LNKMIN, dlnk, N, lnP.get());

    for (int iz = 0; iz < N; ++iz) {
        double k = exp(LNKMIN + iz * dlnk), k_rL = k * rscale_long;
        k_rL = 1.0 - exp(-k_rL * k_rL);

        lnP[iz] = log(evalExplicit(k, k) * k_rL + SAFE_ZERO);
    }
    interp2d_pS.interp_y = std::make_unique<DiscreteInterpolation1D>(
        LNKMIN, dlnk, N, lnP.get());
}


void ArinyoP3DModel::_construcP1D() {
    constexpr int nlnk = 10001;
    constexpr double dlnk = (LNKMAX - LNKMIN) / (nlnk - 1), dlnk2 = 0.02;
    constexpr int nlnk2 = ceil((LNKMAX - LNKMIN) / dlnk2);
    double p1d_integrand[nlnk], p1d[nlnk2];

    for (int i = 0; i < nlnk2; ++i) {
        double kz = exp(LNKMIN + i * dlnk2);

        for (int j = 0; j < nlnk; ++j) {
            double kperp2 = exp(LNKMIN + j * dlnk);
            kperp2 *= kperp2;
            double k = sqrt(kperp2 + kz * kz);
            double k_rL = k * rscale_long;
            k_rL = 1.0 - exp(-k_rL * k_rL);

            p1d_integrand[j] = kperp2 * evalExplicit(k, kz) * k_rL;
        }
        p1d[i] = log(trapz(p1d_integrand, nlnk, dlnk) / (2.0 * MY_PI) + SAFE_ZERO);
    }

    interp_p1d = std::make_unique<DiscreteInterpolation1D>(
        LNKMIN, dlnk2, nlnk2, &p1d[0]);
}


void ArinyoP3DModel::_getCorrFunc2dS() {
    constexpr int Nhankel = 1536;
    Ps2Cf_2D hankel{Nhankel, KMIN, 1 / KMIN};

    auto psarr = std::make_unique<double[]>(Nhankel * Nhankel);
    const double *kperparr = hankel.getKperp(), *kzarr = hankel.getKz();

    for (int iperp = 0; iperp < Nhankel; ++iperp)
        for (int iz = 0; iz < Nhankel; ++iz)
            psarr[iz + Nhankel * iperp] = interp2d_pS.evaluate(kperparr[iperp], kzarr[iz]);

    #ifndef NUSE_LOGR_INTERP
        interp2d_cfS = hankel.transform(psarr.get(), 420, 0, true);
    #else
        interp2d_cfS = hankel.transform(
            psarr.get(), 256, ArinyoP3DModel::MAX_R_FACTOR * rscale_long);
    #endif

    interp1d_cf = interp2d_cfS->get1dSliceX(interp2d_cfS->getY1());
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
    result, lnD;

    lnD = (q_1 * delta2_L) * (
            1 - pow(kz / k_nu, nu_1) * pow(k / k_nu, -nu_0)
    ) - k_kp * k_kp;

    result = plin * (bbeta_lya * bbeta_lya * exp(lnD)
                     + 2.0 * bbeta_lya * bbeta_hcd_kz
                     + bbeta_hcd_kz * bbeta_hcd_kz);

    if ((_sigma_mpc == 0) && (_deltar_mpc == 0))
        return result;

    return result * getSpectroWindow2(kz);
}


void ArinyoP3DModel::write(ioh::Qu3dFile *out) {
    constexpr int nlnk = 502, nlnk2 = nlnk * nlnk;
    constexpr double dlnk = (LNKMAX - LNKMIN) / (nlnk - 2);
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

    for (int iz = 0; iz < nlnk; ++iz)
        pmarr[iz] = evalP1d(karr[iz]);

    out->write(pmarr, nlnk, "PMODEL_1D");
    out->flush();

    constexpr int nr = 512, nr2 = nr * nr;
    double rarr[nr], cfsarr[nr2];

    #ifndef NUSE_LOGR_INTERP
        const double r1 = exp2(interp2d_cfS->getX1()), r2 = 1.0 / r1, dlnr = log(r2/r1) / nr;
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
            #ifndef NUSE_LOGR_INTERP
            cfsarr[iz + nr * iperp] = evalCorrFunc2dS(rarr[iperp] * rarr[iperp], rarr[iz]);
            #else
            cfsarr[iz + nr * iperp] = evalCorrFunc2dS(rarr[iperp], rarr[iz]);
            #endif

    out->write(cfsarr, nr2, "CFMODEL_S_2D");
    out->flush();
}