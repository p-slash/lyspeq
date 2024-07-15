#include "cosmology_3d.hpp"
#include <gsl/gsl_integration.h>

#include "core/global_numbers.hpp"
#include "io/io_helper_functions.hpp"
#include "mathtools/interpolation.hpp"
#include "mathtools/mathutils.hpp"
#include "qu3d/ps2cf_2d.hpp"


constexpr double TWO_PI2 = 2 * MY_PI * MY_PI;
constexpr double KMIN = 1E-6, KMAX = 5.0,
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
    gsl_function F;
    F.function = _integrand_LinearGrowth;
    F.params = &cosmo_params;

    gsl_integration_workspace *w =
        gsl_integration_workspace_alloc(WORKSPACE_SIZE);

    if (w == NULL)
        throw std::bad_alloc();

    for (int i = 0; i < nz; ++i) {
        double error = 0;

        gsl_integration_qagiu(
            &F, z1arr[i],
            ABS_ERROR, REL_ERROR,
            WORKSPACE_SIZE, w,
            linD + i, &error);

        linD[i] *= getHubble(z1arr[i]);
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
    const int nz = 301;
    const double dz = 0.02;
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
        temparr[i] = _inv_comoving_interp.evaluate(
            chi0 + i * dchi);
    interp_invcomov_dist = std::make_unique<DiscreteCubicInterpolation1D>(
        temparr[0], dchi, nz, temparr);

    _integrateLinearGrowth(nz, z1arr, temparr);
    linear_growth_unnorm = std::make_unique<DiscreteCubicInterpolation1D>(
        z1arr[0], dz, nz, temparr);
}


std::unique_ptr<double[]> LinearPowerInterpolator::_appendLinearExtrapolation(
        double lnk1, double lnk2, double dlnk, int N,
        const std::vector<double> &lnP, double &newlnk1
) {
    double m1 = lnP[1] - lnP[0],
           m2 = lnP[N - 1] - lnP[N - 2];

    int n1 = std::max(0, int((lnk1 - LNKMIN) / dlnk)),
        n2 = std::max(0, int((LNKMAX - lnk2) / dlnk));

    newlnk1 = lnk1 - n1 * dlnk;
    auto result = std::make_unique<double[]>(n1 + N + n2);

    for (int i = 0; i < n1; ++i)
        result[i] = lnP[0] + m1 * (i - n1);

    std::copy(lnP.begin(), lnP.end(), result.get() + n1);

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
        lnk1, dlnk, N, appendLnp.get());
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
    interp_p = std::make_unique<LinearPowerInterpolator>(config);
    cosmo = std::make_unique<fidcosmo::FlatLCDM>(config);
    rscale_long = config.getDouble("LongScale");
    _z1_pivot = 1.0 + interp_p->z_pivot;

    _construcP1D();
    _calcVarLss();
    _cacheInterp2D();
    _getCorrFunc2dS();

    _D_pivot = cosmo->getUnnormLinearGrowth(_z1_pivot);
    const int nz = 200;
    const double dz = 0.02;
    double growth[nz];

    for (int i = 0; i < nz; ++i) {
        double z1 = 2.0 + dz * i;
        growth[i] = cosmo->getUnnormLinearGrowth(z1) / _D_pivot
                    * pow(z1 / _z1_pivot, alpha_F);
    }
    interp_growth = std::make_unique<DiscreteCubicInterpolation1D>(
        2.0, dz, nz, &growth[0]);
}


void ArinyoP3DModel::_calcVarLss() {
    constexpr int nlnk = 5001;
    constexpr double dlnk = (LNKMAX - LNKMIN) / (nlnk - 1);
    double powers_kz[nlnk], powers_kperp[nlnk];

    for (int i = 0; i < nlnk; ++i) {
        double kperp2 = exp(LNKMIN + i * dlnk);
        kperp2 *= kperp2;
        for (int j = 0; j < nlnk; ++j) {
            double kz = exp(LNKMIN + j * dlnk),
                   k = sqrt(kperp2 + kz * kz);
            double k_rL = k * rscale_long;
            k_rL *= -k_rL;
            powers_kz[j] = kz * evalExplicit(k, kz) * exp(k_rL);
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
            lnP[iz + N * iperp] = log(evalExplicit(k, kz)) + k_rL;
        }
    }
    interp2d_pL = std::make_unique<DiscreteInterpolation2D>(
        LNKMIN, dlnk, LNKMIN, dlnk, lnP.get(), N, N);

    /* Small-scale 2D */
    for (int iperp = 0; iperp < N; ++iperp) {
        double kperp = exp(LNKMIN + iperp * dlnk);

        for (int iz = 0; iz < N; ++iz) {
            double kz = exp(LNKMIN + iz * dlnk),
                   k = sqrt(kperp * kperp + kz * kz),
                   k_rL = k * rscale_long;

            k_rL = 1.0 - exp(-k_rL * k_rL);
            lnP[iz + N * iperp] = log(evalExplicit(k, kz) * k_rL);
        }
    }
    interp2d_pS = std::make_unique<DiscreteInterpolation2D>(
        LNKMIN, dlnk, LNKMIN, dlnk, lnP.get(), N, N);

    /* Large-scale 1Ds */
    for (int iperp = 0; iperp < N; ++iperp) {
        double k = exp(LNKMIN + iperp * dlnk), k_rL = k * rscale_long;
        k_rL *= -k_rL;

        lnP[iperp] = log(evalExplicit(k, 0)) + k_rL;
    }

    interp_kp_pL = std::make_unique<DiscreteInterpolation1D>(
        LNKMIN, dlnk, N, lnP.get());

    for (int iz = 0; iz < N; ++iz) {
        double kz = exp(LNKMIN + iz * dlnk), k_rL = kz * rscale_long;
        k_rL *= -k_rL;

        lnP[iz] = log(evalExplicit(kz, kz)) + k_rL;
    }
    interp_kz_pL = std::make_unique<DiscreteInterpolation1D>(
        LNKMIN, dlnk, N, lnP.get());


    /* Small-scale 1Ds */
    for (int iperp = 0; iperp < N; ++iperp) {
        double k = exp(LNKMIN + iperp * dlnk), k_rL = k * rscale_long;
        k_rL = 1.0 - exp(-k_rL * k_rL);

        lnP[iperp] = log(evalExplicit(k, 0) * k_rL);
    }

    interp_kp_pS = std::make_unique<DiscreteInterpolation1D>(
        LNKMIN, dlnk, N, lnP.get());

    for (int iz = 0; iz < N; ++iz) {
        double k = exp(LNKMIN + iz * dlnk), k_rL = k * rscale_long;
        k_rL = 1.0 - exp(-k_rL * k_rL);

        lnP[iz] = log(evalExplicit(k, k) * k_rL);
    }
    interp_kz_pS = std::make_unique<DiscreteInterpolation1D>(
        LNKMIN, dlnk, N, lnP.get());
}


void ArinyoP3DModel::_construcP1D() {
    constexpr int nlnk = 10001;
    constexpr double dlnk = (LNKMAX - LNKMIN) / (nlnk - 1), dlnk2 = 0.02;
    constexpr int nlnk2 = ceil((LNKMAX - LNKMIN) / dlnk);
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
        p1d[i] = log(trapz(p1d_integrand, nlnk, dlnk) / (2.0 * MY_PI));
    }

    interp_p1d = std::make_unique<DiscreteInterpolation1D>(
        LNKMIN, dlnk2, nlnk2, &p1d[0]);
}


void ArinyoP3DModel::_getCorrFunc2dS() {
    double dk = 2 * MY_PI / (50.0 * rscale_long);

    double nk_d = KMAX / dk, log2nk = log2(nk_d);
    int log2nk_ceil = ceil(log2nk), nk = 0;

    // If the nearest power of two is >= 2^10
    if (log2nk_ceil > 9) {
        double rem = log2nk - int(log2nk);
        constexpr double y = log(2.0) / log(4.0 / 3.0);
        int m = floor((1.0 - rem) * y);
        nk = 1 << (log2nk_ceil - 2 * m);
        for (int i = 0; i < m; ++i)
            nk *= 3;
    }
    else {
        nk = 1 << log2nk_ceil;
    }

    int nk2 = nk * nk;

    dk = KMAX / (nk - 1);
    Ps2Cf_2D hankel{nk, KMAX};

    auto kzarr = std::make_unique<double[]>(nk),
         psarr = std::make_unique<double[]>(nk2);
    const double *kperparr = hankel.getKperp();

    for (int i = 0; i < nk; ++i)
        kzarr[i] = i * dk;

    for (int iperp = 0; iperp < nk; ++iperp)
        for (int iz = 0; iz < nk; ++iz)
            psarr[iz + nk * iperp] = evaluateSS(kperparr[iperp], kzarr[iz]);

    interp2d_cfS = hankel.transform(psarr.get(), nk, dk);
}


double ArinyoP3DModel::evalExplicit(double k, double kz) {
    if (k == 0)
        return 0;

    double
    plin = interp_p->evaluate(k),
    delta2_L = plin * k * k * k / TWO_PI2,
    k_kp = k / k_p,
    mu = kz / k,
    result, lnD;

    result = b_F * (1 + beta_F * mu * mu);
    result *= result;

    lnD = (q_1 * delta2_L) * (
            1 - pow(kz / k_nu, nu_1) * pow(k / k_nu, -nu_0)
    ) - k_kp * k_kp;

    return result * plin * exp(lnD);
}


void ArinyoP3DModel::write(ioh::Qu3dFile *out) {
    constexpr int nlnk = 502, nlnk2 = nlnk * nlnk;
    constexpr double dlnk = (LNKMAX - LNKMIN) / (nlnk - 2);
    double karr[nlnk], pmarr[nlnk2];

    karr[0] = 0;
    for (int i = 1; i < nlnk; ++i)
        karr[i] = exp(LNKMIN + (i - 1) * dlnk);

    out->write(karr, nlnk, "KMODEL");

    for (int iperp = 0; iperp < nlnk; ++iperp)
        for (int iz = 0; iz < nlnk; ++iz)
            pmarr[iz + nlnk * iperp] = evaluate(karr[iperp], karr[iz]);

    out->write(pmarr, nlnk2, "PMODEL_L");
    out->flush();

    for (int iperp = 0; iperp < nlnk; ++iperp)
        for (int iz = 0; iz < nlnk; ++iz)
            pmarr[iz + nlnk * iperp] = evaluateSS(karr[iperp], karr[iz]);

    out->write(pmarr, nlnk2, "PMODEL_S");
    out->flush();

    for (int iz = 0; iz < nlnk; ++iz)
        pmarr[iz] = evalP1d(karr[iz]);

    out->write(pmarr, nlnk, "PMODEL_1D");
    out->flush();

    constexpr int nr = 500, nr2 = nr * nr;
    const double r2 = 10.0 * rscale_long, dr = r2 / nr;
    double rarr[nr], cfsarr[nr2];

    for (int i = 0; i < nr; ++i)
        rarr[i] = i * dr;

    out->write(rarr, nr, "RMODEL");

    for (int iperp = 0; iperp < nr; ++iperp)
        for (int iz = 0; iz < nr; ++iz)
            cfsarr[iz + nr * iperp] = evalCorrFunc2dS(rarr[iperp], rarr[iz]);

    out->write(cfsarr, nr2, "CFMODEL_S_2D");
    out->flush();
}