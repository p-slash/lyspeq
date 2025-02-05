#include "mathtools/multipole_interpolation.hpp"
#include "mathtools/interpolation.hpp"


constexpr double SAFE_ZERO = 1E-300;

DiscreteLogLogInterpolation2D<DiscreteCubicInterpolation1D, DiscreteBicubicSpline>
MultipoleInterpolation::toDiscreteLogLogInterpolation2D(double x1, double dx, int N) {
    DiscreteLogLogInterpolation2D<DiscreteCubicInterpolation1D, DiscreteBicubicSpline>
        output;
    auto lnP_T = std::make_unique<double[]>(N * N);

    for (int iperp = 0; iperp < N; ++iperp) {
        double kperp = exp(x1 + iperp * dx);

        for (int iz = 0; iz < N; ++iz) {
            double kz = exp(x1 + iz * dx),
                   k = sqrt(kperp * kperp + kz * kz),
                   mu = kz / k;
            k = log(k);
            lnP_T[iz + N * iperp] = log(evaluate(k, mu) + SAFE_ZERO);
        }
    }

    output.setInterp2D(x1, dx, x1, dx, lnP_T.get(), N, N);

    for (int i = 0; i < N; ++i) {
        double k = x1 + i * dx;

        lnP_T[i] = log(evaluate(k, 0) + SAFE_ZERO);
        lnP_T[i + N] = log(evaluate(k, 1) + SAFE_ZERO);
    }

    output.setInterpX(x1, dx, N, lnP_T.get());
    output.setInterpY(x1, dx, N, lnP_T.get() + N);

    return output;
}
/*
MultipoleInterpolation MultipoleInterpolation::fromLog2BicubicSpline(
            const DiscreteBicubicSpline* spl, int num_ls
) {
    return;
    constexpr int nmu = 21, nmu_2 = 501;
    constexpr double mu0 = 1e-2, dmu = (1.0 - 2 * mu0) / (nmu - 1),
                     dmu_2 = 1.0 / (nmu_2 - 1);
    double integrand[nmu], these_mus[nmu], integrand_2[nmu_2];

    double x1 = std::max(spl->getX1(), spl->getY1()),
           x2 = std::min(spl->getX2(), spl->getY2()),
           dx = std::max(spl->getDx(), spl->getDy());

    x1 += 10 * dx;  x2 -= 10 * dx;
    int nx = (x2 - x1) / dx + 1;
    dx = (x2 - x1) / (nx - 1);

    MultipoleInterpolation output(num_ls);
    auto x = std::make_unique<double[]>(nx),
         pell = std::make_unique<double[]>(nx);

    for (int i = 0; i < nx; ++i)
        x[i] = exp2(x1 + i * dx);

    std::vector<std::pair<double, double>> mu_integrad_pair;
    mu_integrad_pair.reserve(nmu);

    for (int l = 0; l < num_ls; ++l) {
        for (int i = 0; i < nx; ++i) {
            mu_integrad_pair.clear();

            for (int j = 0; j < nmu; ++j) {
                double mu = mu0 + j * dmu;
                double xz = log2(x[i] * mu),
                       xperp = log2(x[i] * sqrt(1.0 - mu * mu));
                spl->clamp(xz, xperp);

                double xx = exp2(xz) / sqrt(exp2(2 * xz) + exp2(2 * xperp));
                assert( (0 <= xx) && (xx<=1));
                double yy = spl->evaluateHermite2(xz, xperp);
                assert(!std::isnan(yy));
                mu_integrad_pair.push_back(std::make_pair(xx, yy));
            }

            std::sort(mu_integrad_pair.begin(), mu_integrad_pair.end());

            for (int j = 0; j < nmu; ++j) {
                these_mus[j] = mu_integrad_pair[j].first;
                integrand[j] = mu_integrad_pair[j].second;
                if (j > 0) {
                    assert(these_mus[j] > these_mus[j-1]);
                }
            }

            Interpolation temp(GSL_LINEAR_INTERPOLATION, these_mus, integrand, nmu);

            for (int j = 0; j < nmu_2; ++j) {
                integrand_2[j] = temp.evaluate(j * dmu_2);
                if (std::isnan(integrand_2[j])) {
                    fprintf(stderr, "mu: %.2e  x: %.2e || ", j * dmu_2, x[i]);
                }
            }

            pell[i] = trapz(integrand_2, nmu_2, dmu_2) * (4 * l + 1);
        }

        output.setInterpEll(l, x1, dx, nx, pell.get());
    }

    return output;
};

// static bool _is_cached = false;
// static std::unique_ptr<DiscreteCubicInterpolation1D> _legendre[MAX_NUM_L];
// static _cache() {
//     constexpr int Nmu = 101;
//     constexpr double dmu = 1.0 / (Nmu - 1);

//     auto mu = std::make_unique<double[]>(Nmu);
//     auto leg = std::make_unique<double[]>(Nmu);
//     for (int i = 0; i < Nmu; ++i)
//         mu[i] = i * dmu;

//     for (int ell = 0; ell < MAX_NUM_L; ++ell) {
//         for (int i = 0; i < Nmu; ++i)
//             leg[i] = legendre(2 * ell, mu[i]);

//         _legendre[ell] = std::make_unique<DiscreteCubicInterpolation1D>(
//             0, dmu, dmu, leg.get());
//     }

//     _is_cached = true;
// }
*/