#include <algorithm>
#include <stdexcept>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_errno.h>

#include "mathtools/fftlog.hpp"

constexpr double MY_PI = 3.14159265358979323846;


void throwGslError(int gsl_status) {
    std::string err_msg = "ERROR in FFTLog: " + std::string(gsl_strerror(gsl_status));
    throw std::runtime_error(err_msg);
}


double _calcLnKcRc(double mu, double q, double dlnr, double target=0) {
    int gsl_status = 0;
    gsl_sf_result lng, argp, argm;
    double pi_dlnr = MY_PI / dlnr;

    gsl_status = gsl_sf_lngamma_complex_e(
        (mu + 1.0 + q) / 2.0, pi_dlnr / 2.0, &lng, &argp);
    if (gsl_status)
        throwGslError(gsl_status);

    gsl_status = gsl_sf_lngamma_complex_e(
        (mu + 1.0 - q) / 2.0, -pi_dlnr / 2.0, &lng, &argm);
    if (gsl_status)
        throwGslError(gsl_status);

    double l1 = (log(2.0) - target) / dlnr + (argp.val - argm.val) / MY_PI;
    l1 -= round(l1);
    return target + l1 * dlnr;
}


void _calcUm(
        double mu, double L, double q, double lnkcrc,
        std::vector<std::complex<double>> &um
) {
    double pi_L = MY_PI / L, lnmag = 0, theta = 0;
    int M = um.size();
    for (int m = 0; m < M; ++m) {
        gsl_sf_result lngp, lngm, argp, argm;

        gsl_sf_lngamma_complex_e(
            (mu + 1.0 + q) / 2.0, m * pi_L, &lngp, &argp);

        gsl_sf_lngamma_complex_e(
            (mu + 1.0 - q) / 2.0, -m * pi_L, &lngm, &argm);

        lnmag = q * log(2.0) + lngp.val - lngm.val;
        theta = 2.0 * m * pi_L * (log(2.0) - lnkcrc) + (argp.val - argm.val);

        um[m] = std::polar(exp(lnmag), theta);
    }

    um.back() = std::real(um.back());
}


void FFTLog::construct(
        double _mu, double r1, double r2, double _q, double lnkr
) {
    mu = _mu;
    q = _q;
    dlnr = log(r2 / r1) / (N - 1);
    L = N * dlnr;

    for (int i = 0; i < N; ++i)
        r[i] = exp(log(r1) + i * dlnr);

    lnkcrc = _calcLnKcRc(mu, q, dlnr, lnkr);

    double offset = exp(lnkcrc);
    for (int i = 0; i < N; ++i)
        k[i] = offset / r[N - i - 1];

    _calcUm(mu, L, q, lnkcrc, _u_m);
}


void FFTLog::transform() {
    if (q != 0)
        for (int i = 0; i < N; ++i)
            field[i] /= std::pow(r[i], q);

    fftw_execute(p_x2k);
    for (size_t i = 0; i < _field.size(); ++i)
        _field[i] *= _u_m[i];
    fftw_execute(p_k2x);

    for (int i = 0; i < N; ++i)
        field[i] /= N;

    // std::rotate(field, field + N / 2, field + N);
    std::reverse(field, field + N);

    if (q != 0)
        for (int i = 0; i < N; ++i)
            field[i] /= std::pow(k[i], q);
}
