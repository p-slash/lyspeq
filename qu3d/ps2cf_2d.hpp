#ifndef PS2CF_2D_H
#define PS2CF_2D_H

#include <memory>

#include "mathtools/discrete_interpolation.hpp"
#include "mathtools/interpolation_2d.hpp"
#include "mathtools/mathutils.hpp"
#include "mathtools/fftlog.hpp"
#include "mathtools/smoother.hpp"


class Ps2Cf_2D {
public:
    Ps2Cf_2D(int nk, double k1, double k2, int smooth_sigma=1) : N(nk) {
        fht_z = std::make_unique<FFTLog>(N);
        fht_xy = std::make_unique<FFTLog>(N);

        sqrt_kz = std::make_unique<double[]>(N);
        isqrt_rz = std::make_unique<double[]>(N);

        if (smooth_sigma > 0)
            smoother = std::make_unique<Smoother>(smooth_sigma);

        fht_z->construct(-0.5, k1, k2, -0.25, 0);
        fht_xy->construct(0, k1, k2, -0.25, 0);

        for (int i = 0; i < N; ++i) {
            sqrt_kz[i] = sqrt(fht_z->r[i]);
            isqrt_rz[i] = 1.0 / sqrt(fht_z->k[i]);
        }
    }

    const double* getKperp() const { return fht_xy->r.get(); }
    const double* getKz() const { return fht_z->r.get(); }

    /* p2d must be in kz-first format, that is first N elements are
       P(kperp[0], kz) and so on. Transformation kperp values must be used.

        First creates interpolator in ln(rz), ln(rperp). Then:
       Returns: xi_SS interpolator in rz, rperp
    */
    std::unique_ptr<DiscreteInterpolation2D> transform(
            const double *p2d, int truncate, double rmax,
            bool return_log_interp=false
    ) {
        int Nres = N - 2 * truncate;
        /* Intermediate array will be transposed */
        interm = std::make_unique<double[]>(N * N);
        result = std::make_unique<double[]>(Nres * Nres);

        for (int i = 0; i < N; ++i)
            _fhtZ(p2d + i * N, i);

        const double *kperp = getKperp();

        for (int iz = 0; iz < Nres; ++iz) {
            for (int j = 0; j < N; ++j)
                fht_xy->field[j] = interm[j + (iz + truncate) * N] * kperp[j];

            fht_xy->transform();
            for (int j = 0; j < N; ++j)
                fht_xy->field[j] /= fht_xy->k[j];

            if (smoother)
                smoother->smooth1D(fht_xy->field + truncate, Nres, 1, true);

            for (int iperp = 0; iperp < Nres; ++iperp)
                result[iz + Nres * iperp] = fht_xy->field[iperp + truncate];
        }

        constexpr double MY_2PI = 2.0 * 3.14159265358979323846;
        constexpr double NORM = MY_2PI * sqrt(MY_2PI);
        for (int i = 0; i < Nres * Nres; ++i)
            result[i] /= NORM;

        if (smoother)
            smoother->smooth1D(result.get(), Nres, Nres, true);

        if (return_log_interp)
            return std::make_unique<DiscreteInterpolation2D>(
                log(fht_z->k[truncate]), fht_z->getDLn(),
                log(fht_xy->k[truncate]), fht_xy->getDLn(),
                result.get(), Nres, Nres);

        // Evaluting log for all coordinates is too expensive.
        // Convert input rs to log r
        for (int i = 0; i < N; ++i) {
            fht_z->k[i] = log(fht_z->k[i]);
            fht_xy->k[i] = log(fht_xy->k[i]);
        }

        Interpolation2D logr_interp(
            GSL_BICUBIC_INTERPOLATION,
            fht_z->k.get() + truncate, fht_xy->k.get() + truncate, result.get(),
            Nres, Nres);

        auto lnrlin = std::make_unique<double[]>(Nres);
        double dr = rmax / Nres;

        lnrlin[0] = std::max(fht_z->k[truncate], fht_xy->k[truncate]);
        for (int i = 1; i < Nres; ++i)
            lnrlin[i] = log(i * dr);

        for (int i = 0; i < Nres; ++i)
            for (int j = 0; j < Nres; ++j)
                result[j + Nres * i] = logr_interp.evaluate(lnrlin[j], lnrlin[i]);

        return std::make_unique<DiscreteInterpolation2D>(
            0, dr, 0, dr, result.get(), Nres, Nres);
    }

private:
    int N;
    std::unique_ptr<double[]> sqrt_kz, isqrt_rz, interm, result;
    std::unique_ptr<FFTLog> fht_z, fht_xy;
    std::unique_ptr<Smoother> smoother;

    void _fhtZ(const double *pkz, int i) {
        std::copy_n(pkz, N, fht_z->field);
        for (int j = 0; j < N; ++j)
            fht_z->field[j] = pkz[j] * sqrt_kz[j];

        fht_z->transform();

        for (int j = 0; j < N; ++j)
            interm[i + j * N] = fht_z->field[j] * isqrt_rz[j];
    }
};

#endif
