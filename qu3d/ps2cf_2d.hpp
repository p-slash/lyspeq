#ifndef PS2CF_2D_H
#define PS2CF_2D_H

#include <memory>

#include "mathtools/discrete_interpolation.hpp"
#include "mathtools/mathutils.hpp"
#include "mathtools/fftlog.hpp"
#include "mathtools/smoother.hpp"


class Ps2Cf_2D {
public:
    Ps2Cf_2D(int nk, double k1, double k2, int smooth_sigma=1) : N(nk) {
        fht_z = std::make_unique<FFTLog>(N);
        fht_xy = std::make_unique<FFTLog>(N);
        if (smooth_sigma > 0)
            smoother = std::make_unique<Smoother>(smooth_sigma);

        fht_z->construct(-0.5, k1, k2, -0.25, 0);
        fht_xy->construct(0, k1, k2, 0, 0);
    }

    const double* getKperp() const { return fht_xy->r.get(); }
    const double* getKz() const { return fht_z->r.get(); }

    /* p2d must be in kz-first format, that is first N elements are
       P(kperp[0], kz) and so on. Transformation kperp values must be used.

       Returns: xi_SS interpolator in ln(rz), ln(rperp2)
    */
    std::unique_ptr<DiscreteInterpolation2D> transform(
            const double *p2d, int truncate
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

        return std::make_unique<DiscreteInterpolation2D>(
            log(fht_z->k[truncate]), fht_z->getDLn(),
            2.0 * log(fht_xy->k[truncate]), 2.0 * fht_xy->getDLn(),
            result.get(), Nres, Nres);
    }

private:
    int N;
    std::unique_ptr<double[]> kperp, rperp, interm, result;
    std::unique_ptr<FFTLog> fht_z, fht_xy;
    std::unique_ptr<Smoother> smoother;

    void _fhtZ(const double *pkz, int i) {
        std::copy_n(pkz, N, fht_z->field);
        fht_z->transform();

        for (int j = 0; j < N; ++j)
            interm[i + j * N] = fht_z->field[j] / sqrt(fht_z->k[j]);
    }
};

#endif
