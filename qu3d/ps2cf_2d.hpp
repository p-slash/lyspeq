#ifndef PS2CF_2D_H
#define PS2CF_2D_H

#include <memory>

#include "mathtools/discrete_interpolation.hpp"
#include "mathtools/mathutils.hpp"
#include "mathtools/fftlog.hpp"


constexpr double MY_2PI = 2.0 * 3.14159265358979323846;


class Ps2Cf_2D {
public:
    Ps2Cf_2D(int nk, double k1, double k2) : N(nk) {
        fht_z = std::make_unique<FFTLog>(N);
        fht_xy = std::make_unique<FFTLog>(N);

        fht_z->construct(-0.5, k1, k2, 0, log(k1 * k2));
        fht_xy->construct(0, k1, k2, 0, log(k1 * k2));
    }

    const double* getKperp() const { return fht_xy->r.get(); }
    const double* getKz() const { return fht_z->r.get(); }

    /* p2d must be in kz-first format, that is first N elements are
       P(kperp[0], kz) and so on. Transformation kperp values must be used.
    */
    std::unique_ptr<DiscreteInterpolation2D> transform(const double *p2d) {
        /* Intermediate array will be transposed */
        interm = std::make_unique<double[]>(N * N);
        result = std::make_unique<double[]>(N * N);

        for (int i = 0; i < N; ++i)
            _fhtZ(p2d + i * N, i);

        const double *kperp = getKperp();

        for (int iz = 0; iz < N; ++iz) {
            for (int j = 0; j < N; ++j)
                fht_xy->field[j] = interm[j + iz * N] * kperp[j];

            fht_xy->transform();

            for (int j = 0; j < N; ++j)
                result[iz + N * j] = fht_xy->field[j] / fht_xy->k[j];
        }

        for (int i = 0; i < N * N; ++i)
            result[i] /= MY_2PI;

        return std::make_unique<DiscreteInterpolation2D>(
            log(fht_z->k[0]), fht_z->getDLn(), log(fht_xy->k[0]), fht_xy->getDLn(),
            result.get(), N, N);
    }

private:
    int N;
    std::unique_ptr<double[]> kperp, rperp, interm, result;
    std::unique_ptr<FFTLog> fht_z, fht_xy;

    void _fhtZ(const double *pkz, int i) {
        std::copy_n(pkz, N, fht_z->field);
        fht_z->transform();

        for (int j = 0; j < N; ++j)
            interm[i + j * N] = fht_z->field[j] / sqrt(MY_2PI * fht_z->k[j]);
    }
};

#endif
