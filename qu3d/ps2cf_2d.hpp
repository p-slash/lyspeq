#ifndef PS2CF_2D_H
#define PS2CF_2D_H

#include <memory>

#include "core/global_numbers.hpp"

#include "mathtools/discrete_interpolation.hpp"
#include "mathtools/interpolation_2d.hpp"
#include "mathtools/mathutils.hpp"
#include "mathtools/fftlog.hpp"
#include "mathtools/smoother.hpp"


class Ps2Cf_2D {
public:
    Ps2Cf_2D(
            int nk, double k1, double k2, int smooth_sigma=0,
            double bz=0, double bperp=0, double lnkr=0
    ) : N(nk) {
        fht_z = std::make_unique<FFTLog>(N);
        fht_xy = std::make_unique<FFTLog>(N);

        sqrt_kz = std::make_unique<double[]>(N);
        isqrt_rz = std::make_unique<double[]>(N);
        interm = std::make_unique<double[]>(N * N);

        if (smooth_sigma > 0)
            smoother = std::make_unique<Smoother>(smooth_sigma);

        fht_z->construct(-0.5, k1, k2, bz, lnkr);
        fht_xy->construct(0, k1, k2, bperp, lnkr);

        for (int i = 0; i < N; ++i) {
            sqrt_kz[i] = sqrt(fht_z->r[i]);
            isqrt_rz[i] = 1.0 / sqrt(fht_z->k[i]);
        }
    }

    const double* getKperp() const { return fht_xy->r.get(); }
    const double* getKz() const { return fht_z->r.get(); }
    const double* getRperp() const { return fht_xy->k.get(); }
    const double* getRz() const { return fht_z->k.get(); }
    double getDlnkPerp() const { return fht_xy->getDLn(); }
    double getDlnkZ() const { return fht_z->getDLn(); }

    /* p2d must be in kz-first format, that is first N elements are
       P(kperp[0], kz) and so on. Transformation kperp values must be used.

        First creates interpolator in ln(rz), ln(rperp2). Then:
       Returns: xi_SS interpolator in rz, rperp if return_log_interp=false
       Returns: xi_SS interpolator in ln(rz), ln(rperp) if return_log_interp=true
    */
    template<class T>
    std::unique_ptr<T> transform(
            const double *p2d, int ltrunc, int rtrunc, double rmax,
            bool return_log_interp=false
    ) {
        int Nres;
        auto result = transformBasic(p2d, ltrunc, rtrunc, Nres);

        const double log2_e = log2(exp(1.0));
        if (return_log_interp)
            return std::make_unique<T>(
                log2(fht_z->k[ltrunc]), log2_e * fht_z->getDLn(),
                log2(fht_xy->k[ltrunc]), log2_e * fht_xy->getDLn(),
                result.get(), Nres, Nres);

        // Evaluting log for all coordinates is too expensive.
        // Convert input rs to log r
        for (int i = 0; i < N; ++i) {
            fht_z->k[i] = log(fht_z->k[i]);
            fht_xy->k[i] = log(fht_xy->k[i]);
        }

        Interpolation2D logr_interp(
            GSL_BICUBIC_INTERPOLATION,
            fht_z->k.get() + ltrunc, fht_xy->k.get() + ltrunc, result.get(),
            Nres, Nres);

        auto lnrlin = std::make_unique<double[]>(Nres);
        double dr = rmax / (Nres - 1);

        lnrlin[0] = std::max(fht_z->k[ltrunc], fht_xy->k[ltrunc]);
        for (int i = 1; i < Nres; ++i)
            lnrlin[i] = log(i * dr);

        for (int i = 0; i < Nres; ++i)
            for (int j = 0; j < Nres; ++j)
                result[j + Nres * i] = logr_interp.evaluate(lnrlin[j], lnrlin[i]);

        return std::make_unique<T>(0, dr, 0, dr, result.get(), Nres, Nres);
    }

    std::unique_ptr<double[]> transformBasic(
            const double *in, int ltrunc, int rtrunc, int &Nres
    ) {
        Nres = N - (ltrunc + rtrunc);
        /* Intermediate array will be transposed */
        auto result = std::make_unique<double[]>(Nres * Nres);

        for (int i = 0; i < N; ++i)
            _fhtZ(in + i * N, i);

        const double *kperp = getKperp();

        for (int iz = 0; iz < Nres; ++iz) {
            for (int j = 0; j < N; ++j)
                fht_xy->field[j] = interm[j + (iz + ltrunc) * N] * kperp[j];

            fht_xy->transform();
            for (int j = 0; j < N; ++j)
                fht_xy->field[j] /= fht_xy->k[j];

            if (smoother)
                smoother->smooth1D(fht_xy->field + ltrunc, Nres, 1, true);

            for (int iperp = 0; iperp < Nres; ++iperp)
                result[iz + Nres * iperp] = fht_xy->field[iperp + ltrunc];
        }

        constexpr double MY_2PI = 2.0 * MY_PI;
        const double NORM = MY_2PI * sqrt(MY_2PI);
        for (int i = 0; i < Nres * Nres; ++i)
            result[i] /= NORM;

        if (smoother)
            smoother->smooth1D(result.get(), Nres, Nres, true);

        return result;
    }
private:
    int N;
    std::unique_ptr<double[]> sqrt_kz, isqrt_rz, interm;
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
