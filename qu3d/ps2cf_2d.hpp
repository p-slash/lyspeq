#ifndef PS2CF_2D_H
#define PS2CF_2D_H

#include <memory>

#include <fftw3.h>
#include <gsl/gsl_dht.h>

#include "mathtools/discrete_interpolation.hpp"
#include "mathtools/mathutils.hpp"
#include "mathtools/real_field.hpp"


class Ps2Cf_2D {
public:
    Ps2Cf_2D(int nk, double Lk) : nkperp(nk), L(Lk), dht(nullptr) {
        dht = gsl_dht_alloc(nk);
        if (dht == nullptr)
            throw "ERROR in Ps2Cf_2D::gsl_dht_alloc";
        gsl_dht_init(dht, 0, L);
        _setJzeros();
    }
    Ps2Cf_2D() { gsl_dht_free(dht); }

    const double* getKperp() const { return kperp.get(); }

    /* p2d is in kz-first format, that is first nkz elements are
       P(kperp[0], kz) and so on. Transformation kperp values must be used.
    */
    std::unique_ptr<DiscreteInterpolation2D> transform(
            const double *p2d, int nkz, double dkz
    ) {
        constexpr double MY_PI = 3.14159265359, MY_2PI = 2.0 * MY_PI;
        const int nLz = 2 * nkz - 2;
        const double dz = MY_PI / (nkz * dkz);

        RealField rf(nLz, dz);
        if (rf.size_k() != nkz)
            throw std::runtime_error(
                "Error in RealField size in Ps2Cf_2D::transform");

        /* Use the first nkz points */
        const double *rgrid = rf.x.data();

        /* Intermediate array will be transposed */
        interm = std::make_unique<double[]>(nkperp * nkz);
        result = std::make_unique<double[]>(nkz * nkz);
        row = std::make_unique<double[]>(nkperp);

        for (int ikperp = 0; ikperp < nkperp; ++ikperp)
            _fftZ(rf, p2d + ikperp * nkz, ikperp, nkperp);

        Interpolation _xi1d_interp(
            GSL_CUBIC_INTERPOLATION, rperp.get(), row.get(), nkperp);

        for (int iz = 0; iz < nkz; ++iz) {
            std::fill_n(row.get(), nkperp, 0);
            gsl_dht_apply(dht, interm.get() + iz * nkperp, row.get());
            _xi1d_interp.reset(row.get());

            for (int j = 0; j < nkz; ++j)
                result[iz + nkz * j] = _xi1d_interp.evaluate(rgrid[j]);
        }

        for (int i = 0; i < nkz * nkz; ++i)
            result[i] /= MY_2PI;

        return std::make_unique<DiscreteInterpolation2D>(
            0, dz, 0, dz, result.get(), nkz, nkz);
    }

private:
    int nkperp;  double L;
    std::unique_ptr<double[]> kperp, rperp, interm, result, row;
    gsl_dht *dht;

    void _setJzeros() {
        kperp = std::make_unique<double[]>(nkperp);
        rperp = std::make_unique<double[]>(nkperp);

        for (int i = 0; i < nkperp; ++i) {
            kperp[i] = gsl_dht_x_sample(dht, i);
            rperp[i] = gsl_dht_k_sample(dht, i);
        }

        /* GSL zeros start from 1
        for (int i = 0; i < nkperp; ++i)
            zerosJ0[i] = gsl_sf_bessel_zero_J0(i + 1);

         for (int i = 0; i < nkperp; ++i) {
            kperp[i] = L * (zerosJ0[i] / zerosJ0[nkperp - 1]);
            rperp[i] = zerosJ0[i] / L; */
    }

    void _fftZ(RealField &rf, const double *pkz, int ikperp, int nkperp) {
        const int nkz = rf.size_k();
        rf.zero_field_k();
        for (int i = 0; i < nkz; ++i)
            rf.field_k[i] = pkz[i];
        rf.fftK2X();

        // cblas_dcopy(nkz, rf.field_x.data(), 1, interm.get() + ikperp, nkz);
        for (int i = 0; i < nkz; ++i)
            interm[ikperp + i * nkperp] = rf.field_x[i];
    }

    /* void _old_trapzcode () {
        constexpr double MY_PI = 3.14159265359, MY_2PI = 2.0 * MY_PI;
        const int nLz = 2 * nkz - 2;
        const double dz = MY_PI / (nkz * dkz);

        RealField rf(nLz, dz);
        if (rf.size_k() != nkz)
            throw std::runtime_error(
                "Error in RealField size in Ps2Cf_2D::transform");

        // Use the first nkz points
        const double *rz = rf.x.data();
        for (int iz = 0; iz < nkz; ++iz)
            for (int ikperp = 0; ikperp < nkperp; ++ikperp)
                interm[ikperp + iz * nkperp] *= ikperp * dk;

        // rperp = 0
        for (int iz = 0; iz < nkz; ++iz)
            result[iz] = trapz(interm.get() + iz * nkperp, nkperp, dk);

        for (int irperp = 1; irperp < nkz; ++irperp) {
            for (int ikperp = 0; ikperp < nkperp; ++ikperp)
                tmpJ[ikperp] = gsl_sf_bessel_J0(rgrid[irperp] * dk * ikperp);

            for (int iz = 0; iz < nkz; ++iz) {
                for (int ikperp = 0; ikperp < nkperp; ++ikperp)
                    row[ikperp] = tmpJ[ikperp] * interm[ikperp + iz * nkperp];

                result[iz + irperp * nkz] = trapz(row.get(), nkperp, dk);
            }
        }

        for (int i = 0; i < nkz * nkz; ++i)
            result[i] /= MY_2PI;
    } */
};

#endif
