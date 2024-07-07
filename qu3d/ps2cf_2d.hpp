#ifndef PS2CF_2D_H
#define PS2CF_2D_H

#include <memory>

#include <fftw3.h>
#include <gsl_sf_bessel.h> // gsl_sf_bessel_J0

#include "mathtools/discrete_interpolation.hpp"
#include "mathtools/mathutils.hpp"
#include "mathtools/real_field.hpp"


class Ps2Cf_2D {
public:
    // Ps2Cf_2D(int nJzeros=100) : numJzeros(nJzeros) { _setJzeros(); }
    Ps2Cf_2D();

    /* p2d is in kz-first format, that is first nkz elements are
       P(kperp=0, kz) and so on.
    */
    std::unique_ptr<DiscreteInterpolation2D> transform(
            const double *p2d, int nkperp, int nkz, double dk
    ) {
        constexpr double MY_PI = 3.14159265359, MY_2PI = 2.0 * MY_PI;
        const int nLz = 2 * nkz - 2;
        const double rmax = MY_2PI / dk, dz = MY_PI / (nkz * dk);

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
        tmpJ = std::make_unique<double[]>(nkperp);

        for (int ikperp = 0; ikperp < nkperp; ++ikperp)
            _fftZ(rf, p2d + ikperp * nkz, ikperp, nkperp);

        for (int iz = 0; iz < nkz; ++iz)
            for (int ikperp = 0; ikperp < nkperp; ++ikperp)
                interm[ikperp + iz * nkperp] *= ikperp * dk;

        /* rperp = 0 */
        for (int iz = 0; iz < nkz; ++iz)
            result[iz] = trapz(interm.get() + iz * nkperp, nkperp, dk);

        for (int irperp = 1; irperp < nkz; ++irperp) {
            for (int ikperp = 0; ikperp < nkperp; ++ikperp)
                tmpJ[ikperp] = gsl_sf_bessel_J0(rgrid[iperp] * dk * ikperp);

            for (int iz = 0; iz < nkz; ++iz) {
                for (int ikperp = 0; ikperp < nkperp; ++ikperp)
                    row[ikperp] = tmpJ[ikperp] * interm[ikperp + iz * nkperp];

                result[iz + iperp * nkz] = trapz(row.get(), nkperp, dk);
            }
        }

        for (int i = 0; i < nkz * nkz; ++i)
            result[i] /= MY_2PI;

        return std::make_unique<DiscreteInterpolation2D>(
            0, dz, 0, dz, result.get(), nkz, nkz);
    }

private:
    std::unique_ptr<double[]> interm, result, row, tmpJ;

    // void _setJzeros() {
    //     zerosJ0 = std::make_unique<double[]>(numJzeros);
    //     for (int i = 0; i < numJzeros; ++i)
    //         zerosJ0[i] = gsl_sf_bessel_zero_J0(i);
    // }

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
}

#endif
