#ifndef REALFIELD_3D_H
#define REALFIELD_3D_H

#include <complex>
#include <vector>

#include <fftw3.h>

#include "core/omp_manager.hpp"

#if defined(ENABLE_OMP)
namespace myomp {
    inline void init_fftw() {
        fftw_init_threads();
        fftw_plan_with_nthreads(omp_get_max_threads());
    }

    inline void clean_fftw() { fftw_cleanup_threads(); }
}
#else
namespace myomp {
    inline void init_fftw() {};
    inline void clean_fftw() {};
}
#endif


/* In-place 3D FFT */
class RealField3D {
    fftw_plan p_x2k;
    fftw_plan p_k2x;

public:
    size_t size_complex, size_real;
    int ngrid[3], ngrid_kz, ngrid_z, ngrid_xy;
    double length[3], dx[3], k_fund[3], z0, cellvol, totalvol;
    std::vector<std::complex<double>> field_k;
    double *field_x;

    RealField3D();
     /* Copy constructor. Copy needs to call construct! */
    RealField3D(const RealField3D &rhs);
    // RealField3D(RealField3D &&rhs) = delete;

    void construct();

    ~RealField3D() {
        fftw_destroy_plan(p_x2k);
        fftw_destroy_plan(p_k2x);
    };

    void zero_field_k() { std::fill(field_k.begin(), field_k.end(), 0); }
    void rawFFTX2K() { fftw_execute(p_x2k); }
    void fftX2K();
    void fftK2X();
    double dot(const RealField3D &other);

    size_t getIndex(int nx, int ny, int nz) {
        int n[] = {nx, ny, nz};
        for (int axis = 0; axis < 3; ++axis) {
            if (n[axis] >= ngrid[axis])
                n[axis] -= ngrid[axis];
            if (n[axis] < 0)
                n[axis] += ngrid[axis];
        }

        return n[2] + ngrid_z * (n[1] + ngrid[1] * n[0]);
    }

    size_t getNgpIndex(double coord[3]) {
        int n[3];

        coord[2] -= z0;
        for (int axis = 0; axis < 3; ++axis)
            n[axis] = round(coord[axis] / dx[axis]);

        return getIndex(n[0], n[1], n[2]);
    }

    inline size_t getCorrectIndexX(size_t j) {
        return j + (j / ngrid[2]) * (ngrid[2] - ngrid_z);
    }

    void getKFromIndex(size_t i, double k[3]) {
        int kn[3];

        size_t iperp = i / ngrid_kz;
        kn[2] = i % ngrid_kz;
        kn[0] = iperp / ngrid[1];
        kn[1] = iperp % ngrid[1];

        if (kn[0] > ngrid[0] / 2)
            kn[0] -= ngrid[0];

        if (kn[1] > ngrid[1] / 2)
            kn[1] -= ngrid[1];

        for (int axis = 0; axis < 3; ++axis)
            k[axis] = k_fund[axis] * kn[axis];
    }

    void getK2KzFromIndex(size_t i, double &k2, double &kz) {
        double ks[3];
        getKFromIndex(i, ks);
        kz = ks[2];

        k2 = 0;
        for (int axis = 0; axis < 3; ++axis)
            k2 += ks[axis] * ks[axis];
    }

    void getKperpFromIperp(size_t iperp, double &kperp) {
        int kn[2];
        kn[0] = iperp / ngrid[1];
        kn[1] = iperp % ngrid[1];

        kperp = 0;
        for (int axis = 0; axis < 2; ++axis) {
            if (kn[axis] > ngrid[axis] / 2)
                kn[axis] -= ngrid[axis];

            double t = k_fund[axis] * kn[axis];
            kperp += t * t;
        }

        kperp = sqrt(kperp);
    }

    void getKperpKzFromIndex(size_t i, double &kperp, double &kz) {
        double ks[3];
        getKFromIndex(i, ks);
        kz = ks[2];

        kperp = 0;
        for (int axis = 0; axis < 2; ++axis)
            kperp += ks[axis] * ks[axis];

        kperp = sqrt(kperp);
    }

    double interpolate(double coord[3]);
    void reverseInterpolateCIC(double coord[3], double val);
    void reverseInterpolateNGP(double coord[3], double val) {
        field_x[getNgpIndex(coord)] += val;
    }
};

#endif
