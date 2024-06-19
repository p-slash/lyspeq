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
    int ngrid_z, ngrid_xy, ngrid_kz;
    double k_fund[3], totalvol;

    fftw_plan p_x2k;
    fftw_plan p_k2x;

public:
    size_t size_complex, size_real;
    int ngrid[3];
    double length[3], dx[3], z0, cellvol;
    std::vector<std::complex<double>> field_k;
    double *field_x;

    RealField3D();
    void construct();

    RealField3D(RealField3D &&rhs) = delete;
    RealField3D(const RealField3D &rhs) = delete;
    ~RealField3D() {
        fftw_destroy_plan(p_x2k);
        fftw_destroy_plan(p_k2x);
    };

    void zero_field_k() { std::fill(field_k.begin(), field_k.end(), 0); }
    void rawFFTX2K() { fftw_execute(p_x2k); }
    void fftX2K();
    void fftK2X();

    size_t getIndex(int nx, int ny, int nz) {
        int n[] = {nx, ny, nz};
        for (int axis = 0; axis < 3; ++axis) {
            if (n[axis] >= ngrid[axis])
                n[axis] = 0;
            if (n[axis] < 0)
                n[axis] = ngrid[axis] - 1;
        }

        return n[2] + ngrid_z * (n[1] + ngrid[1] * n[0]);
    }

    void getKFromIndex(size_t i, double k[3]) {
        int kn[3];

        size_t temp = i / ngrid_kz;
        kn[2] = i % ngrid_kz;
        kn[0] = temp / ngrid[1];
        kn[1] = temp % ngrid[1];

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
    void reverseInterpolate(double coord[3], double val);
};

#endif
