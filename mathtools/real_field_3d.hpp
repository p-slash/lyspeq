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
    size_t size_complex, size_real;
    int padding, ngrid_xy;
    double k_fund[3], gridvol, totalvol;

    fftw_plan p_x2k;
    fftw_plan p_k2x;

public:
    int ngrid[3];
    double length[3], dx[3], z0;
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
};

#endif
