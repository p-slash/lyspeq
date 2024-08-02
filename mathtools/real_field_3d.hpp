#ifndef REALFIELD_3D_H
#define REALFIELD_3D_H

#include <complex>
#include <memory>
#include <vector>

#include <fftw3.h>

#include "core/omp_manager.hpp"
#include "mathtools/my_random.hpp"

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


/* 3D FFT 
    Input coordinates are assumed to shifted by the following relation:
        y += length[1] / 2
        z -= z0
*/
class RealField3D {
    fftw_plan p_x2k;
    fftw_plan p_k2x;
    std::vector<MyRNG> rngs;

    bool _inplace;
    std::unique_ptr<double[]> _field_x;
public:
    size_t size_complex, size_real, ngrid_xy, ngrid_z, ngrid_kz;
    int ngrid[3];
    float dx[3], length[3], z0;
    double k_fund[3], cellvol, invtotalvol, invsqrtcellvol;
    std::vector<std::complex<double>> field_k;
    double *field_x;

    RealField3D();
    RealField3D(const RealField3D &rhs) = delete;
    RealField3D(RealField3D &&rhs) = delete;

    /* Copy constructor. Need to call construct! */
    void copy(const RealField3D &rhs);
    void construct(bool inp=true);
    void initRngs(std::seed_seq *seq);

    ~RealField3D() {
        fftw_destroy_plan(p_x2k);
        fftw_destroy_plan(p_k2x);
    };

    void zero_field_k() { std::fill(field_k.begin(), field_k.end(), 0); }
    void zero_field_x() {
        if (_inplace)
            zero_field_k();
        else
            std::fill_n(field_x, size_real, 0);
    }

    void fillRndNormal();
    void fillRndOnes();

    void rawFftX2K() { fftw_execute(p_x2k); }
    void rawFftK2X() { fftw_execute(p_k2x); }
    void fftX2K();
    void fftK2X();
    double dot(const RealField3D &other);

    size_t getIndex(int nx, int ny, int nz) const;
    size_t getNgpIndex(float coord[3]) const;
    void getCicIndices(float coord[3], size_t idx[8]) const;
    inline size_t getCorrectIndexX(size_t j) {
        return j + (j / ngrid[2]) * (ngrid[2] - ngrid_z);
    }

    void getNFromIndex(size_t i, int n[3]) const;
    void getKFromIndex(size_t i, double k[3]) const;
    void getK2KzFromIndex(size_t i, double &k2, double &kz) const;
    double getKperpFromIperp(size_t iperp) const;
    void getKperpKzFromIndex(size_t i, double &kperp, double &kz) const;
    std::unique_ptr<double[]> getKperpArray() const;

    std::vector<size_t> findNeighboringPixels(size_t i, double radius) const;
    double interpolate(float coord[3]) const;
    void reverseInterpolateCIC(float coord[3], double val);
    void reverseInterpolateNGP(float coord[3], double val) {
        field_x[getNgpIndex(coord)] += val;
    }

    void reverseInterpolateNGP_safe(float coord[3], size_t idx, double val) {
        if (idx != getNgpIndex(coord))
            return;
        field_x[idx] += val;
    }
};

#endif
