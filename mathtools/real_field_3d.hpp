#ifndef REALFIELD_3D_H
#define REALFIELD_3D_H

#include <complex>
#include <memory>
#include <vector>

#include <fftw3.h>

#include "core/omp_manager.hpp"
#include "mathtools/my_random.hpp"
#include "mathtools/discrete_interpolation.hpp"

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

    bool _inplace, _periodic_x;
    std::unique_ptr<double[]> _field_x;
    void _setAssignmentWindows();
public:
    size_t size_complex, size_real, ngrid_xy, ngrid_z, ngrid_kz;
    int ngrid[3];
    float dx[3], length[3], z0;
    double k_fund[3], cellvol, invtotalvol, invsqrtcellvol;
    std::unique_ptr<std::complex<double>[]> field_k;
    std::unique_ptr<double[]>  iasgn_window_xy, iasgn_window_z;
    double *field_x;

    RealField3D();
    RealField3D(const RealField3D &rhs) = delete;
    RealField3D(RealField3D &&rhs) = delete;
    explicit operator bool() const { return p_x2k != nullptr; }

    /* Copy constructor. Need to call construct! */
    void copy(const RealField3D &rhs);
    void construct(bool inp=true);
    void disablePeriodicityX() { _periodic_x = false; };

    ~RealField3D() {
        fftw_destroy_plan(p_x2k);
        fftw_destroy_plan(p_k2x);
    };

    void free() {
        field_k.reset();  _field_x.reset();
        iasgn_window_xy.reset();  iasgn_window_z.reset();
        fftw_destroy_plan(p_x2k);
        fftw_destroy_plan(p_k2x);
        p_x2k = nullptr;  p_k2x = nullptr;
    }

    void zero_field_k() { std::fill_n(field_k.get(), size_complex, 0); }
    void zero_field_x() {
        if (_inplace)
            zero_field_k();
        else
            std::fill_n(field_x, size_real, 0);
    }

    void fillRndNormal(std::vector<MyRNG> &rngs_) {
        #pragma omp parallel for
        for (size_t ij = 0; ij < ngrid_xy; ++ij)
            rngs_[myomp::getThreadNum()].fillVectorNormal(
                field_x + ngrid_z * ij, ngrid[2]);
    }
    void fillRndOnes(std::vector<MyRNG> &rngs_) {
        #pragma omp parallel for
        for (size_t ij = 0; ij < ngrid_xy; ++ij)
            rngs_[myomp::getThreadNum()].fillVectorOnes(
                field_x + ngrid_z * ij, ngrid[2]);
    }

    void rawFftX2K() { fftw_execute(p_x2k); }
    void rawFftK2X() { fftw_execute(p_k2x); }
    void fftX2K();
    void fftK2X();

    template<class T1, class T2>
    void convolvePk(const DiscreteLogLogInterpolation2D<T1, T2> &Pk) {
        // S . x multiplication
        // Normalization including cellvol and N^3 yields inverse total volume
        fftw_execute(p_x2k);
        #pragma omp parallel for
        for (size_t ij = 0; ij < ngrid_xy; ++ij) {
            double kperp = getKperpFromIperp(ij);

            for (size_t k = 0; k < ngrid_kz; ++k) {
                // #ifdef DECONV_CIC_WINDOW
                // field_k[k + ngrid_kz * ij] *=
                //     iasgn_window_xy[ij] * iasgn_window_z[k];
                // #endif
                field_k[k + ngrid_kz * ij] *=
                    invtotalvol * Pk.evaluate(kperp, k * k_fund[2]);
            }
        }
        fftw_execute(p_k2x);
    }

    template<class T1, class T2>
    void convolveSqrtPk(const DiscreteLogLogInterpolation2D<T1, T2> &Pk) {
        double norm = cellvol * invsqrtcellvol * invtotalvol;

        fftw_execute(p_x2k);
        #pragma omp parallel for
        for (size_t ij = 0; ij < ngrid_xy; ++ij) {
            double kperp = getKperpFromIperp(ij);

            for (size_t k = 0; k < ngrid_kz; ++k)
                field_k[k + ngrid_kz * ij] *=
                    norm * Pk.evaluateSqrt(kperp, k * k_fund[2]);
        }
        fftw_execute(p_k2x);
    }
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
    double getKperpFromIperp(size_t iperp, double &kx, double &ky) const;
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
