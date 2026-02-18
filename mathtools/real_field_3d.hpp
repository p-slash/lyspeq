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
    float dx[3], length[3], xyz0[3];
    double k_fund[3], cellvol, invtotalvol, invsqrtcellvol, celldiag;
    std::unique_ptr<std::complex<double>[]> field_k;
    std::unique_ptr<double[]>  iasgn_window_xy, iasgn_window_xy2,
                               iasgn_window_z, iasgn_window_z2;
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
        iasgn_window_xy2.reset();  iasgn_window_z2.reset();
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
    void convolvePk(
            const DiscreteLogLogInterpolation2D<T1, T2> &Pk,
            bool predeconvolve=false
    ) {
        // S . x multiplication
        // Normalization including cellvol and N^3 yields inverse total volume
        fftw_execute(p_x2k);

        if (predeconvolve) {
            #pragma omp parallel for
            for (size_t ij = 0; ij < ngrid_xy; ++ij)
                for (size_t k = 0; k < ngrid_kz; ++k)
                    field_k[k + ngrid_kz * ij] *=
                        iasgn_window_xy2[ij] * iasgn_window_z2[k];
        }

        #pragma omp parallel for
        for (size_t ij = 0; ij < ngrid_xy; ++ij) {
            double kperp = getKperpFromIperp(ij);

            for (size_t k = 0; k < ngrid_kz; ++k) {
                field_k[k + ngrid_kz * ij] *=
                    invtotalvol * Pk.evaluate(kperp, k * k_fund[2]);
            }
        }
        fftw_execute(p_k2x);
    }

    template<class T1, class T2>
    void convolveSqrtPk(
            const DiscreteLogLogInterpolation2D<T1, T2> &Pk,
            bool predeconvolve=false
    ) {
        double norm = cellvol * invsqrtcellvol * invtotalvol;
        fftw_execute(p_x2k);

        if (predeconvolve) {
            #pragma omp parallel for
            for (size_t ij = 0; ij < ngrid_xy; ++ij)
                for (size_t k = 0; k < ngrid_kz; ++k)
                    field_k[k + ngrid_kz * ij] *=
                        iasgn_window_xy[ij] * iasgn_window_z[k];
        }

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
    static std::function<double(size_t, size_t)> getNormFunc(
            const RealField3D &mesh, const RealField3D *other=nullptr,
            bool predeconvolve=false
    ) {
            std::function<double(size_t, size_t)> my_norm;
            if ((other == nullptr) || (&mesh == other)) {
                if (predeconvolve)
                    my_norm = [&mesh](size_t ij, size_t k) {
                        size_t jj = k + mesh.ngrid_kz * ij;
                        double window = mesh.iasgn_window_xy2[ij] * mesh.iasgn_window_z2[k];
                        return std::norm(mesh.field_k[jj]) * window;
                    };
                else
                    my_norm = [&mesh](size_t ij, size_t k) {
                        size_t jj = k + mesh.ngrid_kz * ij;
                        return std::norm(mesh.field_k[jj]);
                    };
            }
            else {
                if (predeconvolve)
                    my_norm = [&mesh, &other](size_t ij, size_t k) {
                        size_t jj = k + mesh.ngrid_kz * ij;
                        double window = mesh.iasgn_window_xy2[ij] * mesh.iasgn_window_z2[k];
                        return (mesh.field_k[jj].real() * other.field_k[jj].real()
                               + mesh.field_k[jj].imag() * other.field_k[jj].imag()) * window;
                    };
                else
                    my_norm = [&mesh, &other](size_t ij, size_t k) {
                        size_t jj = k + mesh.ngrid_kz * ij;
                        return mesh.field_k[jj].real() * other.field_k[jj].real()
                        + mesh.field_k[jj].imag() * other.field_k[jj].imag();
                    };
            }
        return my_norm;
    }

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
    double interpolateLanczos(float coord[3]) const;
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
