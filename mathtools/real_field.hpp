#ifndef REALFIELD_H
#define REALFIELD_H

#include <complex>
#include <vector>

#include <fftw3.h>

// this file containes an object to use real fields with fftw
// which should make it easy to track and operate on multiple
// real fields.

class RealField
{
    int size_real, size_complex;
    double length;

    fftw_plan p_x2k;
    fftw_plan p_k2x;

public:
    double dx;
    std::vector<double> field_x, x, k;
    std::vector<std::complex<double>> field_k;

    // If data is not Null, then out-of-place FFT is performed with field_x=data
    // If data is Null, field_k is linked field_x
    RealField(int data_size, double dx1);
    RealField(const RealField &rf);
    ~RealField();

    void resize(int data_size, double dx1);
    void reStep(double dx1);
    void zero_field_x() {
        std::fill(field_x.begin(), field_x.end(), 0);
    }
    void zero_field_k() {
        std::fill(field_k.begin(), field_k.end(), 0);
    }
    int size_k() const { return size_complex; };

    void rawFFTX2K() { fftw_execute(p_x2k); }
    void fftX2K();
    void fftK2X();

    void deconvolveSinc(double m);
    void smoothGaussian(double r);
};


// R2R transforms are not numerically as accurate
class RealFieldR2R {
    int N;
    double length, norm;

    fftw_plan p_r2r;

public:
    std::vector<double> field_x;

    RealFieldR2R(int data_size, double dx, unsigned flags=FFTW_MEASURE) {
        resize(data_size, dx, flags);
    };

    ~RealFieldR2R() {
        fftw_destroy_plan(p_r2r);
    }

    void zero() {
        std::fill(field_x.begin(), field_x.end(), 0);
    }

    void resize(int data_size, double dx, unsigned flags=FFTW_MEASURE) {
        N = 2 * (data_size - 1);
        length = N * dx;
        norm = dx;
        field_x.resize(data_size);
        
        p_r2r = fftw_plan_r2r_1d(
            data_size, field_x.data(), field_x.data(),
            FFTW_REDFT00, flags);
    }

    void fftX2K() {
        fftw_execute(p_r2r);
        for (double &f : field_x)
            f *= norm;
    }

    void fftK2X() {
        fftw_execute(p_r2r);
        for (double &f : field_x)
            f /= length;
    }
};


#endif
