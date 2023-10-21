#ifndef REALFIELD_H
#define REALFIELD_H

#include <complex>
#include <vector>

#include <fftw3.h>

enum FFTW_CURRENT_SPACE
{
    X_SPACE,
    K_SPACE
};

// this file containes an object to use real fields with fftw
// which should make it easy to track and operate on multiple
// real fields.

class RealField
{
    int size_real, size_complex;
    double length;

    fftw_plan p_x2k;
    fftw_plan p_k2x;

    void construct(double *data=NULL);

public:
    FFTW_CURRENT_SPACE current_space;

    double *field_x;
    std::complex<double> *field_k;

    // If data is not Null, then out-of-place FFT is performed with field_x=data
    // If data is Null, field_k is linked field_x
    RealField(int data_size, double length1, double *data=NULL);
    RealField(const RealField &rf);
    ~RealField();

    int size_k() const { return size_complex; };
    void setFFTWSpace(FFTW_CURRENT_SPACE which_space);

    void rawFFTX2K();
    void fftX2K();
    void fftK2X();

    // Must have the same size
    void changeData(double *newdata);

    // void getPowerSpectrum(double *ps, const double *kband_edges, int number_of_bins);
    // void getPowerSpectrum(double *ps, const double *kband_edges, int number_of_bins, int *bincount);
    void deconvolve(double (*f)(double, void*), void *params);
    void deconvolveSinc(double m); //, double downsampling=-1);

    //void applyGaussSmoothing(double r);
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
    };

    void fftK2X() {
        fftw_execute(p_r2r);
        for (double &f : field_x)
            f /= length;
    };
};


#endif
