#ifndef REALFIELD_H
#define REALFIELD_H

#include <complex>
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

    void setFFTWSpace(FFTW_CURRENT_SPACE which_space);
    
    void fftX2K();
    void fftK2X();

    // Must have the same size
    void changeData(double *newdata);

    void getPowerSpectrum(double *ps, const double *kband_edges, int number_of_bins);
    void getPowerSpectrum(double *ps, const double *kband_edges, int number_of_bins, int *bincount);
    void deconvolve(double (*f)(double, void*), void *params);
    void deconvolveSinc(double m); //, double downsampling=-1);

    //void applyGaussSmoothing(double r);
};


#endif
