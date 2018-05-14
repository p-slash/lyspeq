/* this file containes an object to use real fields with fftw
 * which should make it easy to track and operate on multiple
 * real fields.
 */

#ifndef REAL_FIELD_1D_H
#define REAL_FIELD_1D_H

#include <complex>
#include <fftw3.h>

enum FFTW_CURRENT_SPACE
{
    X_SPACE,
    K_SPACE
};

class RealField1D
{
    int N_BIN, N_BIN_COMPLEX;
    double L_BOX;

    fftw_plan p_x2k;
    fftw_plan p_k2x;

    void construct();
    
public:
    FFTW_CURRENT_SPACE current_space;
    
    double *field_x;
    std::complex<double> *field_k;
    
    RealField1D(const double *data, int data_size, double length);
    RealField1D(const RealField1D &rf);
    ~RealField1D();


    void setFFTWSpace(FFTW_CURRENT_SPACE which_space);
    
    void fftX2K();
    void fftK2X();

    void getPowerSpectrum(double *ps, const double *kband_edges, int number_of_bins);
    //void applyGaussSmoothing(double r);
};


#endif
