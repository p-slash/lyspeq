#include "gsltools/real_field.hpp"

#include <algorithm>
#include <cmath>
#include <cassert>

#define PI 3.14159265359

void RealField::construct(double *data)
{
    field_k = new std::complex<double>[size_complex];

    if (data == NULL)
        field_x = reinterpret_cast<double*>(field_k);
    else
        field_x = data;
    
    p_x2k = fftw_plan_dft_r2c_1d(size_real, field_x, reinterpret_cast<fftw_complex*>(field_k),
        FFTW_MEASURE);
    p_k2x = fftw_plan_dft_c2r_1d(size_real, reinterpret_cast<fftw_complex*>(field_k), field_x,
        FFTW_MEASURE);
}

RealField::RealField(int data_size, double length1, double *data) : size_real(data_size), 
    length(length1), current_space(X_SPACE)
{
    size_complex = size_real / 2 + 1;

    construct(data);
}

RealField::RealField(const RealField &rf)
{
    size_real     = rf.size_real;
    size_complex  = rf.size_complex;
    current_space = rf.current_space;
    length        = rf.length;

    construct();
    std::copy(rf.field_k, rf.field_k+size_complex, field_k);
}

RealField::~RealField()
{
    fftw_destroy_plan(p_x2k);
    fftw_destroy_plan(p_k2x);

    delete [] field_k;
}

void RealField::changeData(double *newdata)
{
    if (newdata == field_x)    return;

    fftw_destroy_plan(p_x2k);
    fftw_destroy_plan(p_k2x);

    field_x = newdata;

    p_x2k = fftw_plan_dft_r2c_1d(size_real, field_x, reinterpret_cast<fftw_complex*>(field_k),
        FFTW_MEASURE);
    p_k2x = fftw_plan_dft_c2r_1d(size_real, reinterpret_cast<fftw_complex*>(field_k), field_x,
        FFTW_MEASURE);
}

void RealField::setFFTWSpace(FFTW_CURRENT_SPACE which_space)
{
    current_space = which_space;
}

void RealField::fftX2K()
{
    assert(current_space == X_SPACE);

    fftw_execute(p_x2k);

    // Normalization of FFT
    double norm  = length / size_real;

    std::for_each(field_k, field_k+size_complex, [&](std::complex<double> &f) { f *= norm; });

    current_space = K_SPACE;
} 

void RealField::fftK2X()
{
    assert(current_space == K_SPACE);

    fftw_execute(p_k2x);

    std::for_each(field_x, field_x+size_real, [&](double &f) { f /= length; });

    current_space = X_SPACE;
}

void RealField::getPowerSpectrum(double *ps, const double *kband_edges, int number_of_bins)
{
    int *bincount = new int[number_of_bins];

    getPowerSpectrum(ps, kband_edges, number_of_bins, bincount);
    delete [] bincount;
}

void RealField::getPowerSpectrum(double *ps, const double *kband_edges, int number_of_bins, 
    int *bincount)
{
    if (current_space == X_SPACE)   fftX2K();

    int bin_no = 0;

    double temp;

    for (int n = 0; n < number_of_bins; n++)
    {
        bincount[n] = 0;
        ps[n]       = 0;
    }

    for (int i = 1; i < size_complex; i++)
    {
        temp = i * (2. * PI / length);

        if (temp < kband_edges[0] )             continue;
        if (temp > kband_edges[number_of_bins]) break;

        while (kband_edges[bin_no + 1] < temp)  bin_no++;

        temp = std::norm(field_k[i]);

        bincount[bin_no]++;
        ps[bin_no] += temp;

        // if (i == 0 || i == size_real / 2)
        // {
        //     bincount[bin_no]++;
        //     ps[bin_no] += temp;
        // }
        // else
        // {
        //     bincount[bin_no] += 2;
        //     ps[bin_no] += 2. * temp;
        // }
    }

    for (int n = 0; n < number_of_bins; n++)
    {
        ps[n] /= length * bincount[n];
        
        if (bincount[n] == 0)   ps[n] = 0;
    }
}

void RealField::deconvolve(double (*f)(double, void*), void *params)
{
    if (current_space == X_SPACE)   fftX2K();

    double k;

    for (int i = 0; i < size_complex; i++)
    {
        k = i * (2. * PI / length);

        field_k[i] /= f(k, params);
    }
}

void RealField::deconvolveSinc(double m)//, double downsampling)
{
    if (current_space == X_SPACE)   fftX2K();

    double x, y;

    for (int i = 1; i < size_complex; i++)
    {
        x = PI * m * i / size_real;
        // y = x / (downsampling+1e-8);
        x = sin(x)/x;
        // y = downsampling>1 ? sin(y)/y : 1;
        field_k[i] /= x;
    }

    fftK2X();
}

#undef PI

// void RealField::applyGaussSmoothing(double r)
// {
//     assert(current_space == K_SPACE);
    
//     int k[3];
//     double k_physical_sq;

//     for (long i = 1; i < size_complex; i++)
//     {
//         getKFromIndex(i, k);
        
//         k_physical_sq = (k[0]*k[0] + k[1]*k[1] + k[2]*k[2]) * K_NUMBER_2_PHYSICAL * K_NUMBER_2_PHYSICAL;

//         field_k[i] *= exp(-k_physical_sq * r * r / 2.0);
//     }
// }








