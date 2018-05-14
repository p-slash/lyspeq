#include "real_field_1d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cassert>
#include <new>

#define PI 3.14159265359
#define ROUND_OFF_ERROR_THRESHOLD 1E-12

void copyArray(const std::complex<double> *source, std::complex<double> *target, int size)
{
    std::copy(  &source[0], \
                &source[0] + size, \
                &target[0]);
}

void copyArray(const double *source, double *target, int size)
{
    std::copy(  &source[0], \
                &source[0] + size, \
                &target[0]);
}

void RealField1D::construct()
{
    field_k = new std::complex<double>[N_BIN_COMPLEX];

    field_x = reinterpret_cast<double*>(field_k);
    
    p_x2k  = fftw_plan_dft_r2c_1d(  N_BIN,
                                    field_x, reinterpret_cast<fftw_complex*>(field_k), \
                                    FFTW_MEASURE);
    p_k2x  = fftw_plan_dft_c2r_1d(  N_BIN,
                                    reinterpret_cast<fftw_complex*>(field_k), field_x, \
                                    FFTW_MEASURE);

    // // initialize to 0
    // for (long long int i = 0; i < N_BIN_COMPLEX; i++)
    // {
    //     field_k[i] = 0;
    // }
}

RealField1D::RealField1D(const double *data, int data_size, double length)
{
    N_BIN           = data_size;
    L_BOX           = length;
    N_BIN_COMPLEX   = N_BIN / 2 + 1;

    current_space = X_SPACE;

    construct();
    copyArray(data, field_x, N_BIN);

    // printf("Creating a RealField1D: Done!\n");
    // fflush(stdout);
}

RealField1D::RealField1D(const RealField1D &rf)
{
    N_BIN           = rf.N_BIN;
    L_BOX           = rf.L_BOX;
    N_BIN_COMPLEX   = rf.N_BIN_COMPLEX;
    current_space   = rf.current_space;

    // printf("Copying RealField1D ...\n");
    // fflush(stdout);

    construct();

    copyArray(rf.field_k, field_k, N_BIN_COMPLEX);

    // printf("Copying RealField1D: Done!\n");
    // fflush(stdout);
}

RealField1D::~RealField1D()
{
    fftw_destroy_plan(p_x2k);
    fftw_destroy_plan(p_k2x);

    delete [] field_k;

    // printf("RealField1D destructor called.\n");
}



void RealField1D::setFFTWSpace(FFTW_CURRENT_SPACE which_space)
{
    current_space = which_space;
}

void RealField1D::fftX2K()
{
    printf("Computing Fourier transfrom: ");
    fflush(stdout);

    assert(current_space == X_SPACE);

    fftw_execute(p_x2k);

    // Normalization of FFT
    double norm  = L_BOX / (1.0 * N_BIN);

    for (long long int i = 0; i < N_BIN_COMPLEX; i++)
    {
        field_k[i] *= norm;
    }

    current_space = K_SPACE;

    printf("Done!\n");
    fflush(stdout);
} 

void RealField1D::fftK2X()
{
    printf("Computing inverse Fourier transfrom: ");
    fflush(stdout);

    assert(current_space == K_SPACE);

    fftw_execute(p_k2x);

    for (long long int i = 0; i < N_BIN; i++)
    {
        field_x[i] /= L_BOX;
    }

    current_space = X_SPACE;

    printf("Done!\n");
    fflush(stdout);
}

void RealField1D::getPowerSpectrum(double *ps, const double *kband_edges, int number_of_bins)
{
    if (current_space == X_SPACE)
    {
        fftX2K();
    }

    int bin_no = 0;

    double temp;
    int *bincount = new int[number_of_bins];

    for (int n = 0; n < number_of_bins; n++)
    {
        bincount[n] = 0;
        ps[n]       = 0;
    }

    for (int i = 0; i < N_BIN_COMPLEX; i++)
    {
        temp = i * (2. * PI / L_BOX);

        if (temp < kband_edges[0] || temp > kband_edges[number_of_bins])
        {
            continue;
        }

        if (kband_edges[bin_no + 1] < temp)
        {
            bin_no++;
        }

        temp = std::norm(field_k[i]);

        if (i == 0 || i == N_BIN / 2)
        {
            bincount[bin_no]++;
            ps[bin_no] += temp;
        }
        else
        {
            bincount[bin_no] += 2;
            ps[bin_no] += 2. * temp;
        }
    }

    for (int n = 0; n < number_of_bins; n++)
    {
        ps[n] /= L_BOX * bincount[n];
        if (bincount[n]==0)
        {
            ps[n] = 0;
        }
    }

    delete [] bincount;
}

// void RealField1D::applyGaussSmoothing(double r)
// {
//     assert(current_space == K_SPACE);
    
//     int k[3];
//     double k_physical_sq;

//     for (long i = 1; i < N_BIN_COMPLEX; i++)
//     {
//         getKFromIndex(i, k);
        
//         k_physical_sq = (k[0]*k[0] + k[1]*k[1] + k[2]*k[2]) * K_NUMBER_2_PHYSICAL * K_NUMBER_2_PHYSICAL;

//         field_k[i] *= exp(-k_physical_sq * r * r / 2.0);
//     }
// }








