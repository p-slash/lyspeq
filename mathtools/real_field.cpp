#include "mathtools/real_field.hpp"

#include <algorithm>
#include <cmath>
#include <cassert>

const double MY_PI = 3.14159265358979323846;

RealField::RealField(int data_size, double dx1) : p_x2k(nullptr), p_k2x(nullptr)
{
    size_real = -1;
    dx = -1.;
    resize(data_size, dx1);
}


void RealField::resize(int data_size, double dx1) {
    bool new_data = size_real != data_size;
    size_real = data_size;
    size_complex = size_real / 2 + 1;

    if (new_data) {
        x.resize(size_real);
        k.resize(size_complex);
        field_x.resize(size_real);
        field_k.resize(size_complex);

        if (p_x2k != nullptr) {
            fftw_destroy_plan(p_x2k);
            fftw_destroy_plan(p_k2x);
        }

        p_x2k = fftw_plan_dft_r2c_1d(
            size_real, field_x.data(),
            reinterpret_cast<fftw_complex*>(field_k.data()), FFTW_MEASURE);
        p_k2x = fftw_plan_dft_c2r_1d(
            size_real, reinterpret_cast<fftw_complex*>(field_k.data()),
            field_x.data(), FFTW_MEASURE);
    }

    if ((fabs(dx1 - dx) > 1e-12) || new_data)
        reStep(dx1);
}


void RealField::reStep(double dx1) {
    dx = dx1;
    length = dx * size_real;
    double k_fund = 2 * MY_PI / length;

    for (int i = 0; i < size_real; ++i)
        x[i] = i * dx;
    for (int i = 0; i < size_complex; ++i)
        k[i] = i * k_fund;
}


RealField::RealField(const RealField &rf) {
    p_x2k = nullptr;
    p_k2x = nullptr;
    resize(rf.size_real, rf.dx);

    std::copy(
        rf.field_k.begin(), rf.field_k.end(),
        field_k.begin());
    std::copy(
        rf.field_x.begin(), rf.field_x.end(),
        field_x.begin());
}


RealField::~RealField() {
    fftw_destroy_plan(p_x2k);
    fftw_destroy_plan(p_k2x);
}


void RealField::fftX2K() {
    fftw_execute(p_x2k);

    // Normalization of FFT
    for (auto &f : field_k)
        f *= dx;
} 


void RealField::fftK2X() {
    fftw_execute(p_k2x);

    for (double &f : field_x)
        f /= length;
}


void RealField::deconvolveSinc(double m)//, double downsampling)
{
    fftX2K();

    double x;

    for (int i = 1; i < size_complex; i++)
    {
        x = MY_PI * m * i / size_real;
        // y = x / (downsampling+1e-8);
        x = sin(x)/x;
        // y = downsampling>1 ? sin(y)/y : 1;
        field_k[i] /= x;
    }

    fftK2X();
}
