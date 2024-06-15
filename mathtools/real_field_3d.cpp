#include "mathtools/real_field_3d.hpp"

#include "core/omp_manager.hpp"

// #include <algorithm>
#include <cmath>
// #include <cassert>

const double MY_PI = 3.14159265359;


RealField3D::RealField3D() : p_x2k(nullptr), p_k2x(nullptr)
{
    size_complex = 0;
    size_real = 0;

    for (int axis = 0; axis < 3; ++axis) {
        ngrid[axis] = 0;
        length[axis] = 0;
        dx[axis] = 0;
    }
}


void RealField3D::construct() {
    size_real = 1;
    gridvol = 1;
    totalvol = 1;

    for (int axis = 0; axis < 3; ++axis) {
        k_fund[axis] = 2 * MY_PI / length[axis];
        dx[axis] = length[axis] / ngrid[axis];

        size_real *= ngrid[axis];
        gridvol *= dx[axis];
        totalvol *= length[axis];
    }

    padding = 2 - (ngrid[2] % 2);
    size_complex = ngrid[0] * ngrid[1] * (ngrid[2] / 2 + 1);
    ngrid_xy = ngrid[0] * ngrid[1];

    field_k.resize(size_complex);
    field_x = reinterpret_cast<double*>(field_k.data());
    
    p_x2k = fftw_plan_dft_r2c_3d(
        ngrid[0], ngrid[1], ngrid[2],
        field_x, reinterpret_cast<fftw_complex*>(field_k.data()),
        FFTW_MEASURE);

    p_k2x = fftw_plan_dft_c2r_3d(
        ngrid[0], ngrid[1], ngrid[2],
        reinterpret_cast<fftw_complex*>(field_k.data()), field_x, \
        FFTW_MEASURE);

}

void RealField3D::fftX2K() {
    fftw_execute(p_x2k);

    #pragma omp parallel for simd
    for (auto &f : field_k)
        f *= gridvol;
} 


void RealField3D::fftK2X() {
    fftw_execute(p_k2x);

    #pragma omp parallel for simd collapse(2)
    for (int ij = 0; ij < ngrid_xy; ++ij)
        for (int k = 0; k < ngrid[2]; ++k)
            field_x[k + (ngrid[2] + padding) * ij] /= totalvol;
}
