#include "mathtools/real_field_3d.hpp"

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

    ngrid_z = ngrid[2] + 2 - (ngrid[2] % 2);
    ngrid_kz = ngrid[2] / 2 + 1;
    ngrid_xy = ngrid[0] * ngrid[1];
    size_complex = ngrid_xy * ngrid_kz;

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
    for (size_t i = 0; i < size_complex; ++i)
        field_k[i] *= gridvol;
} 


void RealField3D::fftK2X() {
    fftw_execute(p_k2x);

    #pragma omp parallel for simd collapse(2)
    for (int ij = 0; ij < ngrid_xy; ++ij)
        for (int k = 0; k < ngrid[2]; ++k)
            field_x[k + ngrid_z * ij] /= totalvol;
}


double RealField3D::interpolate(double coord[3]) {
    int n[3];
    double d[3], r;

    coord[2] -= z0;
    for (int axis = 0; axis < 3; ++axis) {
        n[axis] = coord[axis] / dx[axis];
        d[axis] = coord[axis] / dx[axis] - n[axis];
    }

    r = field_x[getIndex(n[0], n[1], n[2])] * (1 - d[0]) * (1 - d[1]) * (1 - d[2]);

    r += field_x[getIndex(n[0], n[1], n[2] + 1)] * (1 - d[0]) * (1 - d[1]) * d[2];
    r += field_x[getIndex(n[0], n[1] + 1, n[2])] * (1 - d[0]) * d[1] * (1 - d[2]);
    r += field_x[getIndex(n[0] + 1, n[1], n[2])] * d[0] * (1 - d[1]) * (1 - d[2]);

    r += field_x[getIndex(n[0], n[1] + 1, n[2] + 1)] * (1 - d[0]) * d[1] * d[2];
    r += field_x[getIndex(n[0] + 1, n[1], n[2] + 1)] * d[0] * (1 - d[1]) * d[2];
    r += field_x[getIndex(n[0] + 1, n[1] + 1, n[2])] * d[0] * d[1] * (1 - d[2]);

    r += field_x[getIndex(n[0] + 1, n[1] + 1, n[2] + 1)] * d[0] * d[1] * d[2];

    return r;
}


void RealField3D::reverseInterpolate(double coord[3], double val) {
    int n[3];
    double d[3], r;

    coord[2] -= z0;
    for (int axis = 0; axis < 3; ++axis) {
        n[axis] = coord[axis] / dx[axis];
        d[axis] = coord[axis] / dx[axis] - n[axis];
    }

    field_x[getIndex(n[0], n[1], n[2])] += val * (1 - d[0]) * (1 - d[1]) * (1 - d[2]);

    field_x[getIndex(n[0], n[1], n[2] + 1)] += val * (1 - d[0]) * (1 - d[1]) * d[2];
    field_x[getIndex(n[0], n[1] + 1, n[2])] += val * (1 - d[0]) * d[1] * (1 - d[2]);
    field_x[getIndex(n[0] + 1, n[1], n[2])] += val * d[0] * (1 - d[1]) * (1 - d[2]);

    field_x[getIndex(n[0], n[1] + 1, n[2] + 1)] += val * (1 - d[0]) * d[1] * d[2];
    field_x[getIndex(n[0] + 1, n[1], n[2] + 1)] += val * d[0] * (1 - d[1]) * d[2];
    field_x[getIndex(n[0] + 1, n[1] + 1, n[2])] += val * d[0] * d[1] * (1 - d[2]);

    field_x[getIndex(n[0] + 1, n[1] + 1, n[2] + 1)] += val * d[0] * d[1] * d[2];
}
