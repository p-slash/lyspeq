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


RealField3D::RealField3D(const RealField3D &rhs) : p_x2k(nullptr), p_k2x(nullptr)
{
    for (int axis = 0; axis < 3; ++axis) {
        ngrid[axis] = rhs.ngrid[axis];
        length[axis] = rhs.length[axis];
    }
    z0 = rhs.z0;
}


void RealField3D::construct() {
    size_real = 1;
    cellvol = 1;
    totalvol = 1;

    for (int axis = 0; axis < 3; ++axis) {
        k_fund[axis] = 2 * MY_PI / length[axis];
        dx[axis] = length[axis] / ngrid[axis];

        size_real *= ngrid[axis];
        cellvol *= dx[axis];
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
        reinterpret_cast<fftw_complex*>(field_k.data()), field_x,
        FFTW_MEASURE);

}

void RealField3D::fftX2K() {
    fftw_execute(p_x2k);

    #pragma omp parallel for simd
    for (size_t i = 0; i < size_complex; ++i)
        field_k[i] *= cellvol;
} 


void RealField3D::fftK2X() {
    fftw_execute(p_k2x);

    #pragma omp parallel for simd collapse(2)
    for (int ij = 0; ij < ngrid_xy; ++ij)
        for (int k = 0; k < ngrid[2]; ++k)
            field_x[k + ngrid_z * ij] /= totalvol;
}


double RealField3D::dot(const RealField3D &other) {
    double result = 0;
    #pragma omp parallel for reduction(+:result)
    for (size_t j = 0; j < size_real; ++j) {
        size_t i = getCorrectIndexX(j);
        result += field_x[i] * other.field_x[i];
    }

    return result;
}


std::vector<size_t> RealField3D::findNeighboringPixels(
        size_t i, double radius
) const {
    int n[3], dn[3], ntot = 1;
    std::vector<size_t> neighbors;

    getNFromIndex(i, n);
    for (int axis = 0; axis < 3; ++axis) {
        dn[axis] = ceil(radius / dx[axis]) - 1;
        ntot *= 2 * dn[axis] + 1;
    }

    radius *= radius;
    neighbors.reserve(ntot);
    for (int x = -dn[0]; x <= dn[0]; ++x) {
        double x2 = x * dx[0];  x2 *= x2;

        for (int y = -dn[1]; y <= dn[1]; ++y) {
            double y2 = y * dx[1];  y2 *= y2;

            for (int z = -dn[2]; z <= dn[2]; ++z) {
                double z2 = z * dx[2];  z2 *= z2;
                z2 += x2 + y2;

                if (z2 > radius)
                    continue;

                neighbors.push_back(getIndex(n[0] + x, n[1] + y, n[2] + z));
            }
        }
    }

    return neighbors;
}


double RealField3D::interpolate(double coord[3]) const {
    int n[3];
    double d[3], r;

    coord[2] -= z0;
    for (int axis = 0; axis < 3; ++axis) {
        d[axis] = coord[axis] / dx[axis];
        n[axis] = d[axis];
        d[axis] -= n[axis];
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


void RealField3D::reverseInterpolateCIC(double coord[3], double val) {
    int n[3];
    double d[3];

    coord[2] -= z0;
    for (int axis = 0; axis < 3; ++axis) {
        d[axis] = coord[axis] / dx[axis];
        n[axis] = d[axis];
        d[axis] -= n[axis];
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
