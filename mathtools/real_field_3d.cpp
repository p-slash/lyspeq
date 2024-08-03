#include <cmath>

#include "mathtools/real_field_3d.hpp"
#include "mathtools/matrix_helper.hpp"
// #include <algorithm>
// #include <cassert>

const double MY_PI = 3.14159265359;

void RealField3D::initRngs(std::seed_seq *seq) {
    const int N = myomp::getMaxNumThreads();
    rngs.resize(N);
    std::vector<size_t> seeds(N);
    seq->generate(seeds.begin(), seeds.end());
    for (int i = 0; i < N; ++i)
        rngs[i].seed(seeds[i]);
}


RealField3D::RealField3D() : p_x2k(nullptr), p_k2x(nullptr) {
    size_complex = 0;
    size_real = 0;

    for (int axis = 0; axis < 3; ++axis) {
        ngrid[axis] = 0;
        length[axis] = 0;
        dx[axis] = 0;
    }
}


void RealField3D::copy(const RealField3D &rhs)
{
    p_x2k = nullptr; p_k2x = nullptr;
    for (int axis = 0; axis < 3; ++axis) {
        ngrid[axis] = rhs.ngrid[axis];
        length[axis] = rhs.length[axis];
    }
    z0 = rhs.z0;
}


void RealField3D::construct(bool inp) {
    _inplace = inp;
    size_real = 1;
    cellvol = 1;
    invtotalvol = 1;

    for (int axis = 0; axis < 3; ++axis) {
        k_fund[axis] = 2 * MY_PI / length[axis];
        dx[axis] = length[axis] / ngrid[axis];

        size_real *= ngrid[axis];
        cellvol *= dx[axis];
        invtotalvol *= length[axis];
    }
    invtotalvol = 1.0 / invtotalvol;
    invsqrtcellvol = 1.0 / sqrt(cellvol);

    ngrid_kz = ngrid[2] / 2 + 1;
    ngrid_xy = ngrid[0] * ngrid[1];
    size_complex = ngrid_xy * ngrid_kz;

    field_k.resize(size_complex);

    if (_inplace) {
        field_x = reinterpret_cast<double*>(field_k.data());
        ngrid_z = ngrid[2] + 2 - (ngrid[2] % 2);
    }
    else {
        _field_x = std::make_unique<double[]>(size_real);
        field_x = _field_x.get();
        ngrid_z = ngrid[2];
    }
    
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

    #pragma omp parallel for
    for (size_t ij = 0; ij < ngrid_xy; ++ij)
        for (size_t k = 0; k < ngrid_kz; ++k)
            field_k[k + ngrid_kz * ij] *= cellvol;
} 


void RealField3D::fftK2X() {
    fftw_execute(p_k2x);

    #pragma omp parallel for
    for (size_t ij = 0; ij < ngrid_xy; ++ij)
        cblas_dscal(ngrid[2], invtotalvol, field_x + ngrid_z * ij, 1);
}


void RealField3D::fillRndNormal() {
    #pragma omp parallel for
    for (size_t ij = 0; ij < ngrid_xy; ++ij)
        rngs[myomp::getThreadNum()].fillVectorNormal(
            field_x + ngrid_z * ij, ngrid[2]);
}


void RealField3D::fillRndOnes() {
    #pragma omp parallel for
    for (size_t ij = 0; ij < ngrid_xy; ++ij)
        rngs[myomp::getThreadNum()].fillVectorOnes(
            field_x + ngrid_z * ij, ngrid[2]);
}


double RealField3D::dot(const RealField3D &other) {
    double result = 0;

    #pragma omp parallel for reduction(+:result)
    for (size_t ij = 0; ij < ngrid_xy; ++ij)
        result += cblas_ddot(
            ngrid[2], field_x + ngrid_z * ij, 1,
            other.field_x + ngrid_z * ij, 1);

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


double RealField3D::interpolate(float coord[3]) const {
    int n[3];
    float d[3];
    double r = 0;
    size_t idx0 = 0;

    for (int axis = 0; axis < 3; ++axis) {
        d[axis] = coord[axis] / dx[axis];
        n[axis] = d[axis];
        d[axis] -= n[axis];
    }

    idx0 = getIndex(n[0], n[1], n[2]);
    r = field_x[idx0] * ((1.0f - d[0]) * (1.0f - d[1]) * (1.0f - d[2]));
    r += field_x[idx0 + 1] * ((1.0f - d[0]) * (1.0f - d[1]) * d[2]);
    r += field_x[idx0 + ngrid_z] * ((1.0f - d[0]) * d[1] * (1.0f - d[2]));
    r += field_x[idx0 + ngrid_z + 1] * ((1.0f - d[0]) * d[1] * d[2]);

    idx0 = getIndex(n[0] + 1, n[1], n[2]);
    r += field_x[idx0] * (d[0] * (1.0f - d[1]) * (1.0f - d[2]));
    r += field_x[idx0 + 1] * (d[0] * (1.0f - d[1]) * d[2]);
    r += field_x[idx0 + ngrid_z] * (d[0] * d[1] * (1.0f - d[2]));
    r += field_x[idx0 + ngrid_z + 1] * (d[0] * d[1] * d[2]);

    return r;
}


void RealField3D::reverseInterpolateCIC(float coord[3], double val) {
    int n[3];
    float d[3];

    for (int axis = 0; axis < 3; ++axis) {
        d[axis] = coord[axis] / dx[axis];
        n[axis] = d[axis];
        d[axis] -= n[axis];
    }

    size_t idx0 = getIndex(n[0], n[1], n[2]);
    field_x[idx0] += val * ((1.0f - d[0]) * (1.0f - d[1]) * (1.0f - d[2]));
    field_x[idx0 + 1] += val * ((1.0f - d[0]) * (1.0f - d[1]) * d[2]);
    field_x[idx0 + ngrid_z] += val * ((1.0f - d[0]) * d[1] * (1.0f - d[2]));
    field_x[idx0 + ngrid_z + 1] += val * ((1.0f - d[0]) * d[1] * d[2]);

    idx0 = getIndex(n[0] + 1, n[1], n[2]);
    field_x[idx0] += val * (d[0] * (1.0f - d[1]) * (1.0f - d[2]));
    field_x[idx0 + 1] += val * (d[0] * (1.0f - d[1]) * d[2]);
    field_x[idx0 + ngrid_z] += val * (d[0] * d[1] * (1.0f - d[2]));
    field_x[idx0 + ngrid_z + 1] += val * (d[0] * d[1] * d[2]);
}


size_t RealField3D::getIndex(int nx, int ny, int nz) const {
    int n[] = {nx, ny, nz};
    // only x direction is Periodic
    if (n[0] >= ngrid[0])
        n[0] -= ngrid[0];
    if (n[0] < 0)
        n[0] += ngrid[0];

    return n[2] + ngrid_z * (n[1] + ngrid[1] * n[0]);
}

size_t RealField3D::getNgpIndex(float coord[3]) const {
    int n[3];

    n[0] = roundf(coord[0] / dx[0]);
    n[1] = roundf(coord[1] / dx[1]);
    n[2] = roundf(coord[2] / dx[2]);

    return getIndex(n[0], n[1], n[2]);
}

void RealField3D::getCicIndices(float coord[3], size_t idx[8]) const {
    int n[3];
    float d[3];

    for (int axis = 0; axis < 3; ++axis) {
        d[axis] = coord[axis] / dx[axis];
        n[axis] = d[axis];
        d[axis] -= n[axis];
    }

    idx[0] = getIndex(n[0], n[1], n[2]);
    idx[1] = idx[0] + 1;
    idx[2] = idx[0] + ngrid_z;
    idx[3] = idx[1] + 1;

    idx[4] = getIndex(n[0] + 1, n[1], n[2]);
    idx[5] = idx[4] + 1;
    idx[6] = idx[4] + ngrid_z;
    idx[7] = idx[6] + 1;
}

void RealField3D::getNFromIndex(size_t i, int n[3]) const {
    size_t nperp = i / ngrid_z;
    n[2] = i % ngrid_z;
    n[0] = nperp / ngrid[1];
    n[1] = nperp % ngrid[1];
}

void RealField3D::getKFromIndex(size_t i, double k[3]) const {
    int kn[3];

    size_t iperp = i / ngrid_kz;
    kn[2] = i % ngrid_kz;
    kn[0] = iperp / ngrid[1];
    kn[1] = iperp % ngrid[1];

    if (kn[0] > (ngrid[0] / 2))
        kn[0] -= ngrid[0];

    if (kn[1] > (ngrid[1] / 2))
        kn[1] -= ngrid[1];

    for (int axis = 0; axis < 3; ++axis)
        k[axis] = k_fund[axis] * kn[axis];
}

void RealField3D::getK2KzFromIndex(size_t i, double &k2, double &kz) const {
    double ks[3];
    getKFromIndex(i, ks);
    kz = ks[2];

    k2 = 0;
    for (int axis = 0; axis < 3; ++axis)
        k2 += ks[axis] * ks[axis];
}

double RealField3D::getKperpFromIperp(size_t iperp) const {
    int kn[2];
    double ks[2];
    kn[0] = iperp / ngrid[1];
    kn[1] = iperp % ngrid[1];

    if (kn[0] > (ngrid[0] / 2))
        kn[0] -= ngrid[0];

    ks[0] = kn[0] * k_fund[0];

    if (kn[1] > (ngrid[1] / 2))
        kn[1] -= ngrid[1];

    ks[1] = kn[1] * k_fund[1];

    return sqrt(ks[0] * ks[0] + ks[1] * ks[1]);
}


std::unique_ptr<double[]> RealField3D::getKperpArray() const {
    auto res = std::make_unique<double[]>(ngrid_xy);
    for (size_t ij = 0; ij < ngrid_xy; ++ij)
        res[ij] = getKperpFromIperp(ij);
    return res;
}


void RealField3D::getKperpKzFromIndex(size_t i, double &kperp, double &kz)
const {
    double ks[3];
    getKFromIndex(i, ks);
    kperp = sqrt(ks[0] * ks[0] + ks[1] * ks[1]);
    kz = ks[2];
}
