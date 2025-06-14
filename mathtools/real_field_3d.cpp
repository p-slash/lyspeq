#include <cmath>

#include "mathtools/real_field_3d.hpp"
#include "mathtools/matrix_helper.hpp"
// #include <algorithm>
#include <cassert>

const double MY_PI = 3.14159265358979323846;
const double H_NYQ_1 = 0.70, H_NYQ_2 = 0.90;
#define A_LANCZOS 3

inline double sinc(double x) {
    if (x == 0)  return 1.0;
    return sin(x) / x;
}


inline double lanczos(double x) {
    if (x == 0)  return 1.0;
    // if (fabs(x) > A_LANCZOS)  return 0.0;
    double pix = MY_PI * x, pix2 = pix * pix;
    return A_LANCZOS * sin(pix) * sin(pix / A_LANCZOS) / pix2;
}


double _hanning(double x, double x1, double x2) {
    double dx = x2 - x1;
    if (x < x1)  return 1.0;
    if (x > x2)  return 0;
    double r = cos((x - x1) * MY_PI / 2 / dx);
    return r * r;
}


double smoothCICtoOne(double k, double a) {
    double window = sinc(k * a / 2.0), knyq = MY_PI / a,
           hann = _hanning(k, knyq * H_NYQ_1, knyq * H_NYQ_2);
    window = (1.0 - window * window) * hann ;
    return 1.0 - window;
}


void RealField3D::_setAssignmentWindows() {
    iasgn_window_xy = std::make_unique<double[]>(ngrid_xy);
    iasgn_window_z = std::make_unique<double[]>(ngrid_kz);

    for (size_t ij = 0; ij < ngrid_xy; ++ij) {
        double kx, ky, window;
        getKperpFromIperp(ij, kx, ky);
        kx = fabs(kx);  ky = fabs(ky);
        iasgn_window_xy[ij] = smoothCICtoOne(kx, dx[0])
                              * smoothCICtoOne(ky, dx[1]);

        iasgn_window_xy[ij] *= iasgn_window_xy[ij];
        iasgn_window_xy[ij] = 1.0 / iasgn_window_xy[ij];
    }

    for (size_t k = 0; k < ngrid_kz; ++k) {
        double kz = k * k_fund[2], window;
        iasgn_window_z[k] = smoothCICtoOne(kz, dx[2]);
        iasgn_window_z[k] *= iasgn_window_z[k];
        iasgn_window_z[k] = 1.0 / iasgn_window_z[k];
    }
}


RealField3D::RealField3D() : p_x2k(nullptr), p_k2x(nullptr) {
    _periodic_x = true;
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
    assert (p_x2k == nullptr);
    // p_x2k = nullptr; p_k2x = nullptr;
    _periodic_x = rhs._periodic_x;
    for (int axis = 0; axis < 3; ++axis) {
        ngrid[axis] = rhs.ngrid[axis];
        length[axis] = rhs.length[axis];
    }
    z0 = rhs.z0;
}


void RealField3D::construct(bool inp) {
    assert (p_x2k == nullptr);

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

    field_k = std::make_unique<std::complex<double>[]>(size_complex);

    if (_inplace) {
        field_x = reinterpret_cast<double*>(field_k.get());
        ngrid_z = ngrid[2] + 2 - (ngrid[2] % 2);
    }
    else {
        _field_x = std::make_unique<double[]>(size_real);
        field_x = _field_x.get();
        ngrid_z = ngrid[2];
    }
    
    p_x2k = fftw_plan_dft_r2c_3d(
        ngrid[0], ngrid[1], ngrid[2],
        field_x, reinterpret_cast<fftw_complex*>(field_k.get()),
        FFTW_MEASURE);

    p_k2x = fftw_plan_dft_c2r_3d(
        ngrid[0], ngrid[1], ngrid[2],
        reinterpret_cast<fftw_complex*>(field_k.get()), field_x,
        FFTW_MEASURE);

    _setAssignmentWindows();
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
        dn[axis] = ceil(radius / dx[axis] + 2) - 1;
        ntot *= 2 * dn[axis] + 1;
    }

    radius += 2.0 * dx[0];
    radius *= radius;

    neighbors.reserve(ntot);
    int _xi, _xf;
    if (_periodic_x) {
        _xi = -dn[0];  _xf = dn[0] + 1;
    } else {
        _xi = std::max(0, n[0] - dn[0]) - n[0];
        _xf = std::min(ngrid[0], n[0] + dn[0] + 1) - n[0];
    }

    for (int x = _xi; x < _xf; ++x) {
        double x2 = x * dx[0];  x2 *= x2;

        for (int y = std::max(0, n[1] - dn[1]);
             y < std::min(ngrid[1], n[1] + dn[1] + 1);
             ++y
        ) {
            double y2 = (y - n[1]) * dx[1];  y2 *= y2;

            for (int z = std::max(0, n[2] - dn[2]);
                 z < std::min(ngrid[2], n[2] + dn[2] + 1);
                 ++z
            ) {
                double z2 = (z - n[2]) * dx[2];  z2 *= z2;
                z2 += x2 + y2;

                if (z2 > radius)
                    continue;

                neighbors.push_back(getIndex(n[0] + x, y, z));
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


double RealField3D::interpolateLanczos(float coord[3]) const {
    /* LanczosK3 interpolation with a = 3
    f(x) = sum_{i = -a}^{a - 1} s_{n - i} L(d + i)
         = sum_{i = 1 - a}^{a} s_{n + i} L(d - i)
         = sum_{i = 0}^{2 * a - 1} s_{n + 1 - a + i} L(d + a - 1 - i)
    */
    #define A2_LANCZOS (2 * A_LANCZOS)
    int n[3];
    double r = 0, lanczos_kernel[3][A2_LANCZOS];

    for (int axis = 0; axis < 3; ++axis) {
        float d = coord[axis] / dx[axis];
        n[axis] = d;
        d -= n[axis];

        // Reverse the kernel
        // lanczos_kernel[axis][3 + i] = lanczos(d + i);
        double norm = 0.0;
        for (int i = -A_LANCZOS; i < A_LANCZOS; ++i) {
            int j = A_LANCZOS - 1 - i;
            lanczos_kernel[axis][j] = lanczos(d + i);
            norm += lanczos_kernel[axis][j];
        }

        for (int i = 0; i < A2_LANCZOS; ++i)
            lanczos_kernel[axis][i] /= norm;
    }

    #pragma omp parallel for reduction(+:r) num_threads(A_LANCZOS)
    for (int i = 0; i < A2_LANCZOS; ++i) {
        size_t idx0 = getIndex(
            n[0] - (A_LANCZOS - 1) + i,
            n[1] - (A_LANCZOS - 1),
            n[2] - (A_LANCZOS - 1));

        double temp[A2_LANCZOS];
        for (int j = 0; j < A2_LANCZOS; ++j)
            temp[j] = cblas_ddot(
                A2_LANCZOS, &lanczos_kernel[2][0], 1,
                field_x + idx0 + j * ngrid_z, 1);

        r += lanczos_kernel[0][i] * cblas_ddot(
            A2_LANCZOS, &lanczos_kernel[1][0], 1, temp, 1);
    }

    return r;
    #undef A2_LANCZOS
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
    if (_periodic_x) {
        if (n[0] >= ngrid[0])  n[0] -= ngrid[0];
        if (n[0] < 0)  n[0] += ngrid[0];
    }

    #ifdef ASSERT_MESH_IDX
    assert ((n[0] >= 0) && (n[0] < ngrid[0]));
    assert ((n[1] >= 0) && (n[1] < ngrid[1]));
    assert ((n[2] >= 0) && (n[2] < ngrid[2]));
    #endif

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


double RealField3D::getKperpFromIperp(size_t iperp, double &kx, double &ky)
const {
    int kn[2];
    kn[0] = iperp / ngrid[1];
    kn[1] = iperp % ngrid[1];

    if (kn[0] > (ngrid[0] / 2))
        kn[0] -= ngrid[0];

    kx = kn[0] * k_fund[0];

    if (kn[1] > (ngrid[1] / 2))
        kn[1] -= ngrid[1];

    ky = kn[1] * k_fund[1];

    return sqrt(kx * kx + ky * ky);
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

#undef A_LANCZOS
