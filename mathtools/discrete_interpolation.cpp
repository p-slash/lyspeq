#include "mathtools/discrete_interpolation.hpp"
#include "mathtools/mathutils.hpp"
#include <algorithm>
#include <cassert>

constexpr double
cubic_notaknot_u[] = {
    6.000000000000000000e+00, 4.000000000000000000e+00,
    3.750000000000000000e+00, 3.733333333333333393e+00,
    3.732142857142857206e+00, 3.732057416267942518e+00,
    3.732051282051282115e+00, 3.732050841635176752e+00,
    3.732050810014727382e+00, 3.732050807744481613e+00,
    3.732050807581485330e+00, 3.732050807569782691e+00,
    3.732050807568942474e+00, 3.732050807568882078e+00,
    3.732050807568877637e+00, 3.732050807568877193e+00},
cubic_notaknot_p[] = {
    1.666666666666666574e-01, 2.500000000000000000e-01,
    2.666666666666666630e-01, 2.678571428571428492e-01,
    2.679425837320574266e-01, 2.679487179487179405e-01,
    2.679491583648230812e-01, 2.679491899852724512e-01,
    2.679491922555185535e-01, 2.679491924185148921e-01,
    2.679491924302174755e-01, 2.679491924310576922e-01,
    2.679491924311180329e-01, 2.679491924311223627e-01,
    2.679491924311226958e-01};

constexpr int cubic_u_size = 16, cubic_p_size = 15;

inline
bool allClose(const double *a, const double *b, int size)
{
    bool result = true;
    for (int i = 0; i < size; ++i)
        result &= isClose(a[i], b[i]);
    return result;
}

DiscreteInterpolation1D::DiscreteInterpolation1D(
        double x_start, double delta_x, int Nsize, double *y_arr, bool alloc
) : _alloc(alloc), x1(x_start), dx(delta_x), N(Nsize)
{
    x2 = x1 + dx * (N - 1);
    if (_alloc) {
        y = new double[N];
        if (y_arr != nullptr)
            std::copy(y_arr, y_arr + N, y);
    }
    else
        y = y_arr;
}


void DiscreteInterpolation1D::resetPointer(
        double x_start, double delta_x, int Nsize, double *y_arr
) {
    assert(!_alloc);
    x1 = x_start;
    dx = delta_x;
    N = Nsize;
    y = y_arr;
}

double DiscreteInterpolation1D::evaluate(double x) const
{
    double xx = (x - x1) / dx;
    int n = (int) xx;

    if (n < 0) n = 0;
    else if (n >= N - 1) n = N - 2;

    double dn = xx - n, y1 = y[n], y2 = y[n + 1];

    return y1 * (1 - dn) + y2 * dn;
}

void DiscreteInterpolation1D::evaluateVector(const double *xarr, int size, double *out)
{
    auto idx = std::make_unique<int[]>(size);

    #pragma omp simd
    for (int i = 0; i < size; ++i) {
        out[i] = (xarr[i] - x1) / dx;
        idx[i] = std::clamp(int(out[i]), 0, N - 2);
        out[i] -= idx[i];
    }

    #pragma omp simd
    for (int i = 0; i < size; ++i)
    {
        int n = idx[i];
        // out[i] = y[n] * (1 - out[i]) + y[n + 1] * out[i];
        out[i] = (y[n + 1] - y[n]) * out[i] + y[n];
    }
}

bool DiscreteInterpolation1D::operator==(const DiscreteInterpolation1D &rhs) const
{
    bool result = true;
    result &= isClose(x1, rhs.x1) && isClose(x2, rhs.x2) && isClose(dx, rhs.dx);
    result &= N == rhs.N;
    result &= allClose(y, rhs.y, N);
    return result;
}



DiscreteCubicInterpolation1D::DiscreteCubicInterpolation1D(
        double x_start, double delta_x, int Nsize, double *y_arr, bool alloc,
        bool notaknot
) : _alloc(alloc), _notaknot(notaknot), x1(x_start), dx(delta_x), N(Nsize)
{
    x2 = x1 + dx * (N - 1);
    if (_alloc) {
        y = new double[N];
        if (y_arr != nullptr)
            std::copy(y_arr, y_arr + N, y);
    }
    else
        y = y_arr;

    _y2p = std::make_unique<double[]>(N);

    if (_notaknot)
        construct_notaknot();
    else
        construct_natural();
}


void DiscreteCubicInterpolation1D::construct_natural() {
    const double sig = 0.5;
    auto u = std::make_unique<double[]>(N - 1);
    u[0] = 0;
    _y2p[0] = 0;
    _y2p[N - 1] = 0;

    for (int i = 1; i < N - 1; ++i) {
        double p = sig * _y2p[i - 1] + 2.0;
        _y2p[i] = -sig / p;
        u[i] = (y[i - 1] - 2 * y[i] + y[i + 1]) / dx;
        u[i] = (- sig * u[i - 1] + 3 * u[i] / dx) / p;
    }

    for (int i = N - 2; i > 0; --i)
        _y2p[i] = _y2p[i] * _y2p[i + 1] + u[i];
}


void DiscreteCubicInterpolation1D::construct_notaknot() {
    for (int i = 1; i < N - 1; ++i)
        _y2p[i] = y[i - 1] - 2 * y[i] + y[i + 1];

    for (int i = 1; i < std::min(N - 3, cubic_p_size + 1); ++i)
        _y2p[i + 1] -= cubic_notaknot_p[i - 1] * _y2p[i];
    for (int i = cubic_p_size + 1; i < N - 3; ++i)
        _y2p[i + 1] -= cubic_notaknot_p[cubic_p_size - 1] * _y2p[i];

    _y2p[N - 2] /= cubic_notaknot_u[0];
    for (int i = N - 4; i > cubic_u_size - 1; --i)
        _y2p[i + 1] = (_y2p[i + 1] - _y2p[i + 2]) / cubic_notaknot_u[cubic_u_size - 1];
    for (int i = std::min(N - 4, cubic_u_size - 1); i > 0; --i)
        _y2p[i + 1] = (_y2p[i + 1] - _y2p[i + 2]) / cubic_notaknot_u[i];

    _y2p[1] /= cubic_notaknot_u[0];
    _y2p[0] = 2 * _y2p[1] - _y2p[2];
    _y2p[N - 1] = 2 * _y2p[N - 2] - _y2p[N - 3];
    // cblas_dscal(N, 6.0 / (dx * dx), _y2p.get(), 1);
}


void DiscreteCubicInterpolation1D::resetPointer(
        double x_start, double delta_x, int Nsize, double *y_arr
) {
    assert(!_alloc);
    x1 = x_start;
    dx = delta_x;
    N = Nsize;
    y = y_arr;
    reconstruct();
}

double DiscreteCubicInterpolation1D::evaluate(double x) const
{
    double xx = (x - x1) / dx;
    int n = (int) xx;

    if (n < 0) n = 0;
    else if (n >= N - 1) n = N - 2;

    double dn = xx - n, y1 = y[n], y2 = y[n + 1],
           ypp1 = _y2p[n], ypp2 = _y2p[n + 1];

    xx = (1 - dn) * y1 + dn * y2
         - ((2 - dn) * ypp1 + (dn + 1) * ypp2) * (1 - dn) * dn;
    return xx;
}


bool DiscreteCubicInterpolation1D::operator==(const DiscreteCubicInterpolation1D &rhs) const
{
    bool result = true;
    result &= isClose(x1, rhs.x1) && isClose(x2, rhs.x2) && isClose(dx, rhs.dx);
    result &= N == rhs.N;
    result &= allClose(y, rhs.y, N);
    return result;
}



DiscreteInterpolation2D::DiscreteInterpolation2D(
        double x_start, double delta_x, double y_start, double delta_y,
        const double *z_arr, int Nxsize, int Nysize
) : x1(x_start), dx(delta_x), y1(y_start), dy(delta_y), Nx(Nxsize), Ny(Nysize)
{
    x2 = x1 + dx * (Nx-1);
    y2 = y1 + dy * (Ny-1);

    size = Nx*Ny;
    z = new double[size];
    std::copy(z_arr, z_arr+size, z);
}

double DiscreteInterpolation2D::evaluate(double x, double y) const
{
    double xx = (x - x1) / dx, yy = (y - y1) / dy;
    int nx = (int) xx, ny = (int) yy;

    if (nx < 0) nx = 0;
    else if (nx >= Nx - 1) nx = Nx - 2;

    if (ny < 0) ny = 0;
    else if (ny >= Ny - 1) ny = Ny - 2;

    size_t ind = _getIndex(nx, ny);
    double dnx = xx - nx, dny = yy - ny;

    double result =
        z[ind] * (1 - dnx) * (1 - dny)
        + z[ind + 1] * (dnx) * (1 - dny)
        + z[ind + Nx] * (1 - dnx) * (dny)
        + z[ind + Nx + 1] * (dnx) * (dny);

    return result;
}

bool DiscreteInterpolation2D::operator==(const DiscreteInterpolation2D &rhs) const
{
    bool result = true;
    result &= isClose(x1, rhs.x1) && isClose(x2, rhs.x2) && isClose(dx, rhs.dx);
    result &= isClose(y1, rhs.y1) && isClose(y2, rhs.y2) && isClose(dy, rhs.dy);
    result &= (Nx == rhs.Nx) && (Ny == rhs.Ny);
    result &= allClose(z, rhs.z, size);
    return result;
}



DiscreteBicubicSpline::DiscreteBicubicSpline(
        double x_start, double delta_x, double y_start, double delta_y,
        const double *z_arr, int Nxsize, int Nysize, int halfn
) : x1(x_start), dx(delta_x), y1(y_start), dy(delta_y), Nx(Nxsize), Ny(Nysize), halfm(halfn)
{
    halfn = std::max(2, halfn);
    m = std::min(2 * halfn, Ny);  halfm = m / 2;  m = halfm * 2;
    _spl_local = std::make_unique<DiscreteCubicInterpolation1D>(y1, dy, m);
    _zy = _spl_local->get();

    z = std::make_unique<double[]>(Nx * Ny);
    std::copy_n(z_arr, Nx * Ny, z.get());
    _spls_y.reserve(Ny);
    for (int i = 0; i < Ny; ++i)
        _spls_y.push_back(
            std::make_unique<DiscreteCubicInterpolation1D>(
                x1, dx, Nx, z.get() + i * Nx, false
            )
        );
}


double DiscreteBicubicSpline::evaluate(double x, double y) {
    double yy = (y - y1) / dy;
    int i1 = yy - (halfm - 1);

    if (i1 < 0)
        i1 = 0;

    if (i1 > (Ny - m))
        i1 = Ny - m;

    for (int i = 0; i < m; ++i)
        _zy[i] = _spls_y[i + i1]->evaluate(x);

    _spl_local->construct_notaknot();
    return _spl_local->evaluate(y - i1 * dy);
}

double _hermiteSplineCasteljau(double beta[], double t) {
    double t1 = 1.0 - t;
    beta[0] = (beta[2] - beta[0]) / 6.0 + beta[1];
    std::swap(beta[0], beta[1]);
    beta[3] = -(beta[3] - beta[1]) / 6.0 + beta[2];
    std::swap(beta[2], beta[3]);

    // de Casteljau's algorithm
    for (int j = 1; j < 4; ++j)
        for (int i = 0; i < 4 - j; ++i)
            beta[i] = beta[i] * t1 + beta[i + 1] * t;
    return beta[0];
}


double DiscreteBicubicSpline::evaluateHermiteY(double x, double y) {
    /* Cubic Hermite spline in y direction using de Casteljau's algorithm
       and the Bernstein form. */
    double ydata[4];
    double yy = (y - y1) / dy;
    int i0 = yy;  i0 = std::clamp(i0, 1, Ny - 3);  yy -= i0;  --i0;

    for (int i = 0; i < 4; ++i)
        ydata[i] = _spls_y[i + i0]->evaluate(x);

    return _hermiteSplineCasteljau(ydata, yy);
}


double DiscreteBicubicSpline::evaluateHermite2(double x, double y) {
    double xdata[4], ydata[4];
    double xx = (x - x1) / dx, yy = (y - y1) / dy;
    int j0 = xx;  j0 = std::clamp(j0, 1, Nx - 3);  xx -= j0;  --j0;
    int i0 = yy;  i0 = std::clamp(i0, 1, Ny - 3);  yy -= i0;  --i0;

    const double *zptr = z.get() + j0 + i0 * Nx;
    for (int i = 0; i < 4; ++i) {
        std::copy_n(zptr + i * Nx, 4, xdata);
        // std::clamp(j + j0, 0, Nx - 1), std::clamp(i + i0, 0, Ny - 1))
        ydata[i] = _hermiteSplineCasteljau(xdata, xx);
    }

    return _hermiteSplineCasteljau(ydata, yy);
}


void DiscreteBicubicSpline::trim(double xmax, double ymax) {
    int Nxtrim = ceil((xmax - x1) / dx), Nytrim = ceil((ymax - y1) / dy);
    auto trimmed_z = std::make_unique<double[]>(Nxtrim * Nytrim);
    _spls_y.clear();
    for (int i = 0; i < Nytrim; ++i) {
        for (int j = 0; j < Nxtrim; ++j)
            trimmed_z[j + Nxtrim * i] = z[_getIndex(j, i)];

        _spls_y.push_back(
            std::make_unique<DiscreteCubicInterpolation1D>(
                x1, dx, Nxtrim, trimmed_z.get() + i * Nxtrim, false)
        );
    }

    if (m > Nytrim) {
        halfm = Nytrim / 2;  m = halfm * 2;
        _spl_local = std::make_unique<DiscreteCubicInterpolation1D>(y1, dy, m);
        _zy = _spl_local->get();
    }

    z.swap(trimmed_z);  Nx = Nxtrim;  Ny = Nytrim;
}



