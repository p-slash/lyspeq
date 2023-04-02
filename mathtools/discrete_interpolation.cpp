#include "discrete_interpolation.hpp"
#include <algorithm>
#include <cmath>

// inlines solve duplicate symbol error in tests/test_utils.cpp
inline
bool isClose(double a, double b, double relerr=1e-5, double abserr=1e-8)
{
    double mag = std::max(fabs(a),fabs(b));
    return fabs(a-b) < (abserr + relerr * mag);
}

inline
bool allClose(const double *a, const double *b, int size)
{
    bool result = true;
    for (int i = 0; i < size; ++i)
        result &= isClose(a[i], b[i]);
    return result;
}

DiscreteInterpolation1D::DiscreteInterpolation1D(
        double x_start, double delta_x, const double *y_arr, int Nsize
) : x1(x_start), dx(delta_x), N(Nsize)
{
    x2 = x1 + dx * (N - 1);
    y = new double[N];
    std::copy(y_arr, y_arr + N, y);
}

double DiscreteInterpolation1D::evaluate(double x)
{
    double xx = (x - x1) / dx;
    int n = (int) xx;

    if (n < 0) n = 0;
    else if (n >= N - 1) n = N - 2;

    double dn = xx - n, y1 = y[n], y2 = y[n + 1];

    return y1 * (1 - dn) + y2 * dn;
}

bool DiscreteInterpolation1D::operator==(const DiscreteInterpolation1D &rhs) const
{
    bool result = true;
    result &= isClose(x1, rhs.x1) & isClose(x2, rhs.x2) & isClose(dx, rhs.dx);
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

double DiscreteInterpolation2D::evaluate(double x, double y)
{
    double xx = (x - x1) / dx, yy = (y - y1) / dy;
    int nx = (int) xx, ny = (int) yy;

    if (nx < 0) nx = 0;
    else if (nx >= Nx - 1) nx = Nx - 2;

    if (ny < 0) ny = 0;
    else if (ny >= Ny - 1) ny = Ny - 2;

    int ind = _getIndex(nx, ny);
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
    result &= isClose(x1, rhs.x1) & isClose(x2, rhs.x2) & isClose(dx, rhs.dx);
    result &= isClose(y1, rhs.y1) & isClose(y2, rhs.y2) & isClose(dy, rhs.dy);
    result &= (Nx == rhs.Nx) & (Ny == rhs.Ny);
    result &= allClose(z, rhs.z, size);
    return result;
}









