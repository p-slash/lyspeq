#include "discrete_interpolation.hpp"
#include <algorithm>

DiscreteInterpolation1D::DiscreteInterpolation1D(double x_start, double delta_x, const double *y_arr, long Nsize)
: x1(x_start), dx(delta_x), N(Nsize)
{
    x2 = x1 + dx * (N-1);
    y = new double[N];
    std::copy(y_arr, y_arr+N, y);
}

void DiscreteInterpolation1D::_limitBoundary(double &x)
{
    if (x >= x2)        x = x2 - 1e-6;
    else if (x < x1)    x = x1;
}

double DiscreteInterpolation1D::evaluate(double x)
{
    _limitBoundary(x);

    long n = (x-x1)/dx;
    double dn = (x-x1)/dx - n, y1 = y[n], y2=y[n+1];

    return y1*(1-dn) + y2*dn;
}

DiscreteInterpolation2D::DiscreteInterpolation2D(double x_start, double delta_x, double y_start, double delta_y,
    const double *z_arr, long Nxsize, long Nysize)
: x1(x_start), dx(delta_x), y1(y_start), dy(delta_y), Nx(Nxsize), Ny(Nysize)
{
    x2 = x1 + dx * (Nx-1);
    y2 = y1 + dy * (Ny-1);

    size = Nx*Ny;
    z = new double[size];
    std::copy(z_arr, z_arr+size, z);
}

long DiscreteInterpolation2D::_getIndex(long nx, long ny)
{
    return nx + Nx*ny;
}

void DiscreteInterpolation2D::_limitBoundary(double &x, double &y)
{
    if (x >= x2)        x = x2 - 1e-6;
    else if (x < x1)    x = x1;

    if (y >= y2)        y = y2 - 1e-6;
    else if (y < y1)    y = y1;
}

double DiscreteInterpolation2D::evaluate(double x, double y)
{
    _limitBoundary(x, y);

    long nx = (x-x1)/dx;
    long ny = (y-y1)/dy;
    long ind = _getIndex(nx, ny);

    double dnx = (x-x1)/dx - nx;
    double dny = (y-y1)/dy - ny;

    double result = z[ind] * (1-dnx) * (1-dny)
    + z[ind+1] * (dnx) * (1-dny)
    + z[ind+Nx] * (1-dnx) * (dny)
    + z[ind+Nx+1] * (dnx) * (dny);

    return result;
}










