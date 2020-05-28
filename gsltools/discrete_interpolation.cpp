#include "discrete_interpolation.hpp"
#include <algorithm>

DiscreteInterpolation1D::DiscreteInterpolation1D(double x_start, double delta_x, const double *y_arr, long Nsize)
: x1(x_start), dx(delta_x), N(Nsize)
{
    x2 = x1 + dx * (N-1);
    y = new double[N];
    std::copy(y_arr, y_arr+N, y);
}

DiscreteInterpolation1D::~DiscreteInterpolation1D()
{
    delete [] y;
}

double DiscreteInterpolation1D::evaluate(double x)
{
    if (x > x2) return y[N-1];

    long n = (x-x1)/dx;
    double xn = x1 + n*dx, y1 = y[n], y2=y[n+1];

    return y1 + (y2-y1)*(x-xn)/dx;
}

DiscreteInterpolation2D::DiscreteInterpolation2D(double x_start, double delta_x, double y_start, double delta_y,
        const double *z_arr, long Nxsize, long Nysize)
: x1(x_start), dx(delta_x), y1(y_start), Nx(Nxsize), Ny(Nysize)
{
    x2 = x1 + dx * (Nx-1);
    y2 = y1 + dy * (Ny-1);

    size = Nx*Ny;
    z = new double[size];
    std::copy(z_arr, z_arr+size, z);
}

DiscreteInterpolation2D::~DiscreteInterpolation2D()
{
    delete [] z;
}

long DiscreteInterpolation2D::_getIndex(long nx, long ny)
{
    return nx + Nx*ny;
}

double DiscreteInterpolation2D::evaluate(double x, double y)
{
    long nx = (x-x1)/dx;
    long ny = (y-y1)/dy;

    double dnx = (x-x1)/dx - nx;
    double dny = (y-y1)/dy - ny;
    

    double result = z[_getIndex(nx, ny)] * (1-dnx) * (1-dny)
    + z[_getIndex(nx+1, ny)] * (dnx) * (1-dny)
    + z[_getIndex(nx, ny+1)] * (1-dnx) * (dny)
    + z[_getIndex(nx+1, ny+1)] * (dnx) * (dny);

    return result;
}










