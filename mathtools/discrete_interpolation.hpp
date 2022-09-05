#ifndef DISCRETE_INTERPOLATION_H
#define DISCRETE_INTERPOLATION_H

#include <memory>

// Stores a copy of y array.
// Assumes evenly spaced x.
// Linearly interpolates
// x equals to the boundary when it exceeds in either end
class DiscreteInterpolation1D
{
    double x1, x2, dx;
    double *y;
    long N;

    void _limitBoundary(double &x);
public:
    DiscreteInterpolation1D(double x_start, double delta_x, const double *y_arr, long Nsize);
    ~DiscreteInterpolation1D() { delete [] y; };
    bool operator==(const DiscreteInterpolation1D &rhs) const;
    bool operator!=(const DiscreteInterpolation1D &rhs) const
    { return ! (*this==rhs); };

    double evaluate(double x);
};

// Stores a copy of z array
// Assumes linearly spaced x, y
// Linearly interpolates
// x and y equal to the boundary when either exceeds in either end
class DiscreteInterpolation2D
{
    double  x1, x2, dx, y1, y2, dy;
    double *z;
    long    Nx, Ny, size;

    long _getIndex(long nx, long ny);
    void _limitBoundary(double &x, double &y);
public:
    DiscreteInterpolation2D(double x_start, double delta_x, double y_start, double delta_y,
        const double *z_arr, long Nxsize, long Nysize);
    ~DiscreteInterpolation2D() { delete [] z; };
    bool operator==(const DiscreteInterpolation2D &rhs) const;
    bool operator!=(const DiscreteInterpolation2D &rhs) const
    { return ! (*this==rhs); };

    double evaluate(double x, double y);
};

typedef std::shared_ptr<DiscreteInterpolation1D> shared_interp_1d;
typedef std::shared_ptr<DiscreteInterpolation2D> shared_interp_2d;

#endif
