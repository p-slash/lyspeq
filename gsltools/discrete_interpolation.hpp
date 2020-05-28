#ifndef DISCRETE_INTERPOLATION_H
#define DISCRETE_INTERPOLATION_H

// Stores a copy of y array.
// Assumes evenly spaced x.
// Linearly interpolates
class DiscreteInterpolation1D
{
    double x1, x2, dx, *y;
    long N;

public:
    DiscreteInterpolation1D(double x_start, double delta_x, const double *y_arr, long Nsize);
    ~DiscreteInterpolation1D();
    
    double evaluate(double x);
};

// Stores a copy of z array
// Assumes linearly spaced x, y
// Linearly interpolates
// Does not check for boundary!
class DiscreteInterpolation2D
{
    double  x1, x2, dx, y1, y2, dy, *z;
    long    Nx, Ny, size;

    long _getIndex(long nx, long ny);
public:
    DiscreteInterpolation2D(double x_start, double delta_x, double y_start, double delta_y,
        const double *z_arr, long Nxsize, long Nysize);
    ~DiscreteInterpolation2D();
    
    double evaluate(double x, double y);
};

#endif
