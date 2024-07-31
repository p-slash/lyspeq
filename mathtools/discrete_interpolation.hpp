#ifndef DISCRETE_INTERPOLATION_H
#define DISCRETE_INTERPOLATION_H

#include <memory>

// Stores a copy of y array.
// Assumes evenly spaced x.
// Linearly interpolates
// Out of bound values are extrapolated
class DiscreteInterpolation1D
{
    bool _alloc;
    double x1, x2, dx;
    double *y;
    int N;

public:
    DiscreteInterpolation1D(
        double x_start, double delta_x, int Nsize,
        double *y_arr=nullptr, bool alloc=true);
    ~DiscreteInterpolation1D() { if (_alloc) delete [] y; };
    void resetPointer(double x_start, double delta_x, int Nsize, double *y_arr);
    bool operator==(const DiscreteInterpolation1D &rhs) const;
    bool operator!=(const DiscreteInterpolation1D &rhs) const
    { return ! (*this==rhs); };

    double evaluate(double x);
    void evaluateVector(const double *xarr, int size, double *out);
    double* get() const { return y; }
    int size() const { return N; }
};


class DiscreteCubicInterpolation1D {
    /* Stores a copy of y array.
       Assumes evenly spaced x.
       cubic interpolation
       Out of bound values are extrapolated
       Natural (second derivatives are zero) or Not-A-Knot boundary conditions
    */

    bool _alloc, _notaknot;
    double x1, x2, dx;
    double *y;
    int N;

    std::unique_ptr<double[]> _y2p;
    void construct_natural();
    void construct_notaknot();

public:
    DiscreteCubicInterpolation1D(
        double x_start, double delta_x, int Nsize,
        double *y_arr=nullptr, bool alloc=true, bool notaknot=true);
    ~DiscreteCubicInterpolation1D() { if (_alloc) delete [] y; };
    void resetPointer(double x_start, double delta_x, int Nsize, double *y_arr);
    bool operator==(const DiscreteCubicInterpolation1D &rhs) const;
    bool operator!=(const DiscreteCubicInterpolation1D &rhs) const
    { return ! (*this==rhs); };

    double evaluate(double x);
    double* get() const { return y; }
    int size() const { return N; }
};

// Stores a copy of z array
// Assumes linearly spaced x, y
// Linearly interpolates
// x and y equal to the boundary when either exceeds in either end
class DiscreteInterpolation2D
{
    double x1, x2, dx, y1, y2, dy;
    double *z;
    int Nx, Ny, size;

    inline
    int _getIndex(int nx, int ny) {  return nx + Nx * ny; };
public:
    DiscreteInterpolation2D(
        double x_start, double delta_x, double y_start, double delta_y,
        const double *z_arr, int Nxsize, int Nysize);
    ~DiscreteInterpolation2D() { delete [] z; };
    bool operator==(const DiscreteInterpolation2D &rhs) const;
    bool operator!=(const DiscreteInterpolation2D &rhs) const
    { return ! (*this==rhs); };

    double evaluate(double x, double y);
    inline double getX1() const { return x1; }
    inline double getY1() const { return y1; }
};

typedef std::shared_ptr<DiscreteInterpolation1D> shared_interp_1d;
typedef std::shared_ptr<DiscreteInterpolation2D> shared_interp_2d;

#endif
