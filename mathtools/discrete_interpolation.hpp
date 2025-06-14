#ifndef DISCRETE_INTERPOLATION_H
#define DISCRETE_INTERPOLATION_H

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

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

    double evaluate(double x) const;
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

public:
    void construct_natural();
    void construct_notaknot();

    DiscreteCubicInterpolation1D(
        double x_start, double delta_x, int Nsize,
        double *y_arr=nullptr, bool alloc=true, bool notaknot=true);
    ~DiscreteCubicInterpolation1D() { if (_alloc) delete [] y; };
    void resetPointer(double x_start, double delta_x, int Nsize, double *y_arr);
    void reconstruct() {
        if (_notaknot)
            construct_notaknot();
        else
            construct_natural();
    }
    bool operator==(const DiscreteCubicInterpolation1D &rhs) const;
    bool operator!=(const DiscreteCubicInterpolation1D &rhs) const
    { return ! (*this==rhs); };

    double clamp(double x) const { return std::clamp(x, x1, x2); }
    double evaluate(double x) const;
    double* get() const { return y; }
    int size() const { return N; }
    inline double getX1() const { return x1; }
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
    size_t _getIndex(int nx, int ny) const {  return nx + Nx * ny; };
public:
    DiscreteInterpolation2D(
        double x_start, double delta_x, double y_start, double delta_y,
        const double *z_arr, int Nxsize, int Nysize);
    ~DiscreteInterpolation2D() { delete [] z; };
    bool operator==(const DiscreteInterpolation2D &rhs) const;
    bool operator!=(const DiscreteInterpolation2D &rhs) const
    { return ! (*this==rhs); };

    double evaluate(double x, double y) const;
    inline double evaluateHermite2(double x, double y) const {
        return evaluate(x, y);
    }
    inline double getX1() const { return x1; }
    inline double getY1() const { return y1; }

    template<class T>
    std::unique_ptr<T> get1dSliceX(double y) const {
        auto sl = std::make_unique<double[]>(Nx);
        for (int i = 0; i < Nx; ++i)
            sl[i] = evaluate(x1 + i * dx, y);

        return std::make_unique<T>(x1, dx, Nx, sl.get());
    }

    void applyFunction(std::function<double(double, double)> &&func) {
        for (int i = 0; i < Ny; ++i) {
            double y = y1 + i * dy;
            for (int j = 0; j < Nx; ++j) {
                double x = x1 + j * dx;
                z[_getIndex(j, i)] *= func(x, y);
            }
        }
    }
};


class DiscreteBicubicSpline {
    double x1, x2, dx, y1, y2, dy;
    int Nx, Ny, halfm, m;

    std::unique_ptr<DiscreteCubicInterpolation1D> _spl_local;
    double *_zy;

    std::unique_ptr<double[]> z;
    std::vector<std::unique_ptr<DiscreteCubicInterpolation1D>> _spls_y;

    inline
    size_t _getIndex(int nx, int ny) const {  return nx + Nx * ny; };

public:
    DiscreteBicubicSpline(
        double x_start, double delta_x, double y_start, double delta_y,
        const double *z_arr, int Nxsize, int Nysize, int halfn=5);

    inline double getDx() const { return dx; }
    inline double getDy() const { return dy; }
    inline double getX1() const { return x1; }
    inline double getY1() const { return y1; }
    inline double getX2() const { return x2; }
    inline double getY2() const { return y2; }

    /* Construct cubic spline in y direction */
    double evaluate(double x, double y);
    /* Use Hermite spline in y direction */
    double evaluateHermiteY(double x, double y);
    /* Use Hermite spline in x & y direction */
    double evaluateHermite2(double x, double y) const;
    void clamp(double &x, double &y) const {
        x = std::clamp(x, x1, x2);  y = std::clamp(y, y1, y2);
    }

    template<class T>
    std::unique_ptr<T> get1dSliceX(double y) {
        auto sl = std::make_unique<double[]>(Nx);
        for (int i = 0; i < Nx; ++i)
            sl[i] = evaluate(x1 + i * dx, y);

        return std::make_unique<T>(x1, dx, Nx, sl.get());
    }

    void applyFunction(std::function<double(double, double)> &&func) {
        for (int i = 0; i < Ny; ++i) {
            double y = y1 + i * dy;
            for (int j = 0; j < Nx; ++j) {
                double x = x1 + j * dx;
                z[_getIndex(j, i)] *= func(x, y);
            }
            _spls_y[i]->reconstruct();
        }
    }

    void trim(double xmax, double ymax);
};


template<class T1, class T2>
class DiscreteLogInterpolation2D {
    /* T1: 1D interpolator class DiscreteInterpolation1D or
           DiscreteCubicInterpolation1D */
    /* T2: 2D interpolator class DiscreteInterpolation2D or
           DiscreteBicubicSpline */
public:
    /* Note interp_2d must be defined y, x due to legacy code */
    std::unique_ptr<T2> interp_2d;
    std::unique_ptr<T1> interp_x, interp_y;
    virtual double evaluate(double x, double y) const {
        if ((x == 0) && (y == 0))
            return 0;
        else if (x == 0)
            return interp_y->evaluate(log(y));
        else if (y == 0)
            return interp_x->evaluate(log(x));

        return interp_2d->evaluateHermite2(log(y), log(x));
    }

    virtual double evaluateSqrt(double x, double y) const {
        if ((x == 0) && (y == 0))
            return 0;
        else if (x == 0)
            return sqrt(interp_y->evaluate(log(y)));
        else if (y == 0)
            return sqrt(interp_x->evaluate(log(x)));

        return sqrt(interp_2d->evaluateHermite2(log(y), log(x)));
    }

    void setInterpX(double x1, double dx, int N, double *y) {
        interp_x = std::make_unique<T1>(x1, dx, N, y);
    }

    void setInterpY(double x1, double dx, int N, double *y) {
        interp_y = std::make_unique<T1>(x1, dx, N, y);
    }

    void setInterp2D(
            double x1, double dx, double y1, double dy, double *z,
            int Nx, int Ny
    ) {
        interp_2d = std::make_unique<T2>(x1, dx, y1, dy, z, Nx, Ny);
    }
};


template<class T1, class T2>
class DiscreteLogLogInterpolation2D : public DiscreteLogInterpolation2D<T1, T2> {
    /* T1: 1D interpolator class DiscreteInterpolation1D or
           DiscreteCubicInterpolation1D */
    /* T2: 2D interpolator class DiscreteInterpolation2D or
           DiscreteBicubicSpline */
public:
    /* Note interp_2d must be defined y, x due to legacy code */
    double evaluate(double x, double y) const override {
        if ((x == 0) && (y == 0))
            return 0;
        else if (x == 0)
            return exp(this->interp_y->evaluate(log(y)));
        else if (y == 0)
            return exp(this->interp_x->evaluate(log(x)));

        return exp(this->interp_2d->evaluateHermite2(log(y), log(x)));
    }

    double evaluateSqrt(double x, double y) const override {
        if ((x == 0) && (y == 0))
            return 0;
        else if (x == 0)
            return exp(0.5 * this->interp_y->evaluate(log(y)));
        else if (y == 0)
            return exp(0.5 * this->interp_x->evaluate(log(x)));

        return exp(0.5 * this->interp_2d->evaluateHermite2(log(y), log(x)));
    }
};


typedef std::shared_ptr<DiscreteInterpolation1D> shared_interp_1d;
typedef std::shared_ptr<DiscreteInterpolation2D> shared_interp_2d;

#endif
