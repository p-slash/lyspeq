#include "mathtools/interpolation_2d.hpp"

#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_errno.h>

#include <cstdio>
#include <stdexcept>

Interpolation2D::Interpolation2D(
    GSL_2D_INTERPOLATION_TYPE interp_type,
    const double *x, const double *y, const double *z,
    long x_size, long y_size)
{
    lowest_x  = x[0];
    highest_x = x[x_size - 1];

    lowest_y = y[0];
    highest_y= y[y_size - 1];

    x_accelerator = gsl_interp_accel_alloc();
    y_accelerator = gsl_interp_accel_alloc();

    if (interp_type == GSL_BILINEAR_INTERPOLATION)
        spline = gsl_spline2d_alloc(gsl_interp2d_bilinear, x_size, y_size);
    else if (interp_type == GSL_BICUBIC_INTERPOLATION)
        spline = gsl_spline2d_alloc(gsl_interp2d_bicubic, x_size, y_size);
    else
        throw std::runtime_error("Interpolation2D::GSL_2D_INTERPOLATION_TYPE");
    
    gsl_spline2d_init(spline, x, y, z, x_size, y_size);
}

Interpolation2D::~Interpolation2D()
{
    gsl_spline2d_free(spline);
    gsl_interp_accel_free(x_accelerator);
    gsl_interp_accel_free(y_accelerator);
}

Interpolation2D::Interpolation2D(const Interpolation2D &itp2d)
{
    lowest_x  = itp2d.lowest_x;
    lowest_y  = itp2d.lowest_y;
    highest_x = itp2d.highest_x;
    highest_y = itp2d.highest_y;

    x_accelerator = gsl_interp_accel_alloc();
    y_accelerator = gsl_interp_accel_alloc();
    
    long x_size = itp2d.spline->interp_object.xsize;
    long y_size = itp2d.spline->interp_object.ysize;
    // Copied from source code of GSL 2.5
    spline = gsl_spline2d_alloc(itp2d.spline->interp_object.type,
        x_size, y_size);
    
    gsl_spline2d_init(spline, itp2d.spline->xarr,
        itp2d.spline->yarr, itp2d.spline->zarr,
        x_size, y_size);
}

double Interpolation2D::evaluate(double x, double y) const
{
    int status;
    double result;

    status = gsl_spline2d_eval_e(spline, x, y, x_accelerator,
        y_accelerator, &result);

    if (!status)
        return result;

    if (status == GSL_EDOM)
        return 0; // extrapolate(x, y, &gsl_spline2d_eval);
    else 
        throw std::runtime_error(gsl_strerror(status));
}

double Interpolation2D::derivate_x(double x, double y) const
{
    int status;
    double result;
    
    status = gsl_spline2d_eval_deriv_x_e(spline, x, y, x_accelerator,
        y_accelerator, &result);

    if (!status)
        return result;

    if (status == GSL_EDOM)
        return 0; // extrapolate(x, y, &gsl_spline2d_eval_deriv_x);
    else 
        throw std::runtime_error(gsl_strerror(status));
}

double Interpolation2D::derivate_y(double x, double y) const
{
    int status;
    double result;
    
    status = gsl_spline2d_eval_deriv_y_e(spline, x, y, x_accelerator,
        y_accelerator, &result);

    if (!status)
        return result;

    if (status == GSL_EDOM)
        return 0; // extrapolate(x, y, &gsl_spline2d_eval_deriv_y);
    else 
        throw std::runtime_error(gsl_strerror(status));
}

double Interpolation2D::extrapolate(
    double x, double y, 
    double (*func)(
        const gsl_spline2d*,
        const double,
        const double,
        gsl_interp_accel*,
        gsl_interp_accel*)
    ) const
{
    double x_eval = x, y_eval = y;

    if (x < lowest_x)
        x_eval = lowest_x;
    else if (x > highest_x)
        x_eval = highest_x;

    if (y < lowest_y)
        y_eval = lowest_y;
    else if (y > highest_y)
        y_eval = highest_y;

    return func(spline, x_eval, y_eval, x_accelerator, y_accelerator);
}







