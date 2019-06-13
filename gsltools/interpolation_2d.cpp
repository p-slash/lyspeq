#include "gsltools/interpolation_2d.hpp"

#include <gsl/gsl_interp2d.h>

#include <cstdio>

Interpolation2D::Interpolation2D(   GSL_2D_INTERPOLATION_TYPE interp_type, \
                                    const double *x, const double *y, const double *z, \
                                    long x_size, long y_size)
{
    lowest_x  = x[0];
    highest_x = x[x_size - 1];

    lowest_y = y[0];
    highest_y= y[y_size - 1];

    x_accelerator = gsl_interp_accel_alloc();
    y_accelerator = gsl_interp_accel_alloc();

    if      (interp_type == GSL_BILINEAR_INTERPOLATION)     spline = gsl_spline2d_alloc(gsl_interp2d_bilinear, x_size, y_size);
    else if (interp_type == GSL_BICUBIC_INTERPOLATION)      spline = gsl_spline2d_alloc(gsl_interp2d_bicubic, x_size, y_size);
    else                                                    throw "2D_INTERP_TYPE_ERROR";
    
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
    spline = gsl_spline2d_alloc(itp2d.spline->interp_object.type, x_size, y_size);
    
    gsl_spline2d_init(spline, itp2d.spline->xarr, itp2d.spline->yarr, itp2d.spline->zarr, x_size, y_size);
}

double Interpolation2D::evaluate(double x, double y) const
{
    double x_eval = x, y_eval = y;

    if (x < lowest_x)
    {
        fprintf(stderr, "WARNING: Extrapolating 2D interpolation for x=%e is smaller than lowest x=%e!\n", x, lowest_x);
        x_eval = lowest_x;
    }
    else if (x > highest_x)
    {
        fprintf(stderr, "WARNING: Extrapolating 2D interpolation for x=%e is larger than highest x=%e!\n", x, highest_x);
        x_eval = highest_x;
    }

    if (y < lowest_y)
    {
        fprintf(stderr, "WARNING: Extrapolating 2D interpolation for y=%e is smaller than lowest y=%e!\n", y, lowest_y);
        y_eval = lowest_y;
    }
    else if (y > highest_y)
    {
        fprintf(stderr, "WARNING: Extrapolating 2D interpolation for y=%e is larger than highest y=%e!\n", y, highest_y);
        y_eval = highest_y;
    }

    return gsl_spline2d_eval(spline, x_eval, y_eval, x_accelerator, y_accelerator);
}
