#ifndef INTERPOLATION_H
#define INTERPOLATION_H

// prevent gsl_cblas.h from being included
#define  __GSL_CBLAS_H__

#ifdef USE_MKL_CBLAS
#include "mkl_cblas.h"
#else
#include "cblas.h"
#endif
#include <gsl/gsl_spline.h>

enum GSL_INTERPOLATION_TYPE
{
    GSL_LINEAR_INTERPOLATION,
    GSL_CUBIC_INTERPOLATION
};

// Intepolation for given x and y arrays with size many elements.
// Stores a copy of x and y arrays in spline.
// Accelerator is NOT thread safe. Create local copies.

// Example for linear interpolation:
//     double x[10], y[10];
//     Interpolation tmp_interp(GSL_LINEAR_INTERPOLATION, x, y, 10);
//     double r = tmp_interp.evaluate((x[5]+x[6])/2.);
class Interpolation {   
    gsl_interp_accel *accelerator;
    gsl_spline *spline;
    
public:
    double lowest_x, highest_x;
    
    Interpolation(
        GSL_INTERPOLATION_TYPE interp_type,
        const double *x, const double *y, long size);
    Interpolation(const Interpolation &itp);

    ~Interpolation() {
        gsl_spline_free(spline);
        gsl_interp_accel_free(accelerator);
    }

    void reset(const double *yp);

    double evaluate(double x) const;
    double derivative(double x) const;
};

#endif
