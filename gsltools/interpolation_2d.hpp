#ifndef INTERPOLATION_2D_H
#define INTERPOLATION_2D_H

#include <gsl/gsl_spline2d.h>

enum GSL_2D_INTERPOLATION_TYPE
{
	GSL_BILINEAR_INTERPOLATION,
	GSL_BICUBIC_INTERPOLATION
};

// Intepolation for given x[i], y[j] and z[j * xsize + i] arrays with x_size and y_size many elements.
// Stores a copy for each x, y and z array in spline.
// Accelerators are NOT thread safe. Create local copies.

// Example for linear interpolation:
//     double x[10], y[10], z[100];
//     Interpolation2D tmp_interp(GSL_BILINEAR_INTERPOLATION, x, y, z, 10, 10);
//     double r = tmp_interp.evaluate((x[5]+x[6])/2., (y[5]+y[6])/2.);
class Interpolation2D
{
	double lowest_x, highest_x;
	double lowest_y, highest_y;

	gsl_interp_accel *x_accelerator;
	gsl_interp_accel *y_accelerator;
	gsl_spline2d *spline;
	
public:

	Interpolation2D(GSL_2D_INTERPOLATION_TYPE interp_type, \
					const double *x, const double *y, const double *z, \
					long x_size, long y_size);
	Interpolation2D(const Interpolation2D &itp2d);

	~Interpolation2D();

	double evaluate(double x, double y) const;
};

#endif
