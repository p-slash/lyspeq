/* This object should make it easier to create,
 * keep track of and evaluate interpolations in 2D
 * with GSL libraries.
 * Interpolate with given x, y and z[y * xsize + x] values
 * or interpolate a anisotropic spectrum directly
 * as[perp + size_perpendicular * para] then perp is x
 */

#include <gsl/gsl_spline2d.h>

#ifndef INTERPOLATION_2D_H
#define INTERPOLATION_2D_H

class Interpolation2D
{
	double lowest_x, highest_x;
	double lowest_y, highest_y;

	gsl_interp_accel *x_accelerator;
	gsl_interp_accel *y_accelerator;
	gsl_spline2d *spline;

	void construct(	const double *x, const double *y, const double *z, \
					long x_size, long y_size);
	
public:

	Interpolation2D(const double *x, const double *y, const double *z, \
					long x_size, long y_size);

	~Interpolation2D();

	double evaluate(double x, double y) const;
	//double derivative(double x, double y);
	//long writeInterpolation(const char *fname);
};

#endif
