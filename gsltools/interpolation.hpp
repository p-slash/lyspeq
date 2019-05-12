#ifndef INTERPOLATION_H
#define INTERPOLATION_H

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
class Interpolation
{
	double normalization;
	
	gsl_interp_accel *accelerator;
	gsl_spline *spline;
	
public:
	double lowest_x, highest_x;
	
	Interpolation(GSL_INTERPOLATION_TYPE interp_type, const double *x, const double *y, long size);
	Interpolation(const Interpolation &itp);

	~Interpolation();

	void setNormalization(double norm);

	double evaluate(double x) const;
	double derivative(double x) const;
	// void writeInterpolation(const char *fname);
};

#endif
