/* This object should make it easier to create,
 * keep track of and evaluate interpolations 
 * with GSL libraries.
 * Interpolate with given x and y values
 */

#ifndef INTERPOLATION_H
#define INTERPOLATION_H

#include <gsl/gsl_spline.h>

class Interpolation
{
	double normalization;
	
	gsl_interp_accel *accelerator;
	gsl_spline *spline;

	void construct(const double *x, const double *y, long long int size);
	
public:
	double lowest_x, highest_x;
	
	Interpolation(const double *x, const double *y, long long int size);
	~Interpolation();

	void setNormalization(double norm);

	double evaluate(double x) const;
	double derivative(double x) const;
	// void writeInterpolation(const char *fname);
};

#endif
