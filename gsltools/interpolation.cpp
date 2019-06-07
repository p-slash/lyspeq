#include "gsltools/interpolation.hpp"

#include <cstdio>

Interpolation::Interpolation(GSL_INTERPOLATION_TYPE interp_type, const double *x, const double *y, long size)
{
	normalization = 1.0;
	lowest_x  = x[0];
	highest_x = x[size - 1];

	accelerator = gsl_interp_accel_alloc();

	if 		(interp_type == GSL_LINEAR_INTERPOLATION)		spline = gsl_spline_alloc(gsl_interp_linear, size);
	else if (interp_type == GSL_CUBIC_INTERPOLATION)		spline = gsl_spline_alloc(gsl_interp_cspline, size);
	else													throw "1D_INTERP_TYPE_ERROR";
	
	gsl_spline_init(spline, x, y, size);
}

Interpolation::Interpolation(const Interpolation &itp)
{
	normalization = itp.normalization;
	lowest_x      = itp.lowest_x;
	highest_x     = itp.highest_x;

	accelerator = gsl_interp_accel_alloc();
	// Copied from source code of GSL 2.5
	spline      = gsl_spline_alloc(itp.spline->interp->type, itp.spline->size);
	gsl_spline_init(spline, itp.spline->x, itp.spline->y, itp.spline->size);
}

Interpolation::~Interpolation()
{
	gsl_spline_free(spline);
	gsl_interp_accel_free(accelerator);

	//printf("Interpolation destructor called.\n");
	//fflush(stdout);
}

void Interpolation::setNormalization(double norm)
{
	normalization = norm;
}

double Interpolation::evaluate(double x) const
{
	double result;

	if (x < lowest_x)
	{
		printf("WARNING: Extrapolating 1D interpolation for smaller x! x = %e < %e = lowest_x\n", x, lowest_x);
		result = gsl_spline_eval(spline, lowest_x, accelerator);
		result += (x - lowest_x) * gsl_spline_eval_deriv(spline, lowest_x, accelerator);

		result *= normalization;

		return result;
	}

	if (x > highest_x)
	{
		printf("WARNING: Extrapolating 1D interpolation for larger x! x = %e > %e = highest_x\n", x, highest_x);
		result = gsl_spline_eval(spline, highest_x, accelerator);
		result += (x - highest_x) * gsl_spline_eval_deriv(spline, highest_x, accelerator);

		result *= normalization;

		return result;
	}

	return normalization * gsl_spline_eval(spline, x, accelerator);
}

double Interpolation::derivative(double x) const
{
	if (x < lowest_x)
	{
		return normalization * gsl_spline_eval_deriv(spline, lowest_x, accelerator);
	}

	if (x > highest_x)
	{
		return normalization * gsl_spline_eval_deriv(spline, highest_x, accelerator);
	}

	return normalization * gsl_spline_eval_deriv(spline, x, accelerator);	
}


