#include "interpolation.hpp"
// #include "../io/io_helper_functions.hpp"


// #include <gsl/gsl_integration.h>
// #include <gsl/gsl_sf_bessel.h>

#include <cstdio>

#define PI 3.14159265359


void Interpolation::construct(const double *x, const double *y, long long int size)
{
	normalization = 1.0;
	lowest_x  = x[0];
	highest_x = x[size - 1];

	accelerator = gsl_interp_accel_alloc();

	spline = gsl_spline_alloc(gsl_interp_linear, size);
	gsl_spline_init(spline, x, y, size);
	
	//printf("Constructing Interpolation object: Done!\n");
	//fflush(stdout);
}

Interpolation::Interpolation(const double *x, const double *y, long long int size)
{
	construct(x, y, size);
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

// void Interpolation::writeInterpolation(const char *fname)
// {
// 	int length = 2000;
// 	const double step_size	= (highest_x - lowest_x) / length;
// 	length = (highest_x - lowest_x) / step_size;
	
// 	FILE *toWrite;

// 	toWrite = open_file(fname, "w");

// 	fprintf(toWrite, "%d\n", length);
	
// 	for (double i = lowest_x; i < highest_x; i += step_size)
// 		fprintf(toWrite, "%e %e\n", i, evaluate(i));

// 	fclose(toWrite);
	
// 	printf("Interpolation saved as %s\n", fname);
// }
