#include "interpolation_2d.hpp"

#include <gsl/gsl_interp2d.h>

#include <cstdio>

void Interpolation2D::construct(const double *x, const double *y, const double *z, \
								long x_size, long y_size)
{
	lowest_x  = x[0];
	highest_x = x[x_size - 1];

	lowest_y = y[0];
	highest_y= y[y_size - 1];

	x_accelerator = gsl_interp_accel_alloc();
	y_accelerator = gsl_interp_accel_alloc();

	spline = gsl_spline2d_alloc(gsl_interp2d_bilinear, x_size, y_size);

	gsl_spline2d_init(spline, x, y, z, x_size, y_size);

	//printf("Constructing Interpolation2D object: Done!\n");
	//fflush(stdout);
}

Interpolation2D::Interpolation2D(	const double *x, const double *y, const double *z, \
									long x_size, long y_size)
{
	construct(x, y, z, x_size, y_size);
}

Interpolation2D::~Interpolation2D()
{
	gsl_spline2d_free(spline);
	gsl_interp_accel_free(x_accelerator);
	gsl_interp_accel_free(y_accelerator);

	//printf("Interpolation2D destructor called.\n");
	//fflush(stdout);
}

double Interpolation2D::evaluate(double x, double y) const
{
	double x_eval = x, y_eval = y;

	if (x < lowest_x)
	{
		printf("WARNING: Extrapolating 2D interpolation for smaller x!\n");
		x_eval = lowest_x;
	}

	else if (x > highest_x)
	{
		printf("WARNING: Extrapolating 2D interpolation for larger x!\n");
		x_eval = highest_x;
	}

	if (y < lowest_y)
	{
		printf("WARNING: Extrapolating 2D interpolation for smaller y!\n");
		y_eval = lowest_y;
	}

	else if (y > highest_y)
	{
		printf("WARNING: Extrapolating 2D interpolation for larger y!\n");
		y_eval = highest_y;
	}

	return gsl_spline2d_eval(spline, x_eval, y_eval, x_accelerator, y_accelerator);
}
