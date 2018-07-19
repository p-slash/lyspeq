#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include <gsl/gsl_integration.h>

enum GSL_TYPE
{
	GSL_QNG,
	GSL_QAG,
	GSL_QAGS,
	GSL_GL
};

class Integrator
{
	GSL_TYPE gsl_type;
	gsl_function F;
	gsl_integration_workspace *w;
	gsl_integration_glfixed_table *t;

public:
	Integrator( GSL_TYPE gsl_type_key, \
				double (*integrand_function)(double, void*), void *params);
	~Integrator();
	
	//double setFunctionNParams(double (*integrand_function)(double, void*), void *params);
	// Intfy functions use a different algorithym 
	// and may not give desired values of GSL_GL.
	double evaluateInftyToInfty();
	double evaluateAToInfty(double lower_limit);
	
	double evaluate(double lower_limit, double upper_limit);
};

#endif
