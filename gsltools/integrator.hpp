#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include <gsl/gsl_integration.h>

enum GSL_INTEGRATION_TYPE
{
	GSL_QNG,
	GSL_QAG,
	GSL_QAGS,
	GSL_GL
};

// Integrates a given function using a specified GSL integration algorithm.
// NOT thread-safe! Create local copies.

// Example:
//     double f(double x, void *params)
//     {
//         double alpha = *(double *) params;
//         double f = log(alpha*x) / sqrt(x);
//         return f;
//     }

//     Integrator temp_integrator(GSL_QAGS, f, 1.);

//     result = temp_integrator.evaluate(0, 1);
class Integrator
{
	GSL_INTEGRATION_TYPE gsl_type;
	gsl_function F;
	gsl_integration_workspace *w;
	gsl_integration_glfixed_table *t;

	int evaluate(double a, double b, double &res, double rel_err);
	double try_twice(double a, double b);
	
public:
	Integrator( GSL_INTEGRATION_TYPE gsl_type_key, \
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
