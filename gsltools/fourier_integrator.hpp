#ifndef FOURIER_INTEGRATOR_H
#define FOURIER_INTEGRATOR_H

#include <gsl/gsl_integration.h>

class FouerierIntegrator
{
    gsl_function F;
    gsl_integration_workspace *w, *cycle_w;
    gsl_integration_qawo_table *t;

    int evaluate(double a, double b, double &res, double rel_err);
    double try_twice(double a, double b);
    
public:
    FouerierIntegrator(double (*integrand_function)(double, void*), void *params);
    ~FouerierIntegrator();
    
    /*  gsl_integration_qawo_enum sin_cos should be chosen from below:
        GSL_INTEG_COSINE
        GSL_INTEG_SINE
    */
    void setTableParameters(double omega, gsl_integration_qawo_enum sin_cos);

    double evaluateAToInfty(double lower_limit);
    double evaluate0ToInfty();
};

#endif
