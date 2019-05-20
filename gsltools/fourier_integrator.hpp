#ifndef FOURIER_INTEGRATOR_H
#define FOURIER_INTEGRATOR_H

#include <gsl/gsl_integration.h>

// NOT thread-safe! Create local copies.
class FourierIntegrator
{
    gsl_function F;
    gsl_integration_workspace *w, *cycle_w;
    gsl_integration_qawo_table *t;
    gsl_integration_qawo_enum GSL_SIN_COS;
    
    void handle_gsl_status(int status);
    
public:
    // gsl_integration_qawo_enum sin_cos should be chosen from below:
    //  GSL_INTEG_COSINE
    //  GSL_INTEG_SINE
    
    FourierIntegrator(  gsl_integration_qawo_enum sin_cos, \
                        double (*integrand_function)(double, void*), void *params);
    ~FourierIntegrator();
    
    void setTableParameters(double omega, double L);

    double evaluate(double omega, double a, double b);
    double evaluateAToInfty(double a);
    double evaluate0ToInfty();
};

#endif
