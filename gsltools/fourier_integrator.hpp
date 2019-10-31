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
    
    FourierIntegrator(gsl_integration_qawo_enum sin_cos, double (*integrand_function)(double, void*), void *params);
    ~FourierIntegrator();
    
    // The length L can take any value for infty integrals, since it is overridden 
    // by this function to a value appropriate for the Fourier integration.
    void setTableParameters(double omega, double L);
    // Table t should be before allocated.
    void changeTableLength(double L);

    // If omega < 0, keeps previous omega and only changes the table length.
    double evaluate(double a, double b, double omega=-1, double epsabs=1E-13, double epsrel=1E-7);
    double evaluateAToInfty(double a, double epsabs=1E-13);
    double evaluate0ToInfty(double epsabs=1E-13);
};

#endif
