#include "fourier_integrator.hpp"

#include <gsl/gsl_errno.h>
#include <new>
#include <cassert>

#define WORKSPACE_SIZE 3000
#define TABLE_SIZE 300

#define ABS_ERROR 1E-15
#define REL_ERROR 1E-7

FourierIntegrator::FourierIntegrator(gsl_integration_qawo_enum sin_cos, \
                                    double (*integrand_function)(double, void*), void *params)
{
    GSL_SIN_COS = sin_cos;

    w       = gsl_integration_workspace_alloc(WORKSPACE_SIZE);
    cycle_w = gsl_integration_workspace_alloc(WORKSPACE_SIZE);

    if (w == NULL || cycle_w == NULL)
    {
        throw std::bad_alloc();
    }

    t = NULL;

    F.function = integrand_function;
    F.params = params;
}

FourierIntegrator::~FourierIntegrator()
{
    gsl_integration_workspace_free(w);
    gsl_integration_workspace_free(cycle_w);

    if (t != NULL)
    {
        gsl_integration_qawo_table_free(t);
    }
}

void FourierIntegrator::setTableParameters(double omega, double L)
{
    /* The length L can take any value for infty integrals, since it is overridden 
     * by this function to a value appropriate for the Fourier integration.
     */

    if (t == NULL)
    {
        t = gsl_integration_qawo_table_alloc(omega, L, GSL_SIN_COS, TABLE_SIZE);
        
        if (t == NULL)
        {
            throw std::bad_alloc();
        }
    }
    else 
    {
        gsl_integration_qawo_table_set(t, omega, L, GSL_SIN_COS);
    }
}

double FourierIntegrator::evaluateAToInfty(double a)
{
    assert(t != NULL);

    double result, error;
    int status = gsl_integration_qawf(  &F, a, ABS_ERROR, \
                                        WORKSPACE_SIZE, w, cycle_w, t, \
                                        &result, &error);
    handle_gsl_status(status);

    if (error/result > REL_ERROR)
    {
        #undef ABS_ERROR
        #define ABS_ERROR result*REL_ERROR
        result = evaluateAToInfty(a);
        #undef ABS_ERROR
        #define ABS_ERROR 1E-12
    }

    return result;
}

double FourierIntegrator::evaluate0ToInfty()
{
    return evaluateAToInfty(0);
}

double FourierIntegrator::evaluate(double omega, double a, double b)
{
    int status;
    double result, error;

    setTableParameters(omega, b-a);

    status = gsl_integration_qawo(  &F, a, ABS_ERROR, REL_ERROR, \
                                    WORKSPACE_SIZE, w, t, \
                                    &result, &error);

    handle_gsl_status(status);

    return result;
}

void FourierIntegrator::handle_gsl_status(int status)
{
    if (status)
    {
        const char *err_msg = gsl_strerror(status);
        fprintf(stderr, "ERROR in FourierIntegrator: %s\n", err_msg);
        
        if (status == GSL_ETABLE)
            fprintf(stderr, "Number of levels %d is insufficient for the requested accuracy.\n", TABLE_SIZE);

        if (status == GSL_EDIVERGE)
            fprintf(stderr, "The integral is divergent, or too slowly convergent to be integrated numerically.\n");
        throw err_msg;
    }
}











