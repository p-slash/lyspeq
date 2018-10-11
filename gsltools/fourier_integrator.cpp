#include "fourier_integrator.hpp"

#include <gsl/gsl_errno.h>
#include <new>
#include <cassert>

#define WORKSPACE_SIZE 3000
#define TABLE_SIZE 300

#define ABS_ERROR 1E-10

FouerierIntegrator::FouerierIntegrator(double (*integrand_function)(double, void*), void *params)
{
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

FouerierIntegrator::~FouerierIntegrator()
{
    gsl_integration_workspace_free(w);
    gsl_integration_workspace_free(cycle_w);

    if (t != NULL)
    {
        gsl_integration_qawo_table_free(t);
    }
}

void FouerierIntegrator::setTableParameters(double omega, gsl_integration_qawo_enum sin_cos)
{
    /* The length L can take any value, since it is overridden 
     * by this function to a value appropriate for the Fourier integration.
     */

    double L = 10.;

    if (t == NULL)
    {
        t = gsl_integration_qawo_table_alloc(omega, L, sin_cos, TABLE_SIZE);
        
        if (t == NULL)
        {
            throw std::bad_alloc();
        }
    }
    else 
    {
        gsl_integration_qawo_table_set(t, omega, L, sin_cos);
    }
}

double FouerierIntegrator::evaluateAToInfty(double lower_limit)
{
    assert(t != NULL);

    double result, error;
    int status = gsl_integration_qawf(  &F, lower_limit, ABS_ERROR, \
                                        WORKSPACE_SIZE, w, cycle_w, t, \
                                        &result, &error);
    if (status == GSL_ETABLE)
    {
        fprintf(stderr, "Number of levels %d is insufficient for the requested accuracy.\n", TABLE_SIZE);
        throw "FTint_LowTable";
    }
    else if (status == GSL_EDIVERGE)
    {
        fprintf(stderr, "The integral is divergent, or too slowly convergent to be integrated numerically\n");
        throw "FTint_Diverge";
    }

    return result;
}

double FouerierIntegrator::evaluate0ToInfty()
{
    return evaluateAToInfty(0);
}













