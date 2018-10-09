#include "integrator.hpp"

#include <gsl/gsl_errno.h>
#include <new>

#define WORKSPACE_SIZE 3000
#define TABLE_SIZE 300
#define GL_INFINTY 1E10

#define ABS_ERROR 0
#define REL_ERROR_LARGE 1E-3
#define REL_ERROR_SMALL 1E-5

Integrator::Integrator( GSL_INTEGRATION_TYPE gsl_type_key, \
                        double (*integrand_function)(double, void *), void *params)
{
    gsl_type = gsl_type_key;

    if (gsl_type == GSL_GL)
    {
        t = gsl_integration_glfixed_table_alloc(TABLE_SIZE);

        if (t == NULL)
        {
            throw std::bad_alloc();
        }
    }

    else if (gsl_type == GSL_QAG || gsl_type == GSL_QAGS)
    {
        w = gsl_integration_workspace_alloc(WORKSPACE_SIZE);

        if (w == NULL)
        {
            throw std::bad_alloc();
        }
    }

    F.function = integrand_function;
    F.params = params;
}

Integrator::~Integrator()
{
    if (gsl_type == GSL_GL)
    {
        gsl_integration_glfixed_table_free(t);
    }

    else if (gsl_type == GSL_QAG || gsl_type == GSL_QAGS)
    {
        gsl_integration_workspace_free(w);
    }
}

int Integrator::evaluate(double a, double b, double &res, double rel_err)
{
    double error = 0;
    int status;

    if (gsl_type == GSL_QNG)
    {
        size_t neval;

        status = gsl_integration_qng(&F, a, b, \
                                    ABS_ERROR, rel_err, \
                                    &res, &error, \
                                    &neval);
    }

    else if (gsl_type == GSL_GL)
    {
        res = gsl_integration_glfixed(&F, a, b, t);
        status = 1;
    }

    else
    {
        if (b == GL_INFINTY)
        {
            if (a == -GL_INFINTY)
            {
                status = gsl_integration_qagi(  &F, \
                                                ABS_ERROR, rel_err, \
                                                WORKSPACE_SIZE, w, \
                                                &res, &error);
            }
            else
            {
                status = gsl_integration_qagiu( &F, a, \
                                                ABS_ERROR, rel_err, \
                                                WORKSPACE_SIZE, w, \
                                                &res, &error);
            }
        }
        
        else if (gsl_type == GSL_QAG)
        {
            
            status = gsl_integration_qag(&F, a, b, \
                                        ABS_ERROR, rel_err, \
                                        WORKSPACE_SIZE, GSL_INTEG_GAUSS51, w, \
                                        &res, &error);
        }
        
        else if (gsl_type == GSL_QAGS)
        {
            status = gsl_integration_qags(&F, a, b, \
                                        ABS_ERROR, rel_err, \
                                        WORKSPACE_SIZE, w, \
                                        &res, &error);
        }
    }

    return status;
}

double Integrator::evaluateInftyToInfty()
{
    double result = 0;

    int status = evaluate(-GL_INFINTY, GL_INFINTY, result, REL_ERROR_SMALL);

    if (status == GSL_EROUND)
    {
        evaluate(-GL_INFINTY, GL_INFINTY, result, REL_ERROR_LARGE);
    }

    return result;
}

double Integrator::evaluateAToInfty(double lower_limit)
{
    double result = 0;
    
    int status = evaluate(lower_limit, GL_INFINTY, result, REL_ERROR_SMALL);

    if (status == GSL_EROUND)
    {
        evaluate(lower_limit, GL_INFINTY, result, REL_ERROR_LARGE);
    }

    return result;
}

double Integrator::evaluate(double lower_limit, double upper_limit)
{
    double result = 0;
    
    int status = evaluate(lower_limit, upper_limit, result, REL_ERROR_SMALL);

    if (status == GSL_EROUND)
    {
        evaluate(lower_limit, upper_limit, result, REL_ERROR_LARGE);
    }

    return result;
}
