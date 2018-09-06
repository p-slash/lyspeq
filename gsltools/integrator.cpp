#include "integrator.hpp"

#include <new>

#define WORKSPACE_SIZE 3000
#define TABLE_SIZE 300
#define GL_INFINTY 1E10

#define ABS_ERROR 0
#define REL_ERROR 1E-5

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

double Integrator::evaluateInftyToInfty()
{
    double result = 0, error = 0;
    
    if (gsl_type == GSL_QNG)
    {
        size_t neval;

        gsl_integration_qng(&F, -GL_INFINTY, GL_INFINTY, \
                            ABS_ERROR, REL_ERROR, \
                            &result, &error, \
                            &neval);
    }

    else if (gsl_type == GSL_GL)
    {
        result = gsl_integration_glfixed(&F, -GL_INFINTY, GL_INFINTY, t);
    }

    else // if (gsl_type == GSL_QAG || gsl_type == GSL_QAGS)
    {
        gsl_integration_qagi(   &F, \
                                ABS_ERROR, REL_ERROR, \
                                WORKSPACE_SIZE, w, \
                                &result, &error);
    }

    return result;
}

double Integrator::evaluateAToInfty(double lower_limit)
{
    double result = 0, error = 0;

    if (gsl_type == GSL_QNG)
    {
        size_t neval;

        gsl_integration_qng(&F, lower_limit, GL_INFINTY, \
                            ABS_ERROR, REL_ERROR, \
                            &result, &error, \
                            &neval);
    }

    else if (gsl_type == GSL_GL)
    {
        result = gsl_integration_glfixed(&F, lower_limit, GL_INFINTY, t);
    }

    else
    {
        gsl_integration_qagiu(  &F, lower_limit, \
                                ABS_ERROR, REL_ERROR, \
                                WORKSPACE_SIZE, w, \
                                &result, &error);
    }

    return result;
}

double Integrator::evaluate(double lower_limit, double upper_limit)
{
    double result = 0, error = 0;

    if (gsl_type == GSL_QNG)
    {
        size_t neval;

        gsl_integration_qng(&F, lower_limit, upper_limit, \
                            ABS_ERROR, REL_ERROR, \
                            &result, &error, \
                            &neval);
    }

    else if (gsl_type == GSL_QAG)
    {
        gsl_integration_qag(    &F, lower_limit, upper_limit, \
                                ABS_ERROR, REL_ERROR, \
                                WORKSPACE_SIZE, GSL_INTEG_GAUSS31, w, \
                                &result, &error);
    } 

    else if (gsl_type == GSL_QAGS)
    {
        gsl_integration_qags(   &F, lower_limit, upper_limit, \
                                ABS_ERROR, REL_ERROR, \
                                WORKSPACE_SIZE, w, \
                                &result, &error);
    }

    else if (gsl_type == GSL_GL)
    {
        result = gsl_integration_glfixed(&F, lower_limit, upper_limit, t);
    }

    return result;
}
