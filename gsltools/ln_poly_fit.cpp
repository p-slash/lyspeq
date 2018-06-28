#include "ln_poly_fit.hpp"

#include <algorithm>
#include <cmath>

void LnPolynomialFit::initialize(const double *x)
{
    // Set with ln of x values
    for (int i = 0; i < number_of_data; i++)
    {
        x_array[i] = log(x[i]);
    }

    PolynomialFit::initialize(x_array);

    for (int i = 0; i < number_of_data; i++)
    {
        x_array[i] = exp(x_array[i]);
    }
}

void LnPolynomialFit::fit(const double *y, const double *w)
{
    PolynomialFit::setYW(y, w);

    // Convert to ln of y and weight values
    for (int i = 0; i < number_of_data; i++)
    {
        double  y_i = gsl_vector_get(y_vector, i), \
                w_i = gsl_vector_get(weight_vector, i);

        if (y_i <= 0)
        {
            gsl_vector_set(y_vector, i, 0);
            gsl_vector_set(weight_vector, i, 0);
        }
        else
        {
            gsl_vector_set(y_vector, i, log(y_i));
            gsl_vector_set(weight_vector, i, y_i * y_i * w_i);
        }
    }
    
    // fit for ln y = Poly(ln x)
    gsl_multifit_wlinear(x_matrix, weight_vector, y_vector, \
                         fit_parameters_vector, fit_parameters_covariance_matrix, \
                         &chi_square, mf_lin_ws);

    for (int i = 0; i < number_of_data; i++)
    {
        fitted_values[i] = getValue(x_array[i]);
    }
}

double LnPolynomialFit::getValue(double x)
{
    int temp = y_times_x2n;
    y_times_x2n = 0;

    double result = PolynomialFit::getValue(log(x));

    y_times_x2n = temp;
    result = exp(result) / pow(x, y_times_x2n);

    return result;
}

void LnPolynomialFit::printFit()
{
    if (y_times_x2n != 0)
        printf("ln(y * x^%d) = ", y_times_x2n);
    else
        printf("ln(y) = \n");

    printf("%.2le ", gsl_vector_get(fit_parameters_vector, 0));

    for (int p = 1; p <= polynomial_degree; p++)
    {
        printf("+ %.2le (ln x)^%d ", gsl_vector_get(fit_parameters_vector, p), p);
    }
    printf("; chi^2 / dof = %.3lf \n", getChiSquarePDOF());
    
    printf("Power spectrum fit : ");
    for (int i = 0; i < number_of_data; i++)
    {
        printf("%.2le ", fitted_values[i]);
    }
    printf("\n");
    fflush(stdout);
}





