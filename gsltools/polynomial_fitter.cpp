#include "polynomial_fitter.hpp"

#include <cmath>

PolynomialFit::PolynomialFit(int degree, int x2n, int size)
{
    polynomial_degree = degree;
    y_times_x2n       = x2n;
    number_of_data    = size;

    x_matrix      = gsl_matrix_alloc(number_of_data, polynomial_degree + 1);
    y_vector      = gsl_vector_alloc(number_of_data);
    weight_vector = gsl_vector_alloc(number_of_data);

    fit_parameters_vector            = gsl_vector_alloc(polynomial_degree + 1);
    fit_parameters_covariance_matrix = gsl_matrix_alloc(polynomial_degree + 1, polynomial_degree + 1);

    fitted_values   = new double[number_of_data];
    x_array         = new double[number_of_data];
    mask_array      = new bool[number_of_data];

    fit_values_view = gsl_vector_view_array(fitted_values, number_of_data);

    mf_lin_ws = gsl_multifit_linear_alloc(number_of_data, polynomial_degree + 1);
}

PolynomialFit::~PolynomialFit()
{
    gsl_matrix_free(x_matrix);
    gsl_matrix_free(fit_parameters_covariance_matrix);

    gsl_vector_free(y_vector);
    gsl_vector_free(weight_vector);
    gsl_vector_free(fit_parameters_vector);

    gsl_multifit_linear_free(mf_lin_ws);
    
    delete [] fitted_values;
    delete [] x_array;
    delete [] mask_array;
}

void PolynomialFit::initialize(const double *x)
{
    for (int i = 0; i < number_of_data; i++)
    {
        x_array[i]    = x[i];
        mask_array[i] = false;

        gsl_matrix_set(x_matrix, i, 0, 1.);

        for (int p = 1; p <= polynomial_degree; p++)
        {
            gsl_matrix_set(x_matrix, i, p, pow(x_array[i], p));
        }
    }
}

void PolynomialFit::applyMask()
{
    for (int i = 0; i < number_of_data; i++)
    {
        if (mask_array[i])
        {
            gsl_vector_set(weight_vector, i, 0);
        }
    }
}

void PolynomialFit::setYW(const double *y, const double *w)
{
    for (int i = 0; i < number_of_data; i++)
    {
        gsl_vector_set(y_vector, i, y[i]);
        gsl_vector_set(weight_vector, i, w[i]);
    }

    if (y_times_x2n != 0)
    {
        double *x2n_array = new double[number_of_data];

        for (int i = 0; i < number_of_data; i++)
        {
            x2n_array[i] = pow(x_array[i], y_times_x2n);
        }

        gsl_vector_view x2n_view = gsl_vector_view_array(x2n_array, number_of_data);

        gsl_vector_mul(y_vector, &x2n_view.vector);
        gsl_vector_div(weight_vector, &x2n_view.vector);
        gsl_vector_div(weight_vector, &x2n_view.vector);

        delete [] x2n_array;
    }

    applyMask();
}

void PolynomialFit::fit(const double *y, const double *w)
{
    setYW(y, w);
    
    gsl_multifit_wlinear(x_matrix, weight_vector, y_vector, \
                         fit_parameters_vector, fit_parameters_covariance_matrix, \
                         &chi_square, mf_lin_ws);

    for (int i = 0; i < number_of_data; i++)
    {
        fitted_values[i] = getValue(x_array[i]);
    }
}

double PolynomialFit::getValue(double x)
{
    double result = 0;

    for (int p = 0; p <= polynomial_degree; p++)
    {
        result += gsl_vector_get(fit_parameters_vector, p) * pow(x, p - y_times_x2n);
    }

    return result;
}

void PolynomialFit::printFit()
{
    if (y_times_x2n != 0)
        printf("y * x^%d = ", y_times_x2n);
    else
        printf("y = \n");

    printf("%.3lf ", gsl_vector_get(fit_parameters_vector, 0));

    for (int p = 1; p <= polynomial_degree; p++)
    {
        printf("+ %.2le x^%d ", gsl_vector_get(fit_parameters_vector, p), p);
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

double PolynomialFit::getChiSquarePDOF()
{
    int non_zero_w = 0;

    for (int i = 0; i < number_of_data; i++)
    {
        if (!mask_array[i])
        {
            non_zero_w++;
        }
    }

    return chi_square / (non_zero_w - polynomial_degree - 1);
}








