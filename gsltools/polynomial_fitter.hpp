#ifndef POLYNOMIAL_FITTER_H
#define POLYNOMIAL_FITTER_H

#include <gsl/gsl_matrix.h> 
#include <gsl/gsl_multifit.h>

class PolynomialFit
{
protected:
    int polynomial_degree, y_times_x2n, \
        number_of_data;

    double chi_square;
    double *x_array;

    gsl_matrix *x_matrix, *fit_parameters_covariance_matrix;
    gsl_vector *y_vector, *weight_vector, *fit_parameters_vector;

    gsl_multifit_linear_workspace *mf_lin_ws;

    void applyMask();
    void setYW(const double *y, const double *w);
    double getChiSquarePDOF();

public:
    bool *mask_array;
    double *fitted_values;
    gsl_vector_view fit_values_view;

    PolynomialFit(int degree, int x2n, int size);
    ~PolynomialFit();
    
    void initialize(const double *x);

    void fit(const double *y, const double *w);
    double getValue(double x);

    void printFit();
};

#endif
