/* Fits ln (y x^n) = Polynomial(ln x)
 * getValue return y not ln y
 */

#ifndef LN_POLY_FIT_H
#define LN_POLY_FIT_H

#include "polynomial_fitter.hpp"

class LnPolynomialFit: public PolynomialFit
{
public:
    LnPolynomialFit(int degree, int x2n, int size): PolynomialFit(degree, x2n, size) {};
    ~LnPolynomialFit() {};
    
    void initialize(const double *x);

    void fit(const double *y, const double *w);
    double getValue(double x);

    void printFit();
};

#endif
