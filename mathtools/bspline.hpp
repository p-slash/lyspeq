#ifndef BSPLINE_H
#define BSPLINE_H
// prevent gsl_cblas.h from being included
#define  __GSL_CBLAS_H__

#ifdef USE_MKL_CBLAS
#include "mkl_cblas.h"
#else
#include "cblas.h"
#endif

#include <gsl/gsl_bspline.h>
#include <gsl/gsl_vector.h>

class BSpline {
    gsl_bspline_workspace *work;
    gsl_vector *c;
public:
    BSpline(
        const double *x, const double* y, int N, const double *wts=nullptr,
        int k=4, int nbreak=0
    );
    ~BSpline() { gsl_bspline_free(work);  gsl_vector_free(c); }

    double evaluate(double x) {
        double r;
        gsl_bspline_calc(x, c, &r, work);
        return r;
    }

    static void smoothInputY(
        const double *x, double* y, int N,
        const double *wts=nullptr, int k=4, int nbreak=0
    );
};

#endif
