#include "bspline.hpp"
#include <gsl/gsl_errno.h>


void _handleGslStatus(int status) {
    const char *err_msg = gsl_strerror(status);
    fprintf(stderr, "ERROR in BSpline: %s\n", err_msg);
}


BSpline::BSpline(
        const double *x, const double* y, int N, 
        const double *wts, int k, int nbreak
) : work(nullptr), c(nullptr)
{
    if (nbreak <= 0)
        nbreak = std::max(2 * k + 2, N / 10);

    gsl_vector_const_view xvec = gsl_vector_const_view_array(x, N);
    gsl_vector_const_view yvec = gsl_vector_const_view_array(y, N);

    int s = 0;
    work = gsl_bspline_alloc(k, nbreak);
    // status = gsl_bspline_init_interp(&xvec.vector, work);
    s = gsl_bspline_init_uniform(x[0], x[N - 1], work);

    if (s)  _handleGslStatus(s);

    const size_t p1 = gsl_bspline_ncontrol(work);
    c = gsl_vector_alloc(p1);

    double chisq;
    if (wts == nullptr) {
        s = gsl_bspline_lssolve(&xvec.vector, &yvec.vector, c, &chisq, work);
    }
    else {
        gsl_vector_const_view wvec = gsl_vector_const_view_array(wts, N);
        s = gsl_bspline_wlssolve(
            &xvec.vector, &yvec.vector, &wvec.vector, c,  &chisq, work);
    }

    if (s)  _handleGslStatus(s);
}


void BSpline::smoothInputY(
        const double *x, double* y, int N, const double *wts, int k, int nbreak
) {
    BSpline spl(x, y, N, wts, k, nbreak);

    for (int i = 0; i < N; ++i)
        gsl_bspline_calc(x[i], spl.c, y + i, spl.work);
}
