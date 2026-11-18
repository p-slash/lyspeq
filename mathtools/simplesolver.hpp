#ifndef SIMPLEFITTING_H
#define SIMPLEFITTING_H

#include "functional"

double simpleCurveFit1D(
        std::function<double(double)> func, double y, double x1, double x2,
        double rtol=1.49012e-08
) {
    constexpr int N_STEPS = 100, N_LOOPS = 10;
    if (y == 0)  y = rtol;  // Avoid division by zero
    double Deltax = (x2 - x1) / N_STEPS,
           min_ydiff = 1e99,
           soln = (x1 + x2) / 2.0;

    for (int loop = 0; loop < N_LOOPS; ++loop) {
        for (int i = 0; i < N_STEPS; ++i) {
            double xi = x1 + i * Deltax;
            double delta_t = fabs(func(xi) / y - 1.0);
            if (delta_t < min_ydiff) {
                min_ydiff = delta_t;
                soln = xi;
            }
            else  break;
        }

        if (min_ydiff < rtol)  break;

        x1 = soln - 2.0 * Deltax;
        Deltax *= 4.0 / N_STEPS;
        min_ydiff = 1e10;
    }
    return soln;
}


double rootBisection(
        std::function<double(double)> func, double x1, double x2,
        int *status=nullptr, int max_iter=1000, double rtol=1.49012e-08
) {
    double xsol_old = x1;
    double f1 = func(x1), f2 = func(x2);
    if (f1 * f2 > 0) {
        if (status) *status = -1; // No root in the interval
        return xsol_old;
    }

    double xsol, fsol;
    bool converged = false;
    for (int i = 0; i < max_iter; i++) {
        xsol = (x1 + x2) / 2.0;
        fsol = func(xsol);
        converged = fsol == 0.0;
        if (xsol != 0.0)
            converged |= fabs(xsol_old - xsol) < rtol * fabs(xsol);
        else
            converged |= fabs(xsol - xsol_old) < rtol;
        if (converged) {
            if (status) *status = 0;
            return xsol;
        }

        if (f1 * fsol < 0) {
            x2 = xsol;
            f2 = fsol;
        } else {
            x1 = xsol;
            f1 = fsol;
        }
    }
    if (status) *status = 1; // Max iterations reached
    return xsol;
}

double rootBrent(
        std::function<double(double)> func, double x1, double x2,
        int *status=nullptr, int max_iter=10000, double rtol=1e-6
) {
    double f1 = func(x1), f2 = func(x2);
    if (fabs(f1) < fabs(f2)) {
        std::swap(x1, x2);
        std::swap(f1, f2);
    }

    if (f1 * f2 > 0) {
        if (status) *status = -1; // No root in the interval
        return x1;
    }

    // x2 is always the best estimate (|f2| <= |f1|), x1 the contrapoint.
    double c = x1, fc = f1, d = x1, xsol, fsol;
    bool mflag = true;
    for (int i = 0; i < max_iter; i++) {
        double delta = 2 * rtol * fabs(x2) + rtol / 2.0;
        if (f2 == 0.0 || fabs(x2 - x1) < delta) {
            if (status) *status = 0;
            return x2;
        }

        if (f1 != fc && f2 != fc) {  // Inverse quadratic interpolation
            xsol = (x1 * f2 * fc) / ((f1 - f2) * (f1 - fc))
                 + (x2 * f1 * fc) / ((f2 - f1) * (f2 - fc))
                 + (c * f1 * f2) / ((fc - f1) * (fc - f2));
        } else {  // Secant method
            xsol = x2 - f2 * (x2 - x1) / (f2 - f1);
        }

        // xsol must lie strictly between (3 * x1 + x2) / 4 and x2,
        // regardless of whether x1 < x2 or x1 > x2.
        bool do_bisection = ((xsol - (3 * x1 + x2) / 4) * (xsol - x2) >= 0);
        do_bisection |= (mflag && fabs(xsol - x2) >= fabs(x2 - c) / 2);
        do_bisection |= (!mflag && fabs(xsol - x2) >= fabs(c - d) / 2);
        do_bisection |= (mflag && fabs(x2 - c) < delta);
        do_bisection |= (!mflag && fabs(c - d) < delta);
        if (do_bisection) {
            xsol = (x1 + x2) / 2;
            mflag = true;
        } else {
            mflag = false;
        }

        fsol = func(xsol);

        d = c;
        c = x2;
        fc = f2;

        if (f1 * fsol < 0) {
            x2 = xsol;
            f2 = fsol;
        } else {
            x1 = xsol;
            f1 = fsol;
        }
        if (fabs(f1) < fabs(f2)) {
            std::swap(x1, x2);
            std::swap(f1, f2);
        }
    }
    if (status) *status = 1; // Max iterations reached
    return x2;
}

#endif
