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

#endif
