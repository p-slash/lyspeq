#ifndef MATHUTILS_H
#define MATHUTILS_H

inline double trapz(const double *y, int N, double dx=1.0) {
    double result = y[N - 1] / 2;
    for (int i = N - 2; i > 0; --i)
        result += y[i];
    result += y[0] / 2;
    return result * dx;
}

#endif
