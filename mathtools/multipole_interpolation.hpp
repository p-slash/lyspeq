#ifndef MULTIPOLE_INTERPOLATION_H
#define MULTIPOLE_INTERPOLATION_H

#include <algorithm>
#include <cassert>

#include "mathtools/discrete_interpolation.hpp"
#include "mathtools/mathutils.hpp"


class MultipoleInterpolation {
public:
    static constexpr int MAX_NUM_L = 10;

    // static MultipoleInterpolation fromLog2BicubicSpline(
    //         const DiscreteBicubicSpline* spl, int num_ls=3);

    MultipoleInterpolation(int num_ls=MultipoleInterpolation::MAX_NUM_L)
    : Nell(num_ls) { assert(Nell <= MAX_NUM_L); };

    void setInterpEll(int ell, double x1, double dx, int N, double *y) {
        assert(ell < Nell);
        interp_ls[ell] = std::make_unique<DiscreteCubicInterpolation1D>(
            x1, dx, N, y);
    }

    double evaluate(double k, double mu, bool clamp=true) const {
        double result = 0;
        for (int ell = 0; ell < Nell; ++ell)
            result += evaluateEll(ell, k, clamp) * legendre(2 * ell, mu);
        return result;
    }

    double evaluateEll(int ell, double k, bool clamp=true) const {
        if (clamp)  k = interp_ls[ell]->clamp(k);
        return interp_ls[ell]->evaluate(k);
    }

    DiscreteLogLogInterpolation2D<DiscreteCubicInterpolation1D, DiscreteBicubicSpline>
        toDiscreteLogLogInterpolation2D(double x1, double dx, int N);

private:
    int Nell;
    std::unique_ptr<DiscreteCubicInterpolation1D> interp_ls[MAX_NUM_L];
};
#endif
