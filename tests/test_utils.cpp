#include "tests/test_utils.hpp"
#include "mathtools/matrix_helper.hpp"
#include <cmath>
#include <cstdio>
#include <algorithm>

bool isClose(double a, double b, double relerr, double abserr)
{
    double mag = std::max(fabs(a),fabs(b));
    return fabs(a-b) < (abserr + relerr * mag);
}

bool allClose(
        const double *a, const double *b, int size,
        double relerr, double abserr
) {
    bool result = true;
    for (int i = 0; i < size; ++i)
        result &= isClose(a[i], b[i], relerr, abserr);
    return result;
}

void printValues(double truth, double result)
{
    printf("Result: %13.5e\n", result);
    printf("VS\nTruth : %13.5e\n", truth);
    printf("===========================================\n\n");
}

void printMatrices(const double *truth, const double *result,
    int nrows, int ncols)
{
    printf("Result:\n");
    mxhelp::printfMatrix(result, nrows, ncols);
    printf("VS.\nTruth:\n");
    mxhelp::printfMatrix(truth, nrows, ncols);
    printf("===========================================\n\n");
}
