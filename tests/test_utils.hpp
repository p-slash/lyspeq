#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#ifndef SRCDIR
#define SRCDIR "."
#endif
#include "mathtools/matrix_helper.hpp"

#include <cassert>
#include <stdexcept>
#include <cmath>
#include <cstdio>
#include <algorithm>


bool isClose(double a, double b, double relerr=1e-5, double abserr=1e-8) {
    double mag = std::max(fabs(a),fabs(b));
    return fabs(a-b) < (abserr + relerr * mag);
}


bool allClose(const double *a, const double *b, int size) {
    bool result = true;
    for (int i = 0; i < size; ++i)
        result &= isClose(a[i], b[i]);
    return result;
}


void printValues(double truth, double result) {
    printf("Result: %13.5e\n", result);
    printf("VS\nTruth : %13.5e\n", truth);
    printf("===========================================\n\n");
}


void printMatrices(
        const double *truth, const double *result,
        int nrows, int ncols) {
    printf("Result:\n");
    mxhelp::printfMatrix(result, nrows, ncols);
    printf("VS.\nTruth:\n");
    mxhelp::printfMatrix(truth, nrows, ncols);
    printf("===========================================\n\n");
    fflush(stdout);
}


int catcher(void (*fnc)(), const char* fnc_name) {
    printf("%s...", fnc_name);
    try {
        fnc();
    } catch (std::exception& e) {
        printf(" ERROR: %s\n",  e.what());
        return 1;
    }
    printf(" passed.\n");
    return 0;
}


void raiser(bool pass_test, const char* sfile, int line) {
    char msg[250] = "";
    sprintf(msg, "not true in %s at %d.", sfile, line);
    if (!pass_test)
        throw std::runtime_error(msg);
}


void assert_allclose_2d(
        const double *expected, const double *current, int nrows, int ncols,
        const char* sfile, int line) {
    bool pass_test = allClose(expected, current, nrows * ncols);

    if (!pass_test)
        printMatrices(expected, current, nrows, ncols);
    raiser(pass_test, sfile, line);
}


void assert_allclose(
        const double *expected, const double *current, int size,
        const char* sfile, int line) {
    assert_allclose_2d(expected, current, size, 1, sfile, line);
}

#endif
