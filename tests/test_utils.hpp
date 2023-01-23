#ifndef SRCDIR
#define SRCDIR "."
#endif
#include <cassert>

int asserter(void (*fnc)(), const char* fnc_name) {
    try {
        fnc();
    } catch (std::exception& e) {
        fprintf(stderr, "ERROR in %s: %s\n", fnc_name,  e.what());
        return 1;
    }
    return 0;
}

bool isClose(double a, double b, double relerr=1e-5, double abserr=1e-8);
bool allClose(const double *a, const double *b, int size);
void printValues(double truth, double result);
void printMatrices(const double *truth, const double *result,
    int nrows, int ncols);
