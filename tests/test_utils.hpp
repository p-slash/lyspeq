#ifndef SRCDIR
#define SRCDIR "."
#endif
#include <cassert>
#include <stdexcept>


bool isClose(double a, double b, double relerr=1e-5, double abserr=1e-8);
bool allClose(const double *a, const double *b, int size);
void printValues(double truth, double result);
void printMatrices(const double *truth, const double *result,
    int nrows, int ncols);


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
