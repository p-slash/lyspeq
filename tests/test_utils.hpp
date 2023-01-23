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
    try {
        fnc();
    } catch (std::exception& e) {
        fprintf(stderr, "ERROR in %s: %s\n", fnc_name,  e.what());
        return 1;
    }
    return 0;
}


void raiser(bool pass_test, const char* sfile, int line) {
    char msg[250] = "";
    sprintf(msg, "not true in %s at %d.", sfile, line);
    if (!pass_test)
        throw std::runtime_error(msg);
}


void assert_allclose(
        const double *expected, const double *current, int size,
        const char* sfile, int line) {
    bool pass_test = allClose(expected, current, size);

    if (!pass_test)
        printMatrices(expected, current, size, 1);
    raiser(pass_test, sfile, line);
}
