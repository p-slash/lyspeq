#ifndef SRCDIR
#define SRCDIR "."
#endif
#include <cassert>
#include <stdexcept>

constexpr char* err_liner(const char* sfile, int line) {
    char buf[250] = "";
    sprintf(buf, "not true in %s at %d.", sfile, line);
    return buf;
}

int asserter(void (*fnc)(), const char* fnc_name) {
    try {
        fnc();
    } catch (std::exception& e) {
        fprintf(stderr, "ERROR in %s: %s\n", fnc_name,  e.what());
        return 1;
    }
    return 0;
}


void raiser(bool pass_test, const char* msg) {
    if (! pass_test)
        throw std::runtime_error(msg);
}

bool isClose(double a, double b, double relerr=1e-5, double abserr=1e-8);
bool allClose(const double *a, const double *b, int size);
void printValues(double truth, double result);
void printMatrices(const double *truth, const double *result,
    int nrows, int ncols);
