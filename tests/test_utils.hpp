#ifndef SRCDIR
#define SRCDIR "."
#endif

int asserter(void (*fnc)(), const char* err_msg) {
    try {
        fnc();
    } catch {
        fprintf(stderr, "%s\n", err_msg);
        return 1;
    }
    return 0;
}

bool isClose(double a, double b, double relerr=1e-5, double abserr=1e-8);
bool allClose(const double *a, const double *b, int size);
void printValues(double truth, double result);
void printMatrices(const double *truth, const double *result,
    int nrows, int ncols);
