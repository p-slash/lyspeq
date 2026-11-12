#include <string>
extern std::string SRCDIR;

bool isClose(double a, double b, double relerr=1e-5, double abserr=1e-8);
bool allClose(const double *a, const double *b, int size, double relerr=1e-5, double abserr=1e-8);
void getMaxDiffWithIndex(const double *a, const double *b, int size, double &max_diff, int &max_index);
void printValues(double truth, double result);
void printMatrices(const double *truth, const double *result,
    int nrows, int ncols);
