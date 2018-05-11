#include <complex>

#ifndef IO_HELPER_FUNCTIONS_H
#define IO_HELPER_FUNCTIONS_H

void copyArray(const std::complex<double> *source, std::complex<double> *target, long long int size);
void copyArray(const double *source, double *target, long long int size);

FILE * open_file(const char *fname, const char *read_write);

#endif
