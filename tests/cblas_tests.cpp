#include "core/matrix_helper.hpp"
#include <gsl/gsl_cblas.h>

#define N 8
int main()
{
    double A[N], B[N], C=0, r=0;
    for (int i = 0; i < N; ++i)
    {
        A[i] = 0.1*i;
        B[i] = 2.5*i + 1;
        C += A[i] * B[i];
    }

    r = cblas_ddot(N, A, 1, B, 1);
    printf("True: %e\nCBLAS: %e\n", C, r);

    return 0;
}
