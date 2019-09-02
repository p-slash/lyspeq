#include "core/matrix_helper.hpp"
#include <gsl/gsl_cblas.h>
#include <cmath>

#define N 8
int main()
{
    double A[N], B[N], C=0, r=-10, rel_err = 0;
    for (int i = 0; i < N; ++i)
    {
        A[i] = 0.1*i;
        B[i] = 2.5*i + 1;
        C += A[i] * B[i];
    }

    r = cblas_ddot(N, A, 1, B, 1);
    printf("True: %e\nCBLAS: %e\n", C, r);

    rel_err = fabs(C-r)/C;
    if (rel_err > 1e-8)
    {
        fprintf(stderr, "True and CBLAS does not match. Rel error: %e.\n", rel_err);
        return 1;
    }

    return 0;
}
