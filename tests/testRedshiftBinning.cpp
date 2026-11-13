#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "core/global_numbers.hpp"

namespace {
void setBins(int nz, double z0, double dz, bins::BinningMethod method) {
    bins::NUMBER_OF_Z_BINS = nz;
    bins::NZ_MINUS_ONE = nz - 1;
    bins::Z_BIN_WIDTH = dz;
    bins::Z_BINNING_METHOD = method;
    bins::ZBIN_CENTERS.clear();
    for (int zm = 0; zm < nz; ++zm)
        bins::ZBIN_CENTERS.push_back(z0 + dz * zm);
}

// Sums weights over all z bins on a sorted z grid. Weights outside
// [low, up) are zero by contract and must not be read.
int checkPartitionOfUnity(const char *name) {
    const int nz = bins::NUMBER_OF_Z_BINS;
    const double dz = bins::Z_BIN_WIDTH, z0 = bins::ZBIN_CENTERS[0];
    const double zmin = z0 - 3 * dz, zmax = bins::ZBIN_CENTERS[nz - 1] + 3 * dz;
    const int N = 2001;

    std::vector<double> z(N), sum(N, 0.), w(N);
    for (int i = 0; i < N; ++i)
        z[i] = zmin + (zmax - zmin) * i / (N - 1);

    for (int zm = 0; zm < nz; ++zm) {
        int low = 0, up = 0;
        std::fill(w.begin(), w.end(), 0.);
        bins::setRedshiftBinningFunction(zm);
        bins::redshiftBinningFunction(z.data(), N, zm, w.data(), low, up);

        for (int i = low; i < up; ++i) {
            if (w[i] < -1e-14 || w[i] > 1 + 1e-14) {
                printf("%s: nz=%d zm=%d weight %g out of [0,1] at z=%g\n",
                       name, nz, zm, w[i], z[i]);
                return 1;
            }
            sum[i] += w[i];
        }
    }

    for (int i = 0; i < N; ++i) {
        if (std::abs(sum[i] - 1.) > 1e-12) {
            printf("%s: nz=%d sum of weights = %.15g at z=%g\n",
                   name, nz, sum[i], z[i]);
            return 1;
        }
    }
    return 0;
}
}

int main() {
    int r = 0;
    for (int nz : {1, 2, 3, 4, 5, 8}) {
        setBins(nz, 2.2, 0.2, bins::TSCBinningMethod);
        r += checkPartitionOfUnity("TSC");
        setBins(nz, 2.2, 0.2, bins::TriangleBinningMethod);
        r += checkPartitionOfUnity("Triangular");
    }

    if (r == 0)
        printf("Redshift binning partition of unity passed.\n");
    return r;
}
