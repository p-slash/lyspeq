#include <algorithm>

#include "mathtools/fftlog.hpp"
#include "mathtools/matrix_helper.hpp"
#include "tests/test_utils.hpp"


int test_rotate() {
    std::vector<int> v{0, 1, 2, 3, 4, -3, -2, -1};
    std::rotate(v.begin(), v.begin() + v.size() / 2 + 1, v.end());
    for (const int &a : v)
        printf("%d ", a);
    printf("\n");
    return 0;
}


int test_fftlog() {
    constexpr int N = 128;
    FFTLog fht(N);

    // r^(mu + 1) * exp(-r^2/2) -> k^mu+1 exp(-k^2/2)
    fht.construct(0, 1e-7, 1e1, 0, -6.0 * log(10.0));
    for (int i = 0; i < N; ++i)
        fht.field[i] = fht.r[i] * std::exp(-fht.r[i] * fht.r[i] / 2.0);

    fht.transform();

    printf("lnkcrc: %.5e\n", fht.getLnKcRc());
    printf("k \t exp \t FHT.real\n");
    for (int i = 0; i < N; ++i)
        printf(
            "%d %.3e \t %.3e \t %.3e\n",
               i, fht.k[i], fht.k[i] * std::exp(-fht.k[i] * fht.k[i] / 2.0),
               std::real(fht.field[i]));

    return 0;
}


int main() {
    int r = 0;

    r += test_rotate();
    r += test_fftlog();

    return r;
}
