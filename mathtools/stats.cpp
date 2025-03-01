#include "mathtools/stats.hpp"

#include <cmath>
#include <memory>
#include <numeric>


void stats::medianOffBalanceStats(
        std::vector<double> &v, double &med_offset, double &max_diff_offset
) {
    // Obtain some statistics
    // convert to off-balance
    double ave_balance = std::accumulate(v.begin(), v.end(), 0.) / v.size();

    std::for_each(
        v.begin(), v.end(), [ave_balance](double &t) { t = t / ave_balance - 1; }
    );

    // find min and max offset
    auto minmax_off = std::minmax_element(v.begin(), v.end());
    max_diff_offset = *minmax_off.second - *minmax_off.first;

    // convert to absolute values and find find median
    std::for_each(v.begin(), v.end(), [](double &t) { t = fabs(t); });
    med_offset = stats::medianOfUnsortedVector(v);
}


std::vector<double> stats::getCdfs(double *v, int size, int nsigma) {
    std::sort(v, v + size);
    std::vector<double> result(2 * nsigma + 1);

    for (int s = -nsigma; s <= nsigma; s++) {
        double pdf = 0.5 * (1. + erf(s / sqrt(2.)));
        result[s + nsigma] = v[int(pdf)];
    }

    return result;
}


std::vector<double> stats::medianFilter(
        const double *v, int N, int width, PAD_MODE mode
) {
    std::vector<double> out(N);
    auto temp = std::make_unique<double[]>(width);
    int hw = width / 2;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < width; ++j) {
            int k = i - hw + j;

            if (mode == NEAREST) {
                k = std::clamp(k, 0, N - 1);
            }
            else {
                if (k < 0) k = -k;
                if (k > N - 1)  k = 2 * (N - 1) - k;
            }

            temp[j] = v[k];
        }

        out[i] = medianOfUnsortedVector(temp.get(), width);
    }

    return out;
}
