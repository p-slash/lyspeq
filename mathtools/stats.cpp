#include "mathtools/stats.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>


double stats::medianOfSortedArray(const double *sorted_arr, int size) {
    if (size % 2 == 0) {
        int jj = size / 2;
        return (sorted_arr[jj - 1] + sorted_arr[jj]) / 2;
    } else {
        int jj = size / 2;
        return sorted_arr[jj];
    }
}


double stats::medianOfUnsortedVector(std::vector<double> &v) {
    std::sort(v.begin(), v.end());
    return stats::medianOfSortedArray(v.data(), v.size());
}


double stats::medianOfUnsortedVector(double *v, int size) {
    std::sort(v, v + size);
    return stats::medianOfSortedArray(v, size);
}


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
