#include "mathtools/stats.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>


double stats::medianOfSortedArray(const double *sorted_arr, int size) {
    int jj = size / 2;
    double median = sorted_arr[jj];

    if (size % 2 == 0) {
        median += sorted_arr[jj - 1];
        median /= 2;
    }

    return median;
}


double stats::medianOfUnsortedVector(std::vector<double> &v) {
    std::sort(v.begin(), v.end());

    int jj = v.size() / 2;
    double median = v[jj];

    if (v.size() % 2 == 0) {
        median += v[jj - 1];
        median /= 2;
    }

    return median;
}


double stats::meanBelowThreshold(const double *x, int size, int incx, double t) {
    double sum = 0;
    int c = 0;

    for (int i = 0; i < size; i += incx) {
        if (x[i] > t) continue;

        sum += x[i];
        ++c;
    }

    if (c == 0) return 0;

    return sum / c;
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
