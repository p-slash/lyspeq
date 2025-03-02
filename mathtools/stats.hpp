#ifndef STATS_H
#define STATS_H

#include <algorithm>
#include <vector>

namespace stats {
    enum PAD_MODE { REFLECT, NEAREST };
    static double medianOfSortedArray(const double *sorted_arr, int size) {
        if (size % 2 == 0) {
            int jj = size / 2;
            return (sorted_arr[jj - 1] + sorted_arr[jj]) / 2;
        } else {
            int jj = size / 2;
            return sorted_arr[jj];
        }
    }

    static double medianOfUnsortedVector(std::vector<double> &v) {
        std::sort(v.begin(), v.end());
        return stats::medianOfSortedArray(v.data(), v.size());
    }
    static double medianOfUnsortedVector(double *v, int size) {
        std::sort(v, v + size);
        return stats::medianOfSortedArray(v, size);
    }

    template<PAD_MODE mode=NEAREST>
    std::vector<double> medianFilter(const double *v, int N, int width);

    void medianOffBalanceStats(
        std::vector<double> &v, double &med_offset, double &max_diff_offset
    );
    std::vector<double> getCdfs(double *v, int size, int nsigma=2);
}

#endif
