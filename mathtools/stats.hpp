#ifndef STATS_H
#define STATS_H

#include <vector>

namespace stats {
    double medianOfSortedArray(const double *sorted_arr, int size);
    double medianOfUnsortedVector(std::vector<double> &v);
    double meanBelowThreshold(const double *x, int size, int incx=1, double t=1e4);
    void medianOffBalanceStats(
        std::vector<double> &v, double &med_offset, double &max_diff_offset
    );
}

#endif
