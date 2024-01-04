#ifndef STATS_H
#define STATS_H

#include <vector>

namespace stats {
    double medianOfSortedArray(const double *sorted_arr, int size);
    double medianOfUnsortedVector(std::vector<double> &v);
    void medianOffBalanceStats(
        std::vector<double> &v, double &med_offset, double &max_diff_offset
    );
}

#endif
