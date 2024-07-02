#ifndef PROGRESS_H
#define PROGRESS_H

#include "core/global_numbers.hpp"
#include "io/logger.hpp"


class Progress {
public:
    Progress(int total, int percThres=5) {
        size = total;
        last_progress = 0;
        pcounter = 0;
        thres = percThres;
        init_time = mytime::timer.getTime();
        last_weighted_time = 0;
        last_weight = 0;
    };

    Progress& operator++() {
        ++pcounter;
        int curr_progress = (100 * pcounter) / size;

        if (curr_progress - last_progress >= thres) {
            last_progress = curr_progress;

            double time_passed_progress =
                mytime::timer.getTime() - init_time;

            double curr_total_time = (
                time_passed_progress * size + last_weighted_time
            ) / (pcounter + last_weight);

            double remain_progress = curr_total_time - time_passed_progress;

            last_weight = pcounter * 0.8;
            last_weighted_time = curr_total_time * last_weight;

            LOG::LOGGER.STD(
                "Progress: %3d%%. Elapsed: %5.1f mins. Remaining: %5.1f mins. "
                "Total: %5.1f mins.\n",
                curr_progress, time_passed_progress, remain_progress,
                curr_total_time);
        }
        return *this;
    };

    void reset(int percThres=-1) {
        last_progress = 0;
        pcounter = 0;
        init_time = mytime::timer.getTime();
        last_weighted_time = 0;
        last_weight = 0;
        if (percThres > 0)
            thres = percThres;
    }

private:
    int size, last_progress, pcounter, thres;
    double init_time, last_weighted_time, last_weight;
};

#endif
