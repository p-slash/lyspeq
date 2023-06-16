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
    };

    Progress& operator++() {
        LOG::LOGGER.DEB("One done.\n");
        ++pcounter;
        int curr_progress = (100 * pcounter) / size;

        if (curr_progress - last_progress >= thres) {
            last_progress = curr_progress;

            double time_passed_progress =
                mytime::timer.getTime() - init_time;
            double remain_progress =
                time_passed_progress * (size - pcounter) / pcounter;

            LOG::LOGGER.STD(
                "Progress: %d%%. Elapsed: %.1f mins. Remaining: %.1f mins.\n",
                curr_progress, time_passed_progress, remain_progress);
        }
        return *this;
    };

private:
    int size, last_progress, pcounter, thres;
    double init_time;
};

#endif
