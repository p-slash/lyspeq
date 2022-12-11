#ifndef SMOOTHER_H
#define SMOOTHER_H

#include <memory>
#include "io/config_file.hpp"

const config_map smoother_default_parameters ({
        {"SmoothNoiseWeights", "-1"} });

class Smoother
{
    static const int HWSIZE = 25, KS = 2*HWSIZE+1;

    int sigmapix;
    double gaussian_kernel[KS];
    bool useMedianNoise, isSmoothingOn;

public:
    /* This function reads following keys from config file:
    SmoothNoiseWeights: int (Default: -1)
        If > 0, sets sigmapix for smooting. If <0, turns off smoothing.
        If equals to 0, smoothing uses the median noise.
    */
    Smoother(ConfigFile &config);
    Smoother(Smoother &&rhs) = delete;
    Smoother(const Smoother &rhs) = delete;

    void smoothNoise(const double *n2, double *out, int size);
};

namespace process
{
    extern std::unique_ptr<Smoother> noise_smoother;
}

#endif
