#ifndef CROSS_EXPOSURE_ESTIMATE_H
#define CROSS_EXPOSURE_ESTIMATE_H

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

#include "core/quadratic_estimate.hpp"
#include "cross/one_qso_exposures.hpp"
#include "mathtools/stats.hpp"
#include "io/config_file.hpp"


class OneDCrossExposureQMLE: public OneDQuadraticPowerEstimate
{
    std::unordered_map<long, std::unique_ptr<OneQsoExposures>> quasars;
    // Reads the entire file
    void _readOneDeltaFile(const std::string &fname);
    void _readQSOFiles();
    void _countZbinHistogram();
public:
    /* This function reads following keys from config file:
    NumberOfIterations: int
        Number of iterations. Default 1.
    PrecomputedFisher: string
        File to precomputed Fisher matrix. If present, Fisher matrix is not
            calculated for spectra. Off by default.
    FileNameList: string
        File to spectra to list. Filenames are wrt FileInputDir.
    FileInputDir: string
        Directory where files reside.
    */
    OneDCrossExposureQMLE(ConfigFile &con) : OneDQuadraticPowerEstimate(con) {};

    void iterate();
};

#endif

