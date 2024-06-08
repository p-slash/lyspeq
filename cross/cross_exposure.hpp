#ifndef CROSS_EXPOSURE_ESTIMATE_H
#define CROSS_EXPOSURE_ESTIMATE_H

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

#include "core/quadratic_estimate.hpp"
#include "cross/one_qso_exposures.hpp"


typedef std::unordered_map<long, std::unique_ptr<OneQsoExposures>> targetid_quasar_map;

const config_map xe_default_parameters ({
    {"DifferentNight", "1"}, {"DifferentFiber", "-1"},
    {"DifferentPetal", "-1"}, {"MinXWaveOverlapRatio", "0.6"}
});


class OneDCrossExposureQMLE: public OneDQuadraticPowerEstimate
{
    targetid_quasar_map quasars;
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

    Does not support Oversampling!
    */
    OneDCrossExposureQMLE(ConfigFile &con) : OneDQuadraticPowerEstimate(con) {
        if (specifics::OVERSAMPLING_FACTOR > 0)
            throw std::invalid_argument(
                "xQMLE does not support oversampling using OversampleRmat.");

        config.addDefaults(xe_default_parameters);
        specifics::X_WAVE_OVERLAP_RATIO = config.getDouble("MinXWaveOverlapRatio");
        specifics::X_NIGHT = config.getInteger("DifferentNight") > 0;
        specifics::X_FIBER = config.getInteger("DifferentFiber") > 0;
        specifics::X_PETAL = config.getInteger("DifferentPetal") > 0;
    };

    void xQmlEstimate();
};

#endif

