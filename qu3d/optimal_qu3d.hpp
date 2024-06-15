#ifndef OPTIMAL_QU3D_H
#define OPTIMAL_QU3D_H

#include <vector>
#include <string>
#include <memory>
// #include <unordered_map>

#include "mathtools/real_field_3d.hpp"

#include "io/config_file.hpp"
#include "io/qso_file.hpp"

// typedef std::unordered_map<long, std::unique_ptr<qio::QSOFile>> targetid_quasar_map;

const config_map qu3d_default_parameters ({
    {"NGRID_X", "1024"}, {"NGRID_Y", "512"}, {"NGRID_Z", "48"},
    {"LENGTH_X", "45000"}, {"LENGTH_Y", "25000"}, {"LENGTH_Z", "2000"},
    {"ZSTART", "5200"}, {"NumberOfIterations", "5"}
});


class Qu3DEstimator
{
    std::vector<std::unique_ptr<qio::QSOFile>> quasars;
    int num_iterations;
    RealField3D mesh;
    // targetid_quasar_map quasars;
    // Reads the entire file
    void _readOneDeltaFile(const std::string &fname);
    void _readQSOFiles(const std::string &flist, const std::string &findir);
public:
    /* This function reads following keys from config file:
    FileNameList: string
        File to spectra to list. Filenames are wrt FileInputDir.
    FileInputDir: string
        Directory where files reside.
    */
    Qu3DEstimator(ConfigFile &config);

    void multiplyCov_x_Vector(double *y);
    void estimate();
};

#endif
