#ifndef OPTIMAL_QU3D_H
#define OPTIMAL_QU3D_H

#include <vector>
#include <string>
#include <memory>
// #include <unordered_map>

#include "mathtools/real_field_3d.hpp"

#include "io/config_file.hpp"
#include "qu3d/cosmic_quasar.hpp"

// typedef std::unordered_map<long, std::unique_ptr<qio::QSOFile>> targetid_quasar_map;

const config_map qu3d_default_parameters ({
    {"NGRID_X", "1024"}, {"NGRID_Y", "288"}, {"NGRID_Z", "48"},
    {"LENGTH_X", "45000"}, {"LENGTH_Y", "14000"}, {"LENGTH_Z", "2000"},
    {"ZSTART", "5200"}, {"MaxConjGradSteps", "5"}, {"MaxMonteCarlos", "100"},
    {"ConvergenceTolerance", "1e-6"}, {"LongScale", "50"}
});


class Qu3DEstimator
{
    std::vector<std::unique_ptr<CosmicQuasar>> quasars;
    size_t num_all_pixels;
    int max_conj_grad_steps, max_monte_carlos;
    double tolerance, rscale_long;
    RealField3D mesh;

    std::unique_ptr<double[]> power_est, bias_est, fisher;
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

    void reverseInterpolate() {
        double coord[3];
        mesh.zero_field_k();
        for (auto &qso : quasars) {
            for (int i = 0; i < qso->N; ++i) {
                qso->getCartesianCoords(i, coord);
                mesh.reverseInterpolate(coord, qso->in[i]);
            }
        }
    }

    void multMeshComp();
    void multParticleComp();

    /* Multiply each quasar's *in pointer and save to *out pointer. */
    void multiplyCovVector() {
        // init new results to Cy = I.y
        #pragma omp parallel for
        for (auto &qso : quasars)
            std::copy_n(qso->in, qso->N, qso->out);

        // Add long wavelength mode to Cy
        multMeshComp();
        // multParticleComp();
    }

    void multiplyDerivVector(int iperp, int iz);

    /* Return residual^T . residual */
    double calculateResidualNorm2();
    void updateY(double residual_norm2);
    void calculateNewDirection(double beta);
    void conjugateGradientDescent();

    void estimatePowerBias();
};

#endif
