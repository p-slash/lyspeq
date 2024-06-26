#ifndef OPTIMAL_QU3D_H
#define OPTIMAL_QU3D_H

#include <vector>
#include <string>
#include <memory>

#include "mathtools/real_field_3d.hpp"

#include "io/config_file.hpp"
#include "qu3d/cosmic_quasar.hpp"

const config_map qu3d_default_parameters ({
    {"NGRID_X", "1024"}, {"NGRID_Y", "288"}, {"NGRID_Z", "48"},
    {"LENGTH_X", "45000"}, {"LENGTH_Y", "14000"}, {"LENGTH_Z", "2000"},
    {"ZSTART", "5200"}, {"MaxConjGradSteps", "5"}, {"MaxMonteCarlos", "100"},
    {"ConvergenceTolerance", "1e-6"}, {"LongScale", "50"}, {"ScaleFactor", "4"},
    {"DownsampleFactor", "3"}
});


class Qu3DEstimator
{
    ConfigFile &config;
    std::vector<std::unique_ptr<CosmicQuasar>> quasars;
    size_t num_all_pixels;
    int max_conj_grad_steps, max_monte_carlos;
    double tolerance, radius, rscale_factor;
    RealField3D mesh;

    std::unique_ptr<double[]>
        raw_power, filt_power, raw_bias, filt_bias,
        fisher, covariance;
    // targetid_quasar_map quasars;
    // Reads the entire file
    void _readOneDeltaFile(const std::string &fname);
    void _readQSOFiles(const std::string &flist, const std::string &findir);

    void _constructMap();
    void _findNeighbors();
public:
    /* This function reads following keys from config file:
    FileNameList: string
        File to spectra to list. Filenames are wrt FileInputDir.
    FileInputDir: string
        Directory where files reside.
    */
    Qu3DEstimator(ConfigFile &configg);

    void preconditionJacobi() {};
    void multMeshComp();
    void multParticleComp();

    /* Reverse interopates qso->in onto the mesh */
    void reverseInterpolate();
    /* Reverse interopates qso->in x qso->isig onto the mesh */
    void reverseInterpolateIsig();

    /* Multiply each quasar's *in pointer and save to *out pointer.
       (I + N^-1/2 S N^-1/2) z = out */
    void multiplyCovVector() {
        // init new results to Cy = I.y
        #pragma omp parallel for
        for (auto &qso : quasars)
            std::copy_n(qso->in, qso->N, qso->out);

        // Add long wavelength mode to Cy
        multMeshComp();
        // multParticleComp();
    }

    double multiplyDerivVector(int iperp, int iz);

    /* Return residual^T . residual */
    double calculateResidualNorm2();
    void updateY(double residual_norm2);
    void calculateNewDirection(double beta);

    /* Solve (I + N^-1/2 S N^-1/2) z = m, until z converges,
    where y = N^-1/2 z and m = N^-1/2 delta. Then get y.
    */
    void conjugateGradientDescent();

    void estimatePower();
    void estimateBiasMc();


    void drawRndDeriv(int i);
    void estimateFisher();

    void filter();
    void write();
};

#endif
