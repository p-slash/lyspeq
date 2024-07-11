#ifndef OPTIMAL_QU3D_H
#define OPTIMAL_QU3D_H

#include <vector>
#include <string>
#include <memory>

#include "mathtools/real_field_3d.hpp"

#include "io/config_file.hpp"
#include "qu3d/cosmic_quasar.hpp"
#include "qu3d/qu3d_file.hpp"

const config_map qu3d_default_parameters ({
    {"NGRID_X", "1024"}, {"NGRID_Y", "256"}, {"NGRID_Z", "64"},
    {"TurnOnPpCovariance", "-1"},
    {"MaxConjGradSteps", "5"}, {"MaxMonteCarlos", "100"},
    {"ConvergenceTolerance", "1e-6"}, {"LongScale", "50"}, {"ScaleFactor", "4"},
    {"DownsampleFactor", "3"}, {"TestGaussianField", "-1"}, {"Seed", "6722"}
});


class Qu3DEstimator
{
    ConfigFile &config;

    bool pp_enabled;
    int max_conj_grad_steps, max_monte_carlos;
    double tolerance, radius, rscale_factor;
    size_t num_all_pixels;

    std::vector<std::unique_ptr<CosmicQuasar>> quasars;
    std::unique_ptr<std::seed_seq> seed_generator;
    std::unique_ptr<ioh::Qu3dFile> result_file;
    RealField3D mesh;

    std::unique_ptr<double[]>
        mc1, mc2,
        raw_power, filt_power, raw_bias, filt_bias,
        fisher, covariance;
    // targetid_quasar_map quasars;
    // Reads the entire file
    void _readOneDeltaFile(const std::string &fname);
    void _readQSOFiles(const std::string &flist, const std::string &findir);
    void _openResultsFile();

    void _calculateBoxDimensions(double L[3], double &z0);
    void _setupMesh(double radius);
    void _constructMap();
    void _findNeighbors();

    bool _syncMonteCarlo(int nmc, double *o1, double *o2,
                         int ndata, const std::string &ext);
public:
    /* This function reads following keys from config file:
    FileNameList: string
        File to spectra to list. Filenames are wrt FileInputDir.
    FileInputDir: string
        Directory where files reside.
    */
    Qu3DEstimator(ConfigFile &configg);

    void replaceDeltasWithGaussianField();
    void initGuessDiag();
    void multMeshComp();
    void multParticleComp();

    /* Reverse interopates qso->in onto the mesh */
    void reverseInterpolate();
    /* Reverse interopates qso->in x qso->isig onto the mesh */
    void reverseInterpolateIsig();

    /* Multiply each quasar's *in pointer and save to *out pointer.
       (I + N^-1/2 S N^-1/2) z = out */
    void multiplyCovVector();

    void multiplyDerivVectors(double *o1, double *o2);

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
