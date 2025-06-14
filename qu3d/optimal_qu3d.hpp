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
    {"MatchCellSizeOfZToXY", "-1"},
    {"TurnOnPpCovariance", "-1"}, {"NumberOfMultipoles", "4"},
    {"MaxConjGradSteps", "5"}, {"MaxMonteCarlos", "100"},
    {"MinimumRa", "0.0"}, {"MaximumRa", "360.0"},
    {"MinimumDec", "-90.0"}, {"MaximumDec", "90.0"},
    {"MinimumKperp", "0"}, {"MinimumKlos", "0"},
    {"ConvergenceTolerance", "1e-6"}, {"AbsoluteTolerance", "-1"},
    {"LongScale", "50"}, {"ScaleFactor", "4"},
    {"DownsampleFactor", "3"}, {"TestGaussianField", "-1"},
    {"MockGridResolutionFactor", "1"},
    {"EstimateTotalBias", "1"}, {"EstimateTotalBiasDirectly", "1"},
    {"EstimateNoiseBias", "1"}, {"EstimateFisherDirectly", "-1"},
    {"EstimateMaxEigenValues", "-1"}, {"TestSymmetry", "-1"}, {"Seed", "6722"},
    {"PadeOrder", "4"},
    {"TestHsqrt", "-1"}, {"UniquePrefixTmp", ""}, {"NeighborsCache", ""}
});


class Qu3DEstimator
{
    ConfigFile &config;

    bool pp_enabled, absolute_tolerance;
    int max_conj_grad_steps, max_monte_carlos, number_of_multipoles,
        pade_order;
    double tolerance, mc_tol, radius, rscale_factor, effective_chi;
    size_t num_all_pixels;

    std::function<void()> updateYMatrixVectorFunction;

    std::vector<std::unique_ptr<CosmicQuasar>> quasars;
    std::unique_ptr<std::seed_seq> seed_generator;
    std::unique_ptr<ioh::Qu3dFile> result_file;

    std::vector<MyRNG> rngs;
    RealField3D mesh, mesh_rnd, mesh_fh;

    std::unique_ptr<double[]>
        mc1, mc2, mesh_z1_values,
        raw_power, filt_power, raw_bias, filt_bias,
        fisher, covariance;

    void _initRngs(std::seed_seq *seq) {
        const int N = myomp::getMaxNumThreads();
        rngs.resize(N);
        std::vector<size_t> seeds(N);
        seq->generate(seeds.begin(), seeds.end());
        for (int i = 0; i < N; ++i)
            rngs[i].seed(seeds[i]);
    }
    // targetid_quasar_map quasars;
    // Reads the entire file
    void _readOneDeltaFile(const std::string &fname);
    void _readQSOFiles(const std::string &flist, const std::string &findir);
    void _openResultsFile();

    void _calculateBoxDimensions(float L[3], float &z0);
    void _setupMesh(double radius);
    void _constructMap();
    void _findNeighbors();
    void _saveNeighbors();
    void _readNeighbors(const std::string &neighbors_file);
    void _createRmatFiles(const std::string &prefix);

    bool _syncMonteCarlo(int nmc, double *o1, double *o2,
                         int ndata, const std::string &ext);

public:
    int mock_grid_res_factor;
    bool total_bias_enabled, total_bias_direct_enabled, noise_bias_enabled,
         fisher_direct_enabled, max_eval_enabled;

    /* This function reads following keys from config file:
    FileNameList: string
        File to spectra to list. Filenames are wrt FileInputDir.
    FileInputDir: string
        Directory where files reside.
    */
    Qu3DEstimator(ConfigFile &configg);

    // These functions are in optimal_qu3d_mc.cpp
    /* Multiply (m I + H) (*sc_eta) = (*out)
       input is const *in, output is *out, uses: *in_isig */
    double findMaxDiagonalAs();
    void conjugateGradientIpH(double m, double s=1.0);
    void multiplyCovSmallSqrt();
    void multiplyCovSmallSqrtPade(int pade_order);
    void replaceDeltasWithGaussianField();
    void replaceDeltasWithHighResGaussianField();
    void estimateNoiseBiasMc();
    void estimateTotalBiasMc();
    void estimateTotalBiasDirect();
    void testCovSqrt();
    void estimateFisherFromRndDeriv();
    void multiplyFisherDerivs(double *o1, double *o2);
    void estimateFisherDirect();

    void multiplyAsVector(double m=0, double s=1.0);
    void multiplyNewtonSchulzY(int n, double s);
    void multiplyNewtonSchulzZ(int n, double s);
    double estimateMaxEvalAs(double m=0);
    void multiplyCovSmallSqrtNewtonSchulz(int order);

    // These functions are in extra.cpp
    /* This is called only for small-scale direct multiplication. */
    double updateRng(double residual_norm2);
    void calculateNewDirection(double beta);
    void conjugateGradientSampler();
    void dumpSearchDirection();
    void testSymmetry();
    void estimateMaxEvals();
    double estimateFrobeniusNormAs();

    // These are in original source file
    void multMeshComp();
    void multParticleComp();
    // from mesh_rnd to *truth
    void multDerivMatrixVec(int i);
    /* Multiply each quasar's *in pointer and save to *out pointer.
       (I + N^-1/2 S N^-1/2) z = out */
    void multiplyCovVector(bool mesh_enabled=true);
    void multiplyDerivVectors(
        double *o1, double *o2, double *lout, const RealField3D &other);
    void multiplyDerivVectors(double *o1, double *o2, double *lout=nullptr) {
        multiplyDerivVectors(o1, o2, lout, mesh);
    };

    /* Reverse interopates qso->in onto the mesh */
    void reverseInterpolate(RealField3D &m);
    /* Reverse interopates qso->in times G^1/2(z) onto the mesh */
    void reverseInterpolateZ(RealField3D &m);
    /* Reverse interopates qso->in x qso->isig onto the mesh */
    void reverseInterpolateIsig(RealField3D &m);
    double updateY(double residual_norm2);
    /* Solve (I + N^-1/2 S N^-1/2) z = m, until z converges,
    where y = N^-1/2 z and m = truth = N^-1/2 delta. Then get y if z2y=true.
    */
    void conjugateGradientDescent();
    void preconditionerSolution();
    void estimatePower();

    void filter();
    void write();
};

#endif
