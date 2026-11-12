#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <string>

#include <gsl/gsl_errno.h>

#include "core/global_numbers.hpp"
#include "core/mpi_manager.hpp"
#include "core/omp_manager.hpp"
#include "qu3d/optimal_qu3d.hpp"

int main(int argc, char *argv[]) {
    mympi::init(argc, argv);

    if (argc < 2) {
        fprintf(stderr, "Missing config file!\n");
        return 1;
    }

    const char *FNAME_CONFIG = argv[1];

    myomp::init_fftw();
    gsl_set_error_handler_off();

    ConfigFile config = ConfigFile();

    try {
        config.readFile(FNAME_CONFIG);

        const std::string output_dir = config.get("OutputDir", ".");
        bool output_dir_ready = true;

        if (mympi::this_pe == 0) {
            std::error_code ec;
            const bool output_dir_exists = std::filesystem::exists(output_dir, ec);

            if (ec) {
                fprintf(stderr,
                        "Cannot access output directory '%s': %s\n",
                        output_dir.c_str(), ec.message().c_str());
                output_dir_ready = false;
            }
            else if (!output_dir_exists) {
                if (!std::filesystem::create_directories(output_dir, ec)) {
                    fprintf(stderr,
                            "Cannot create output directory '%s': %s\n",
                            output_dir.c_str(), ec.message().c_str());
                    output_dir_ready = false;
                }
            }
            else if (!std::filesystem::is_directory(output_dir, ec)) {
                fprintf(stderr,
                        "Output path '%s' is not a directory.\n",
                        output_dir.c_str());
                output_dir_ready = false;
            }
        }

        mympi::bcast(&output_dir_ready);
        if (!output_dir_ready)
            throw std::runtime_error("Output directory is not available.");

        mympi::barrier();

        LOG::LOGGER.open(output_dir, mympi::this_pe);
        specifics::printBuildSpecifics();
        mytime::writeTimeLogHeader();
    }
    catch (std::exception& e) {
        fprintf(stderr, "Error while reading config file: %s\n", e.what());
        myomp::clean_fftw();
        mympi::finalize();
        return 1;
    }

    try {
        process::readProcess(config);
        bins::readBins(config);
        specifics::readSpecifics(config);
    }
    catch (std::exception& e) {
        LOG::LOGGER.ERR("Error while parsing config file: %s\n",
            e.what());
        myomp::clean_fftw();
        mympi::finalize();
        return 1;
    }

    Qu3DEstimator qps(config);
    bool test_symmetry = config.getInteger("TestSymmetry") > 0;
    bool test_hsqrt = config.getInteger("TestHsqrt") > 0;
    config.checkUnusedKeys();

    if (qps.max_eval_enabled)
        qps.estimateMaxEvals();

    if (test_symmetry)
        qps.testSymmetry();

    if (test_hsqrt)
        qps.testCovSqrt();

    if (qps.test_gaussian_field) {
        if (qps.mock_grid_res_factor > 1) {
            qps.replaceDeltasWithHighResGaussianField();
            goto EndOptimalQu3DNormally;
        }
        else {
            qps.replaceDeltasWithGaussianField();
        }
    }

    qps.estimatePower();

    if (qps.total_bias_enabled)
        qps.estimateTotalBiasMc();

    if (qps.total_bias_direct_enabled)
        qps.estimateTotalBiasDirect();

    if (qps.noise_bias_enabled)
        qps.estimateNoiseBiasMc();

    if (qps.fisher_direct_enabled)
        qps.estimateFisherDirect();

EndOptimalQu3DNormally:
    myomp::clean_fftw();
    mympi::finalize();
    return 0;
}
