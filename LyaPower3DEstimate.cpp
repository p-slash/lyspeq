#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

#include <gsl/gsl_errno.h>

#include "core/mpi_manager.hpp"

#include "io/logger.hpp"
#include "io/io_helper_functions.hpp"
#include "io/config_file.hpp"

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

    try
    {
        config.readFile(FNAME_CONFIG);
        LOG::LOGGER.open(config.get("OutputDir", "."), mympi::this_pe);
        specifics::printBuildSpecifics();
        mytime::writeTimeLogHeader();
    }
    catch (std::exception& e)
    {
        fprintf(stderr, "Error while reading config file: %s\n", e.what());
        myomp::clean_fftw();
        mympi::finalize();
        return 1;
    }

    try
    {
        process::readProcess(config);
        bins::readBins(config);
        specifics::readSpecifics(config);
        // conv::readConversion(config);
        // fidcosmo::readFiducialCosmo(config);
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR("Error while parsing config file: %s\n",
            e.what());
        myomp::clean_fftw();
        mympi::finalize();
        return 1;
    }

    Qu3DEstimator qps(config);
    config.checkUnusedKeys();

    qps.estimatePower();

    if (qps.total_bias_enabled)
        qps.estimateTotalBiasMc();

    if (qps.noise_bias_enabled)
        qps.estimateNoiseBiasMc();

    if (qps.fisher_rnd_enabled) {
        qps.estimateFisherFromRndDeriv();
        qps.filter();
    }

    qps.write();
    myomp::clean_fftw();
    mympi::finalize();
    return 0;
}










