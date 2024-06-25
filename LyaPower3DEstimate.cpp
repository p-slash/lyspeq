#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

#include <gsl/gsl_errno.h>

#include "io/logger.hpp"
#include "io/io_helper_functions.hpp"
#include "io/config_file.hpp"

#include "qu3d/optimal_qu3d.hpp"


int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Missing config file!\n");
        return 1;
    }

    const char *FNAME_CONFIG = argv[1];

    myomp::init_fftw();
    gsl_set_error_handler_off();

    ConfigFile config = ConfigFile();
    std::unique_ptr<Qu3DEstimator> qps;

    try
    {
        config.readFile(FNAME_CONFIG);
        LOG::LOGGER.open(config.get("OutputDir", "."), 0);
        specifics::printBuildSpecifics();
        mytime::writeTimeLogHeader();
    }
    catch (std::exception& e)
    {
        fprintf(stderr, "Error while reading config file: %s\n", e.what());
        myomp::clean_fftw();
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
        return 1;
    }


    qps = std::make_unique<Qu3DEstimator>(config);
    config.checkUnusedKeys();

    qps->estimatePower();
    qps->estimateBiasMc();
    qps->estimateFisher();
    qps->write();
    myomp::clean_fftw();
    return 0;
}










