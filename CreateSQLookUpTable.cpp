// Q matrices are not scaled with redshift binning function
#include <cstdio>
#include <cstdlib>
#include <cstring> // strcmp
#include <stdexcept>

#include <gsl/gsl_errno.h>

#include "core/global_numbers.hpp"
#include "core/mpi_manager.hpp"
#include "core/sq_table.hpp"
#include "core/fiducial_cosmology.hpp"

#include "io/config_file.hpp"
#include "io/logger.hpp"

int main(int argc, char *argv[])
{
    mympi::init(argc, argv);

    if (argc<2)
    {
        fprintf(stderr, "Missing config file!\n");
        mympi::finalize();
        return -1;
    }

    const char *FNAME_CONFIG = argv[1];
    bool force_rewrite = true;

    if (argc == 3)
        force_rewrite = !(strcmp(argv[2], "--unforce") == 0);

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
        return 1;
    }

    try
    {
        process::readProcess(config);
        bins::readBins(config);
        specifics::readSpecifics(config);
        fidcosmo::readFiducialCosmo(config);
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR("Error while parsing config file: %s\n",
            e.what());
        mympi::abort();
        return 1;
    }

    gsl_set_error_handler_off();

    try
    {
        process::sq_private_table = std::make_unique<SQLookupTable>(config);
        const std::vector<std::string> ignored_keys({
            "FileNameList", "FileInputDir", "NumberOfIterations",
            "UseChunksMeanFlux", "InputIsDeltaFlux", "MeanFluxFile",
            "SmoothNoiseWeights", "PrecomputedFisher", "DifferentNight",
            "DifferentFiber", "DifferentPetal", "MinXWaveOverlapRatio"
        });
        config.checkUnusedKeys(ignored_keys);
        process::sq_private_table->computeTables(force_rewrite);
    }
    catch (std::exception& e)
    {   
        LOG::LOGGER.ERR("Error constructing SQ Table: %s\n", e.what());
        mympi::abort();

        return 1;
    }    

    mympi::finalize();

    return 0;
}










