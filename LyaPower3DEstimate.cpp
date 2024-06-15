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

    if (argc < 2)
    {
        fprintf(stderr, "Missing config file!\n");
        return 1;
    }

    const char *FNAME_CONFIG = argv[1];

    gsl_set_error_handler_off();

    ConfigFile config = ConfigFile();
    std::unique_ptr<Qu3DEstimator> qps;

    return 0;
}










