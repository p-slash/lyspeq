#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

#include "core/bootstrapper.hpp"
#include "core/global_numbers.hpp"
#include "core/fiducial_cosmology.hpp"
#include "core/one_qso_estimate.hpp"
#include "core/omp_manager.hpp"
#include "core/progress.hpp"
#include "core/sq_table.hpp"

#include "mathtools/discrete_interpolation.hpp"
#include "mathtools/matrix_helper.hpp"
#include "mathtools/smoother.hpp"
#include "mathtools/stats.hpp"

#include "io/bootstrap_file.hpp"
#include "io/io_helper_functions.hpp"
#include "io/logger.hpp"
#include "io/qso_file.hpp"


inline void usage() {
    fprintf(stdout, "Usage: lyspeqBootStats FNAME_BOOT FNAME_BASE.\n");
}


int printUse(char *argv[]) {
    std::string opt = argv[1];
    if (opt == "-h" || opt == "--help") { usage(); return 1; }
    return 0;
}

int main(int argc, char *argv[]) {
    process::this_pe = 0;
    process::total_pes = 1;

    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Missing arguments!\n");
        usage();
        return 1;
    }
    if (printUse(argv)) return 0;

    const std::string FNAME_BOOT = argv[1];
    process::FNAME_BASE = argv[2];

    PoissonBootstrapper pbooter(FNAME_BOOT);
    pbooter.run();

    return 0;
}
