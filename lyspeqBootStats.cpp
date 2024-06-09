#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

#include "core/bootstrapper.hpp"
#include "core/global_numbers.hpp"


inline void usage() {
    fprintf(stdout, "Usage: lyspeqBootStats FNAME_BOOT FNAME_BASE.\n");
}


int printUse(char *argv[]) {
    std::string opt = argv[1];
    if (opt == "-h" || opt == "--help") { usage(); return 1; }
    return 0;
}

int main(int argc, char *argv[]) {
    mympi::this_pe = 0;
    mympi::total_pes = 1;

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
