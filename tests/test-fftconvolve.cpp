#include <cstdio>

#include "io/qso_file.hpp"

int main(int argc, char *argv[])
{
    if (argc<2)
    {
        fprintf(stderr, "Missing fits file!\n");
        return -1;
    }

    specifics::OVERSAMPLING_FACTOR = 4;

    auto fname = std::string(argv[1]);

    qio::QSOFile qFile(fname, qio::Picca);
    qFile.readParameters();
    qFile.readData();

    qFile.readAllocResolutionMatrix();
    qFile.recalcDvDLam();  // meansnr: 0.67616031758204
    printf("Rkms: %.2f\n", qFile.R_kms);  // 47.5047929893881
    // Ivar: [0.51403648, 0.03894964, 0.54754096, 0.89146243, 0.98388764]
    qFile.Rmat->oversample(specifics::OVERSAMPLING_FACTOR, qFile.dlambda);
    qFile.recalcDvDLam();
    printf("Rkms: %.2f\n", qFile.R_kms);  // 47.5047929893881
    qFile.Rmat->fprintfMatrix("debugoutput-oversampled-resomat.txt");
    return 0;
}

