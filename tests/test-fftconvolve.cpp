#include <cstdio>

#include "io/qso_file.hpp"

int main(int argc, char *argv[])
{
    if (argc<2)
    {
        fprintf(stderr, "Missing fits file!\n");
        return -1;
    }

    char *fname = argv[1];

    qio::QSOFile qFile(fname, qio::Picca);
    qFile.readParameters();
    qFile.readData();

    qFile.readAllocResolutionMatrix();
    qFile.Rmat->oversample(3, qFile.dlambda);
    qFile.Rmat->fprintfMatrix("debugoutput-oversampled-resomat.txt");
    return 0;
}

