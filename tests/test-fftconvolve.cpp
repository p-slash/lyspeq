#include <cstdio>

#include "io/qso_file.hpp"

int main(int argc, char *argv[])
{
    char *fname = argv[1];

    qio::QSOFile qFile(fname, qio::Picca);
    qFile.readParameters();
    qFile.readData();

    qFile.readAllocResolutionMatrix();
    qFile.Rmat->oversample(3, qFile.dlambda);
    qFile.Rmat->fprintfMatrix("debugoutput-oversampled-resomat.txt");
    return 0;
}

