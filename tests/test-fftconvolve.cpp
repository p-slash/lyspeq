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

    std::vector<double> k_A, window2;

    qFile.calcAverageWindowFunctionFromRMat(k_A, window2);

    for (int i = 0; i < window2.size(); i += 10)
        printf("%.5e  %.5e\n", k_A[i], window2[i]);

    // Ivar: [0.51403648, 0.03894964, 0.54754096, 0.89146243, 0.98388764]
    qFile.Rmat->oversample(specifics::OVERSAMPLING_FACTOR, qFile.dlambda);
    qFile.recalcDvDLam();
    printf("Rkms: %.2f\n", qFile.R_kms);  // 47.5047929893881
    qFile.Rmat->fprintfMatrix("debugoutput-oversampled-resomat.txt");


    qFile.calcAverageWindowFunctionFromRMat(k_A, window2);

    for (int i = 0; i < window2.size(); i += 10)
        printf("%.5e  %.5e\n", k_A[i], window2[i]);
    return 0;
}

