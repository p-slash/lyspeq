#include <fitsio.h>
#include <cstdio>

#include "io/qso_file.hpp"

int main(int argc, char *argv[])
{
    fitsfile *fits_file;
    char *fname = argv[1];
    int status = 0, N;

    fits_open_file(&fits_file, fname, READONLY, &status);
    fits_read_key(fits_file, TINT, "NAXIS2", &N, NULL, &status);

    int nonull, naxis, colnum;
    long naxes[2];
    char resotmp[]="RESOMAT";
    fits_get_colnum(fits_file, CASEINSEN, resotmp, &colnum, &status);
    fits_read_tdim(fits_file, colnum, N, &naxis, &naxes[0], &status);

    double *matrix = new double[N * naxes[0]];
    fits_read_col(fits_file, TDOUBLE, colnum, 1, 1, N*naxes[0], 0, 
        matrix, &nonull, &status);

    for (int i = 0; i < N; ++i)
        printf("%.3e  ", matrix[i]);

    printf("\n");
    printf("=========================\n");
    double* newmat = new double[N * naxes[0]];

    for (int d = 0; d < naxes[0]; ++d)
        for (int i = 0; i < N; ++i)
            *(newmat + i+d*N) = *(matrix + i*naxes[0]+d);

    int d = 1;
    for (int i = 0; i < N; ++i)
        printf("%.3e  ", newmat[i+d*N]);

    printf("\n");

    delete [] matrix;
    delete [] newmat;
    fits_close_file(fits_file, &status);

    qio::QSOFile qFile(fname, qio::Picca);
    qFile.readParameters();
    qFile.readData();

    qFile.readAllocResolutionMatrix();
    qFile.Rmat->oversample(3, qFile.dlambda);
    qFile.Rmat->fprintfMatrix("debugoutput-oversampled-resomat.txt");
    return 0;
}

