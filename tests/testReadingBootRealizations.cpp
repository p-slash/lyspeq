#include <fitsio.h>
#include <cstdio>
#include <memory>


void checkError(int status) {
    if (status == 0)
        return;

    char fits_msg[80];
    fits_get_errstatus(status, fits_msg);
    printf("Error: %s\n", fits_msg);
}


int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Missing fits file!\n");
        return 1;
    }

    fitsfile *fits_file;
    char *fname = argv[1];
    int status = 0, nboots, nk, nz, nfound, fastbootstrap;
    long naxes[2];

    fits_open_file(&fits_file, fname, READONLY, &status);
    checkError(status);

    char extname[] = "REALIZATIONS";
    fits_movnam_hdu(fits_file, IMAGE_HDU, extname, 0, &status);
    checkError(status);

    fits_read_keys_lng(fits_file, "NAXIS", 1, 2, naxes, &nfound, &status);
    fits_read_key(fits_file, TINT, "NBOOTS", &nboots, nullptr, &status);
    printf("nboots: %d\n", nboots);
    fits_read_key(fits_file, TINT, "NK", &nk, nullptr, &status);
    printf("nk: %d\n", nk);
    fits_read_key(fits_file, TINT, "NZ", &nz, nullptr, &status);
    printf("nz: %d\n", nz);
    fits_read_key(
        fits_file, TLOGICAL, "FASTBOOT", &fastbootstrap, nullptr, &status);
    checkError(status);

    printf("%d %d %d\n", nz, nboots, nk);
    if (fastbootstrap) printf("Fastboot\n");

    if (naxes[0] == nk * nz) printf("naxes[0] good\n");
    if (naxes[1] == nboots) printf("naxes[1] good\n");

    long size = naxes[0] * naxes[1];
    double nullval = 0;
    auto data = std::make_unique<double[]>(naxes[0] * naxes[1]);
    fits_read_img(
        fits_file, TDOUBLE, 1, size, &nullval, data.get(), nullptr, &status);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < nk * nz; ++j)
            printf("%.2e  ", data[j + i * naxes[0]]);
        printf("\n");
    }
    

    char extname2[] = "SOLV_INVF";
    fits_movnam_hdu(fits_file, IMAGE_HDU, extname2, 0, &status);
    fits_read_keys_lng(fits_file, "NAXIS", 1, 2, naxes, &nfound, &status);
    if (naxes[0] == nk * nz) printf("naxes[0] good\n");
    if (naxes[1] == nk * nz) printf("naxes[1] good\n");

    size = naxes[0] * naxes[1];
    data.reset();
    data = std::make_unique<double[]>(naxes[0] * naxes[1]);
    fits_read_img(
        fits_file, TDOUBLE, 1, size, &nullval, data.get(), nullptr, &status);

    for (int i = 0; i < nk * nz; ++i) {
        for (int j = 0; j < nk * nz; ++j)
            printf("%.2e  ", data[j + i * naxes[0]]);
        printf("\n");
    }

    checkError(status);
    fits_close_file(fits_file, &status);
    return 0;
}

