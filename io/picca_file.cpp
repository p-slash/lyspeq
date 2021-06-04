#include "io/picca_file.hpp"

#include <cstdio>
#include <cmath>
#include <stdexcept>

// #include "io/io_helper_functions.hpp"
#include "core/fiducial_cosmology.hpp"


#define LN10 2.30258509299
#define ONE_SIGMA_2_FWHM 2.35482004503

PiccaFile::PiccaFile(const char *fname)
{
    int hdutype;

    sprintf(file_name, "%s", fname);

    fits_open_file(&fits_file, fname, READONLY, &status);
    fits_get_num_hdus(fits_file, &no_spectra, &status);
    no_spectra--;

    curr_spec_index=0;
    fits_movabs_hdu(fits_file, 1, &hdutype, &status);
}

void PiccaFile::readParameters(int index, int &N, double &z, int &fwhm_resolution, double &sig2noi, double &dv_kms)
{
    _move(index);

    fits_read_key(fits_file, TINT, "NAXIS2", &N, NULL, &status);

    fits_read_key(fits_file, TDOUBLE, "Z", &z, NULL, &status);

    double r_kms;
    fits_read_key(fits_file, TDOUBLE, "MEANRESO", &r_kms, NULL, &status);
    fwhm_resolution = int(SPEED_OF_LIGHT/r_kms/ONE_SIGMA_2_FWHM/100 + 0.5)*100

    fits_read_key(fits_file, TDOUBLE, "MEANSNR", &sig2noi, NULL, &status);

    fits_read_key(fits_file, TDOUBLE, "DLL", &dv_kms, NULL, &status);
    dv_kms *= SPEED_OF_LIGHT / LN10;

    curr_N = N;
}

void PiccaFile::readData(int index, int N, double *lambda, double *delta, double *noise)
{
    int nonull;
    _move(index);

    fits_read_col(fits_file, TDOUBLE, 1, 1, 1, N, 0, lambda, &nonull, &status);
    fits_read_col(fits_file, TDOUBLE, 2, 1, 1, N, 0, delta, &nonull, &status);
    fits_read_col(fits_file, TDOUBLE, 3, 1, 1, N, 0, noise, &nonull, &status);
    
    std::for_each(lambda, lambda+N, [](double &ld) { ld = pow(10, ld); });
    std::for_each(noise, noise+N, [](double &ld) { ld = pow(ld, -0.5); });
}

void PiccaFile::readResolutionMatrix(int index, int N, double *Rmat, int &mdim)
{
    int naxis;
    long *naxes = new long[2];
    
    _move(index);

    fits_read_tdim(fits_file, 6, N, &naxis, naxes, &status);
    mdim = naxes[0];

    fits_read_col(fits_file, TDOUBLE, 6, 1, 1, N*mdim, 0, Rmat, &nonull, &status);
}

PiccaFile::~PiccaFile()
{
    fits_close_file(fits_file, &status);
}

void PiccaFile::_move(int index)
{
    int hdutype;
    if (index >= no_spectra)
        std::runtime_error("Trying go beyond # HDU in fits file.");

    if (curr_spec_index != index)
    {
        fits_movabs_hdu(fits_file, index+1, &hdutype, &status);
        curr_spec_index = index;
    }
}