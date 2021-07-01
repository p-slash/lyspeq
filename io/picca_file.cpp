#include "io/picca_file.hpp"

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// #include "io/io_helper_functions.hpp"
#include "core/fiducial_cosmology.hpp"


#define LN10 2.30258509299

PiccaFile::PiccaFile(std::string fname_qso)
{
    // Assume fname to be ..fits.gz[1]
    // get hdunum from fname
    file_name = fname_qso; //fname_qso.substr(0, fname_qso.size-3);

    fits_open_file(&fits_file, file_name.c_str(), READONLY, &status);
    fits_get_hdu_num(fits_file, &curr_spec_index);
    fits_get_num_hdus(fits_file, &no_spectra, &status);
    no_spectra--;
    // _move(fname_qso[fname_qso.size-2] - '0');
}

void PiccaFile::readParameters(int &N, double &z, int &fwhm_resolution, double &sig2noi, double &dv_kms)
{
    // _move(newhdu);

    // This is not ndiags in integer, but length in bytes that includes other columns
    // fits_read_key(fits_file, TINT, "NAXIS1", &curr_ndiags, NULL, &status);
    fits_read_key(fits_file, TINT, "NAXIS2", &curr_N, NULL, &status);

    fits_read_key(fits_file, TDOUBLE, "Z", &z, NULL, &status);

    double r_kms;
    fits_read_key(fits_file, TDOUBLE, "MEANRESO", &r_kms, NULL, &status);
    fwhm_resolution = int(SPEED_OF_LIGHT/r_kms/ONE_SIGMA_2_FWHM/100 + 0.5)*100;

    fits_read_key(fits_file, TDOUBLE, "MEANSNR", &sig2noi, NULL, &status);

    fits_read_key(fits_file, TDOUBLE, "DLL", &dv_kms, NULL, &status);
    dv_kms = round(dv_kms*SPEED_OF_LIGHT/LN10/5)*5;

    N = curr_N;
    curr_ndiags = -1;
}

void PiccaFile::readData(double *lambda, double *delta, double *noise)
{
    int nonull, colnum;
    // _move(newhdu);
    fits_get_colnum(fits_file, CASEINSEN, (char*)"LOGLAM", &colnum, &status);
    fits_read_col(fits_file, TDOUBLE, colnum, 1, 1, curr_N, 0, lambda, &nonull, &status);

    fits_get_colnum(fits_file, CASEINSEN, (char*)"DELTA", &colnum, &status);
    fits_read_col(fits_file, TDOUBLE, colnum, 1, 1, curr_N, 0, delta, &nonull, &status);

    fits_get_colnum(fits_file, CASEINSEN, (char*)"IVAR", &colnum, &status);
    fits_read_col(fits_file, TDOUBLE, colnum, 1, 1, curr_N, 0, noise, &nonull, &status);
    
    std::for_each(lambda, lambda+curr_N, [](double &ld) { ld = pow(10, ld); });
    std::for_each(noise, noise+curr_N, [](double &ld) { ld = pow(ld, -0.5); });
}

void PiccaFile::readAllocResolutionMatrix(mxhelp::Resolution *& Rmat)
{
    int nonull, naxis, colnum;
    long *naxes = new long[2];
    fits_get_colnum(fits_file, CASEINSEN, (char*)"RESOMAT", &colnum, &status);
    fits_read_tdim(fits_file, colnum, curr_N, &naxis, naxes, &status);
    curr_ndiags = naxes[0];
    Rmat = new mxhelp::Resolution(curr_N, curr_ndiags);
    
    fits_read_col(fits_file, TDOUBLE, colnum, 1, 1, curr_N*curr_ndiags, 0, Rmat->matrix, &nonull, &status);
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

    if (index > 0 && curr_spec_index != index)
    {
        fits_movabs_hdu(fits_file, index, &hdutype, &status);
        curr_spec_index = index;
    }
}