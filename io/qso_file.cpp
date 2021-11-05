#include "io/qso_file.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "io/io_helper_functions.hpp"
#include "core/fiducial_cosmology.hpp"

namespace qio
{

// ============================================================================
// Umbrella QSO file
// ============================================================================

// double size, z_qso, snr, dv_kms, dlambda;
// int R_fwhm;
// double *wave, *delta, *noise;
// mxhelp::Resolution *Rmat;
QSOFile::QSOFile(std::string fname_qso, ifileformat p_or_b)
    : PB(p_or_b), pfile(NULL), bqfile(NULL),
    wave(NULL), delta(NULL), noise(NULL), Rmat(NULL)
{
    if (PB == Picca)
        pfile = new PiccaFile(fname_qso);
    else
        bqfile = new BQFile(fname_qso);

    dlambda=-1;
    oversampling=1;
}

QSOFile::~QSOFile()
{
    delete pfile;
    delete bqfile;
    delete Rmat;
    delete [] wave;
    delete [] delta;
    delete [] noise;
}

void QSOFile::readParameters()
{
    if (pfile != NULL)
        pfile->readParameters(size, z_qso, R_fwhm, snr, dv_kms, dlambda, oversampling);
    else
        bqfile->readParameters(size, z_qso, R_fwhm, snr, dv_kms);
}

void QSOFile::readData()
{
    wave  = new double[size];
    delta = new double[size];
    noise = new double[size];

    if (pfile != NULL)
        pfile->readData(wave, delta, noise);
    else
        bqfile->readData(wave, delta, noise);
}

void QSOFile::readMinMaxMedRedshift(double &zmin, double &zmax, double &zmed)
{
    if (wave == NULL)
    {
        wave  = new double[size];
        delta = new double[size];
        if (pfile != NULL)
            pfile->readData(wave, delta, delta);
        else
            bqfile->readData(wave, delta, delta);
    }

    zmin = wave[0] / LYA_REST - 1;
    zmax = wave[size-1] / LYA_REST - 1;
    zmed = wave[size/2] / LYA_REST - 1;
}

void QSOFile::readAllocResolutionMatrix()
{
    if (pfile != NULL)
        Rmat = pfile->readAllocResolutionMatrix(oversampling, dlambda);
    else
        throw std::runtime_error("Cannot read resolution matrix from Binary file!");
}

// ============================================================================
// Binary QSO file
// ============================================================================
BQFile::BQFile(std::string fname_qso)
{
    qso_file = ioh::open_file(fname_qso.c_str(), "rb");
}

BQFile::~BQFile()
{
    fclose(qso_file);
}

void BQFile::readParameters(int &data_number, double &z, int &fwhm_resolution, 
    double &sig2noi, double &dv_kms)
{
    rewind(qso_file);

    if (fread(&header, sizeof(qso_io_header), 1, qso_file) != 1)
        throw std::runtime_error("fread error in header BQFile!");

    data_number      = header.data_size;
    z                = header.redshift;
    fwhm_resolution  = header.spectrograph_resolution_fwhm;
    sig2noi          = header.signal_to_noise;
    dv_kms           = header.pixel_width;
}

void BQFile::readData(double *lambda, double *fluxfluctuations, double *noise)
{
    int rl, rf, rn;
    fseek(qso_file, sizeof(qso_io_header), SEEK_SET);

    rl = fread(lambda,             sizeof(double), header.data_size, qso_file);
    rf = fread(fluxfluctuations,   sizeof(double), header.data_size, qso_file);
    rn = fread(noise,              sizeof(double), header.data_size, qso_file);

    if (rl != header.data_size || rf != header.data_size || rn != header.data_size)
        throw std::runtime_error("fread error in data BQFile!");
}

// ============================================================================
// Picca File
// ============================================================================

PiccaFile::PiccaFile(std::string fname_qso) : status(0)
{
    // Assume fname to be ..fits.gz[1]
    fits_open_file(&fits_file, fname_qso.c_str(), READONLY, &status);
    fits_get_hdu_num(fits_file, &curr_spec_index);
    fits_get_num_hdus(fits_file, &no_spectra, &status);

    no_spectra--;
    _checkStatus();
    // _move(fname_qso[fname_qso.size-2] - '0');
}

void PiccaFile::_checkStatus()
{
    char error_msg[50]="FITS ERROR ";
    fits_get_errstatus(status, &error_msg[12]);
    if (status)     throw std::runtime_error(std::string(error_msg));
}

void PiccaFile::readParameters(int &N, double &z, int &fwhm_resolution, 
    double &sig2noi, double &dv_kms, double &dlambda, int &oversampling)
{
    // This is not ndiags in integer, but length in bytes that includes other columns
    // fits_read_key(fits_file, TINT, "NAXIS1", &curr_ndiags, NULL, &status);
    fits_read_key(fits_file, TINT, "NAXIS2", &curr_N, NULL, &status);

    fits_read_key(fits_file, TDOUBLE, "Z", &z, NULL, &status);

    double r_kms;
    fits_read_key(fits_file, TDOUBLE, "MEANRESO", &r_kms, NULL, &status);
    fwhm_resolution = int(SPEED_OF_LIGHT/r_kms/ONE_SIGMA_2_FWHM/100 + 0.5)*100;

    fits_read_key(fits_file, TDOUBLE, "MEANSNR", &sig2noi, NULL, &status);

    fits_read_key(fits_file, TDOUBLE, "DLL", &dv_kms, NULL, &status);
    fits_read_key(fits_file, TDOUBLE, "DLAMBDA", &dlambda, NULL, &status);
    fits_read_key(fits_file, TINT, "OVERSAMP", &oversampling, NULL, &status);

    #define LN10 2.30258509299
    dv_kms = round(dv_kms*SPEED_OF_LIGHT*LN10/5)*5;
    #undef LN10

    N = curr_N;
    curr_elem_per_row = -1;
}

int PiccaFile::_getColNo(char *tmplt)
{
    int colnum;
    status = 0;
    fits_get_colnum(fits_file, CASEINSEN, tmplt, &colnum, &status);
    _checkStatus();
    return colnum;
}

void PiccaFile::readData(double *lambda, double *delta, double *noise)
{
    int nonull, colnum;
    // _move(newhdu);
    char logtmp[]="LOGLAM";
    colnum = _getColNo(logtmp);
    fits_read_col(fits_file, TDOUBLE, colnum, 1, 1, curr_N, 0, lambda, &nonull, 
        &status);

    char deltmp[]="DELTA";
    colnum = _getColNo(deltmp);
    fits_read_col(fits_file, TDOUBLE, colnum, 1, 1, curr_N, 0, delta, &nonull, 
        &status);

    char ivartmp[]="IVAR";
    colnum = _getColNo(ivartmp);
    fits_read_col(fits_file, TDOUBLE, colnum, 1, 1, curr_N, 0, noise, &nonull, 
        &status);

    std::for_each(lambda, lambda+curr_N, [](double &ld) { ld = pow(10, ld); });
    std::for_each(noise, noise+curr_N, [](double &ld) { ld = pow(ld+1e-16, -0.5); });
}

mxhelp::Resolution* PiccaFile::readAllocResolutionMatrix(int oversampling, double dlambda)
{
    int nonull, naxis, colnum;
    long naxes[2];
    mxhelp::Resolution* Rmat;
    char resotmp[]="RESOMAT";
    colnum = _getColNo(resotmp);
    fits_read_tdim(fits_file, colnum, curr_N, &naxis, &naxes[0], &status);
    
    curr_elem_per_row = naxes[0];
    if ((curr_elem_per_row < 1) || (curr_elem_per_row-1)%(2*oversampling) != 0)
        throw std::runtime_error("Resolution matrix is not properly formatted.");

    Rmat = new mxhelp::Resolution(curr_N, curr_elem_per_row, oversampling, dlambda);

    fits_read_col(fits_file, TDOUBLE, colnum, 1, 1, curr_N*curr_elem_per_row, 0, 
        Rmat->values, &nonull, &status);
    // Rmat->orderTranspose();
    return Rmat;
}

PiccaFile::~PiccaFile()
{
    fits_close_file(fits_file, &status);
}

void PiccaFile::_move(int index)
{
    int hdutype;
    if (index >= no_spectra)
        throw std::runtime_error("Trying go beyond # HDU in fits file.");

    if (index > 0 && curr_spec_index != index)
    {
        fits_movabs_hdu(fits_file, index, &hdutype, &status);
        curr_spec_index = index;
    }
}

}










