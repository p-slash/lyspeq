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

QSOFile::QSOFile(std::string fname_qso, ifileformat p_or_b)
    : PB(p_or_b), pfile(NULL), bqfile(NULL)
{
    if (PB == Picca)
        pfile = new PiccaFile(fname_qso);
    else
        bqfile = new BQFile(fname_qso);
}

QSOFile::~QSOFile()
{
    delete pfile;
    delete bqfile;
}

void QSOFile::readParameters(int &data_number, double &z, int &fwhm_resolution, 
    double &sig2noi, double &dv_kms)
{
    if (pfile != NULL)
        pfile->readParameters(data_number, z, fwhm_resolution, sig2noi, dv_kms);
    else
        bqfile->readParameters(data_number, z, fwhm_resolution, sig2noi, dv_kms);
}

void QSOFile::readData(double *lambda, double *fluxfluctuations, double *noise)
{
    if (pfile != NULL)
        pfile->readData(lambda, fluxfluctuations, noise);
    else
        bqfile->readData(lambda, fluxfluctuations, noise);
}

void QSOFile::readAllocResolutionMatrix(mxhelp::Resolution *& Rmat)
{
    if (pfile != NULL)
        pfile->readAllocResolutionMatrix(Rmat);
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
    printf("Openning file:%s\n", fname_qso.c_str());
    int hdunum = fname_qso[fname_qso.size()-2]-'0'+1;
    printf("%d\n", hdunum);
    fits_open_file(&fits_file, fname_qso.c_str(), READONLY, &status);
    fits_get_hdu_num(fits_file, &curr_spec_index);
    fits_get_num_hdus(fits_file, &no_spectra, &status);

    printf("%d/%d\n", curr_spec_index, no_spectra);
    if (hdunum != curr_spec_index)
        throw std::runtime_error("FITS ERROR: Read HDU is not expected HDU!\n");

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
    double &sig2noi, double &dv_kms)
{
    int id;
    fits_get_hdu_num(fits_file, &id);
    printf("%d vs %d\n", id, curr_spec_index);
    fits_movabs_hdu(fits_file, curr_spec_index, &id, &status);
    // _move(newhdu);
    char comment[100];
    // This is not ndiags in integer, but length in bytes that includes other columns
    // fits_read_key(fits_file, TINT, "NAXIS1", &curr_ndiags, NULL, &status);
    fits_read_key(fits_file, TINT, "NAXIS2", &curr_N, comment, &status);

    fits_read_key(fits_file, TDOUBLE, "Z", &z, comment, &status);

    double r_kms;
    fits_read_key(fits_file, TDOUBLE, "MEANRESO", &r_kms, comment, &status);
    fwhm_resolution = int(SPEED_OF_LIGHT/r_kms/ONE_SIGMA_2_FWHM/100 + 0.5)*100;

    fits_read_key(fits_file, TDOUBLE, "MEANSNR", &sig2noi, comment, &status);

    fits_read_key(fits_file, TDOUBLE, "DLL", &dv_kms, comment, &status);

    #define LN10 2.30258509299
    dv_kms = round(dv_kms*SPEED_OF_LIGHT/LN10/5)*5;
    #undef LN10

    N = curr_N;
    curr_ndiags = -1;
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
    fits_movabs_hdu(fits_file, curr_spec_index, &nonull, &status);
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
    std::for_each(noise, noise+curr_N, [](double &ld) { ld = pow(ld, -0.5); });
}

void PiccaFile::readAllocResolutionMatrix(mxhelp::Resolution *& Rmat)
{
    int nonull, naxis, colnum;
    long *naxes = new long[2];
    char resotmp[]="RESOMAT";
    colnum = _getColNo(resotmp);
    fits_read_tdim(fits_file, colnum, curr_N, &naxis, naxes, &status);
    curr_ndiags = naxes[0];
    Rmat = new mxhelp::Resolution(curr_N, curr_ndiags);
    
    fits_read_col(fits_file, TDOUBLE, colnum, 1, 1, curr_N*curr_ndiags, 0, 
        Rmat->matrix, &nonull, &status);
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










