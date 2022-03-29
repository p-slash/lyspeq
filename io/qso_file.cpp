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
QSOFile::QSOFile(const std::string &fname_qso, ifileformat p_or_b)
    : fname(fname_qso), PB(p_or_b), pfile(NULL), bqfile(NULL),
    wave_head(NULL), delta_head(NULL), noise_head(NULL), Rmat(NULL)
{
    if (PB == Picca)
        pfile = new PiccaFile(fname);
    else
        bqfile = new BQFile(fname);

    dlambda=-1;
    oversampling=-1;
    id = 0;
}

void QSOFile::closeFile()
{
    delete pfile;
    delete bqfile;
    pfile=NULL;
    bqfile=NULL;
}

QSOFile::~QSOFile()
{
    closeFile();
    delete Rmat;
    delete [] wave_head;
    delete [] delta_head;
    delete [] noise_head;
}

void QSOFile::readParameters()
{
    if (pfile != NULL)
        pfile->readParameters(id, size, z_qso, R_fwhm, snr, dv_kms, dlambda, oversampling);
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

    wave_head  = wave;
    delta_head = delta;
    noise_head = noise;

    // Update dv and dlambda
    if (dlambda < 0)
        dlambda = wave[1]-wave[0];
    if (dv_kms < 0)
        dv_kms = round(dlambda/wave[size/2]*SPEED_OF_LIGHT/5)*5;
}

int QSOFile::cutBoundary(double z_lower_edge, double z_upper_edge)
{
    double l1 = LYA_REST * (1+z_lower_edge), l2 = LYA_REST * (1+z_upper_edge);
    int wi1, wi2;

    wi1 = std::lower_bound(wave, wave+size, l1)-wave;
    wi2 = std::upper_bound(wave, wave+size, l2)-wave;
    int newsize = wi2-wi1;

    if ((wi1 == size) || (wi2 == 0)) // empty
        return 0;

    if ((wi1 == 0) && (wi2 == size)) // no change
        return size;

    wave  += wi1;
    delta += wi1;
    noise += wi1;
    size  = newsize;

    if (Rmat != NULL)
        Rmat->cutBoundary(wi1, wi2);

    return size;
}

void QSOFile::readMinMaxMedRedshift(double &zmin, double &zmax, double &zmed)
{
    if (wave_head == NULL)
    {
        wave  = new double[size];
        delta = new double[size];
        if (pfile != NULL)
            pfile->readData(wave, delta, delta);
        else
            bqfile->readData(wave, delta, delta);

        wave_head  = wave;
        delta_head = delta;
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
BQFile::BQFile(const std::string &fname_qso)
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

std::string decomposeFname(const std::string &fname, int &hdunum)
{
    std::size_t i1 = fname.rfind('[')+1, i2 = fname.rfind(']');
    std::string basefname = fname.substr(0, i1-1);

    hdunum = std::stoi(fname.substr(i1, i2-i1))+1;

    return basefname;
}

// Statics
std::map<std::string, fitsfile*> PiccaFile::cache;
void PiccaFile::clearCache()
{
    int status=0;
    std::map<std::string, fitsfile*>::iterator it;
    for (it = cache.begin(); it != cache.end(); ++it)
        fits_close_file(it->second, &status);

    cache.clear();
}

bool PiccaFile::compareFnames(const std::string &s1, const std::string &s2)
{
    int hdu1, hdu2;
    std::string b1 = decomposeFname(s1, hdu1), b2 = decomposeFname(s2, hdu2);
    int comp = b1.compare(b2);

    if (comp != 0)
        return comp > 0;
    return hdu1 < hdu2;
}

#define MAX_NO_FILES 1

// Normals
// Assume fname to be ..fits.gz[1]
PiccaFile::PiccaFile(const std::string &fname_qso) : status(0)
{
    int hdunum, hdutype;
    std::string basefname = decomposeFname(fname_qso, hdunum);

    std::map<std::string, fitsfile*>::iterator it = cache.find(basefname);
    if (it != cache.end())
    {
        fits_file = it->second;
        fits_movabs_hdu(fits_file, hdunum, &hdutype, &status);
    }
    else
    {
        if (cache.size() == MAX_NO_FILES)
        {
            fits_close_file(cache.begin()->second, &status);
            cache.erase(cache.begin());
        }
        fits_open_file(&fits_file, fname_qso.c_str(), READONLY, &status);
        cache[basefname] = fits_file;
    }

    _checkStatus();

    // fits_get_hdu_num(fits_file, &curr_spec_index);
    // fits_get_num_hdus(fits_file, &no_spectra, &status);

    // no_spectra--;
    // _move(fname_qso[fname_qso.size-2] - '0');
}

void PiccaFile::_checkStatus()
{
    char error_msg[50]="FITS ERROR ";
    fits_get_errstatus(status, &error_msg[11]);
    if (status)     throw std::runtime_error(std::string(error_msg));
}

void PiccaFile::readParameters(long &thid, int &N, double &z, int &fwhm_resolution, 
    double &sig2noi, double &dv_kms, double &dlambda, int &oversampling)
{
    curr_elem_per_row = -1;

    // This is not ndiags in integer, but length in bytes that includes other columns
    // fits_read_key(fits_file, TINT, "NAXIS1", &curr_ndiags, NULL, &status);
    fits_read_key(fits_file, TINT, "NAXIS2", &curr_N, NULL, &status);
    N = curr_N;

    fits_read_key(fits_file, TLONG, "TARGETID", &thid, NULL, &status);
    if (status)
    {
        fits_clear_errmsg();
        status = 0;
        fits_read_key(fits_file, TLONG, "THING_ID", &thid, NULL, &status);
    }

    fits_read_key(fits_file, TDOUBLE, "Z", &z, NULL, &status);

    double r_kms;
    fits_read_key(fits_file, TDOUBLE, "MEANRESO", &r_kms, NULL, &status);
    fwhm_resolution = int(SPEED_OF_LIGHT/r_kms/ONE_SIGMA_2_FWHM/100 + 0.5)*100;

    fits_read_key(fits_file, TDOUBLE, "MEANSNR", &sig2noi, NULL, &status);

    // Soft read DLL, if not present set to -1 to be fixed later while reading data
    fits_read_key(fits_file, TDOUBLE, "DLL", &dv_kms, NULL, &status);
    if (status)
    {
        fits_clear_errmsg();
        status = 0;
        dv_kms = -1;
    }
    else
    {
        #define LN10 2.30258509299
        dv_kms = round(dv_kms*SPEED_OF_LIGHT*LN10/5)*5;
        #undef LN10
    }

    // Soft read DLAMBDA, if not present set to -1 to be fixed later while reading data
    fits_read_key(fits_file, TDOUBLE, "DLAMBDA", &dlambda, NULL, &status);
    if (status)
    {
        fits_clear_errmsg();
        status = 0;
        dlambda = -1;
    }

    try
    {
        fits_read_key(fits_file, TINT, "OVERSAMP", &oversampling, NULL, &status);
        _checkStatus();
    }
    catch (std::exception& e)
    {
        oversampling=-1;
        status=0;
    }
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

    char logtmp[]="LOGLAM";
    // Soft call if LOGLAM column exists
    fits_get_colnum(fits_file, CASEINSEN, logtmp, &colnum, &status);
    // If not, look for LAMBDA column
    if (status)
    {
        fits_clear_errmsg();
        status = 0;
        char lambtmp[]="LAMBDA";
        colnum = _getColNo(lambtmp);
        fits_read_col(fits_file, TDOUBLE, colnum, 1, 1, curr_N, 0, lambda, &nonull, 
            &status);
    }
    else
    {
        fits_read_col(fits_file, TDOUBLE, colnum, 1, 1, curr_N, 0, lambda, &nonull, 
            &status);
        std::for_each(lambda, lambda+curr_N, [](double &ld) { ld = pow(10, ld); });
    }

    char deltmp[]="DELTA";
    colnum = _getColNo(deltmp);
    fits_read_col(fits_file, TDOUBLE, colnum, 1, 1, curr_N, 0, delta, &nonull, 
        &status);

    char ivartmp[]="IVAR";
    colnum = _getColNo(ivartmp);
    fits_read_col(fits_file, TDOUBLE, colnum, 1, 1, curr_N, 0, noise, &nonull, 
        &status);

    _checkStatus();

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
    _checkStatus();

    curr_elem_per_row = naxes[0];
    if (oversampling == -1) // matrix is in dia format
        Rmat = new mxhelp::Resolution(curr_N, curr_elem_per_row);
    else
        Rmat = new mxhelp::Resolution(curr_N, curr_elem_per_row, oversampling, dlambda);

    if ((curr_elem_per_row < 1) || (curr_elem_per_row-1)%(2*oversampling) != 0)
        throw std::runtime_error("Resolution matrix is not properly formatted.");

    fits_read_col(fits_file, TDOUBLE, colnum, 1, 1, curr_N*curr_elem_per_row, 0, 
        Rmat->values, &nonull, &status);
    _checkStatus();

    if (oversampling == -1)
        Rmat->orderTranspose();

    return Rmat;
}

PiccaFile::~PiccaFile()
{
    // fits_close_file(fits_file, &status);
}

// void PiccaFile::_move(int index)
// {
//     int hdutype;
//     if (index >= no_spectra)
//         throw std::runtime_error("Trying go beyond # HDU in fits file.");

//     if (index > 0 && curr_spec_index != index)
//     {
//         fits_movabs_hdu(fits_file, index, &hdutype, &status);
//         curr_spec_index = index;
//     }
// }

}










