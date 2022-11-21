#include "io/qso_file.hpp"
#include "io/io_helper_functions.hpp"

#include <cmath>
#include <algorithm>
#include <numeric> // std::adjacent_difference
#include <limits>
#include <stdexcept>

namespace qio
{
double _calcdv(double w2, double w1) { return log(w2/w1)*SPEED_OF_LIGHT; }
double _getMediandv(const double *wave, int size)
{
    static std::vector<double> temp_arr;
    temp_arr.clear();
    temp_arr.reserve(size);

    std::adjacent_difference(wave, wave+size, std::back_inserter(temp_arr), _calcdv);
    std::sort(temp_arr.begin()+1, temp_arr.end());

    double median = temp_arr[1+(size-1)/2];
    median = round(median/5)*5;

    return  median;
}

double _getMediandlambda(const double *wave, int size)
{
    static std::vector<double> temp_arr;
    temp_arr.clear();
    temp_arr.reserve(size);

    std::adjacent_difference(wave, wave+size,  std::back_inserter(temp_arr));
    std::sort(temp_arr.begin()+1, temp_arr.end());

    double median = temp_arr[1+(size-1)/2];

    return median;
}

// ============================================================================
// Umbrella QSO file
// ============================================================================

// double size, z_qso, snr, dv_kms, dlambda;
// int R_fwhm;
// double *wave, *delta, *noise;
// mxhelp::Resolution *Rmat;
QSOFile::QSOFile(const std::string &fname_qso, ifileformat p_or_b) :
PB(p_or_b), wave_head(NULL), delta_head(NULL), noise_head(NULL),
arr_size(0), shift(0), num_masked_pixels(0), fname(fname_qso)
{
    if (PB == Picca)
        pfile = std::make_unique<PiccaFile>(fname);
    else
        bqfile = std::make_unique<BQFile>(fname);

    dlambda=-1;
    oversampling=-1;
    id = 0;
}

QSOFile::QSOFile(const qio::QSOFile &qmaster, int i1, int i2)
: PB(qmaster.PB), shift(0), num_masked_pixels(0), fname(qmaster.fname), 
z_qso(qmaster.z_qso), snr(qmaster.snr), id(qmaster.id),
R_fwhm(qmaster.R_fwhm), oversampling(qmaster.oversampling)
// dv_kms(qmaster.dv_kms), dlambda(qmaster.dlambda),
{
    arr_size = i2-i1;

    wave_head  = new double[arr_size];
    delta_head = new double[arr_size];
    noise_head = new double[arr_size];

    std::copy(qmaster.wave()+i1, qmaster.wave()+i2, wave());
    std::copy(qmaster.delta()+i1, qmaster.delta()+i2, delta());
    std::copy(qmaster.noise()+i1, qmaster.noise()+i2, noise());

    _cutMaskedBoundary();

    if (qmaster.Rmat)
        Rmat = std::make_unique<mxhelp::Resolution>(
            qmaster.Rmat.get(), i1+shift, i1+shift+arr_size);
    recalcDvDLam();
}

void QSOFile::closeFile()
{
    pfile.reset();
    bqfile.reset();
}

QSOFile::~QSOFile()
{
    closeFile();
    delete [] wave_head;
    delete [] delta_head;
    delete [] noise_head;
}

void QSOFile::readParameters()
{
    if (pfile)
        pfile->readParameters(id, arr_size, z_qso, R_fwhm, snr, dv_kms, dlambda, oversampling);
    else
        bqfile->readParameters(arr_size, z_qso, R_fwhm, snr, dv_kms);
}

void QSOFile::readData()
{
    if (arr_size <= 0)
        throw std::runtime_error("File is non-positive!");

    wave_head  = new double[arr_size];
    delta_head = new double[arr_size];
    noise_head = new double[arr_size];

    if (pfile)
        pfile->readData(wave(), delta(), noise());
    else
        bqfile->readData(wave(), delta(), noise());

    // _zeroMaskedFlux();

    // Update dv and dlambda
    if (dlambda < 0)
        dlambda = _getMediandlambda(wave(), size());
    if (dv_kms < 0)
        dv_kms  = _getMediandv(wave(), size());
}

void QSOFile::_countMaskedPixels(double sigma_cut)
{
    num_masked_pixels = 0;
    for (int i = 0; i < size(); ++i)
        if (noise()[i] > sigma_cut)
            ++num_masked_pixels;
}

void QSOFile::_cutMaskedBoundary(double sigma_cut)
{
    int ni1 = 0, ni2 = size();

    while ((ni1 < size()) && (noise()[ni1] > sigma_cut))
        ++ni1;
    while ((ni2 > 0) && (noise()[ni2-1] > sigma_cut))
        --ni2;

    if ((ni1 == 0) && (ni2 == arr_size)) // no change
        return;

    shift += ni1;
    arr_size = ni2-ni1;

    if (arr_size < 0)
        throw std::runtime_error(
            "Empty spectrum when masked pixels at boundary are removed!"
        );

    _countMaskedPixels(sigma_cut);
}

void QSOFile::cutBoundary(double z_lower_edge, double z_upper_edge)
{
    double l1 = LYA_REST * (1+z_lower_edge), l2 = LYA_REST * (1+z_upper_edge);
    int wi1, wi2;

    wi1 = std::lower_bound(wave(), wave()+size(), l1)-wave();
    wi2 = std::upper_bound(wave(), wave()+size(), l2)-wave();
    int newsize = wi2-wi1;

    if ((wi1 == arr_size) || (wi2 == 0)) // empty
        return;

    if ((wi1 == 0) && (wi2 == arr_size)) // no change
        return;

    shift += wi1;
    arr_size = newsize;

    _cutMaskedBoundary();

    if (Rmat)
        Rmat->cutBoundary(shift, shift+arr_size);
}

void QSOFile::readMinMaxMedRedshift(double &zmin, double &zmax, double &zmed)
{
    if (wave_head == NULL)
    {
        wave_head  = new double[size()];
        delta_head = new double[size()];
        noise_head = new double[size()];
        if (pfile)
            pfile->readData(wave(), delta(), noise());
        else
            bqfile->readData(wave(), delta(), noise());
        _cutMaskedBoundary();
    }

    zmin = wave()[0] / LYA_REST - 1;
    zmax = wave()[size()-1] / LYA_REST - 1;
    zmed = wave()[size()/2] / LYA_REST - 1;
}

void QSOFile::readAllocResolutionMatrix()
{
    if (pfile)
        Rmat = pfile->readAllocResolutionMatrix(oversampling, dlambda);
    else
        throw std::runtime_error("Cannot read resolution matrix from Binary file!");
}

void QSOFile::recalcDvDLam()
{
    dlambda = _getMediandlambda(wave(), size());
    dv_kms  = _getMediandv(wave(), size());
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
    for (auto &it : cache)
        fits_close_file(it.second, &status);

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

const int MAX_NO_FILES = 1;

// Normals
// Assume fname to be ..fits.gz[1]
PiccaFile::PiccaFile(const std::string &fname_qso) : status(0)
{
    int hdunum, hdutype;
    std::string basefname = decomposeFname(fname_qso, hdunum);

    auto it = cache.find(basefname);
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

    _setHeaderKeys();
    _setColumnNames();
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

void PiccaFile::_setHeaderKeys()
{
    int nkeys;
    char keyname[FLEN_KEYWORD], value[FLEN_VALUE];

    fits_get_hdrspace(fits_file, &nkeys, NULL, &status);
    header_keys.reserve(nkeys);

    for (int i = 1; i <= nkeys; ++i)
    {
        fits_read_keyn(fits_file, i, keyname, value, NULL, &status);
        header_keys.push_back(std::string(keyname));
    }
}

void PiccaFile::_setColumnNames()
{
    int ncols;
    char keyname[FLEN_KEYWORD], colname[FLEN_VALUE];
    fits_get_num_cols(fits_file, &ncols, &status);
    colnames.reserve(ncols);

    for (int i = 1; i <= ncols; i++)
    {
        fits_make_keyn("TTYPE", i, keyname, &status); /* make keyword */
        fits_read_key(fits_file, TSTRING, keyname, colname, NULL, &status);

        colnames.push_back(std::string(colname));
    }
}

bool PiccaFile::_isHeaderKey(const std::string &key)
{
    return std::any_of(header_keys.begin(), header_keys.end(), 
        [key](const std::string &elem) { return elem == key; });
}

bool PiccaFile::_isColumnName(const std::string &key)
{
    return std::any_of(colnames.begin(), colnames.end(), 
        [key](const std::string &elem) { return elem == key; });
}

void PiccaFile::readParameters(long &thid, int &N, double &z, int &fwhm_resolution, 
    double &sig2noi, double &dv_kms, double &dlambda, int &oversampling)
{
    curr_elem_per_row = -1;

    // This is not ndiags in integer, but length in bytes that includes other columns
    // fits_read_key(fits_file, TINT, "NAXIS1", &curr_ndiags, NULL, &status);
    fits_read_key(fits_file, TINT, "NAXIS2", &curr_N, NULL, &status);
    N = curr_N;

    if (_isHeaderKey("TARGETID"))
        fits_read_key(fits_file, TLONG, "TARGETID", &thid, NULL, &status);
    else if (_isHeaderKey("THING_ID"))
        fits_read_key(fits_file, TLONG, "THING_ID", &thid, NULL, &status);
    else
        throw std::runtime_error("Header must have TARGETID  or THING_ID!\n");


    fits_read_key(fits_file, TDOUBLE, "Z", &z, NULL, &status);

    double r_kms;
    fits_read_key(fits_file, TDOUBLE, "MEANRESO", &r_kms, NULL, &status);
    fwhm_resolution = int(SPEED_OF_LIGHT/r_kms/ONE_SIGMA_2_FWHM/100 + 0.5)*100;

    fits_read_key(fits_file, TDOUBLE, "MEANSNR", &sig2noi, NULL, &status);

    const double LN10 = 2.30258509299;
    // Soft read DLL, if not present set to -1 to be fixed later while reading data
    if (_isHeaderKey("DLL"))
    {
        fits_read_key(fits_file, TDOUBLE, "DLL", &dv_kms, NULL, &status);
        dv_kms = round(dv_kms*SPEED_OF_LIGHT*LN10/5)*5;
    }
    else
        dv_kms = -1;

    // Soft read DLAMBDA, if not present set to -1 to be fixed later while reading data
    if (_isHeaderKey("DLAMBDA"))
        fits_read_key(fits_file, TDOUBLE, "DLAMBDA", &dlambda, NULL, &status);
    else
        dlambda = -1;

    if (_isHeaderKey("OVERSAMP"))
        fits_read_key(fits_file, TINT, "OVERSAMP", &oversampling, NULL, &status);
    else
        oversampling = -1;
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

    // Read wavelength
    char logtmp[]="LOGLAM", lambtmp[]="LAMBDA";
    // Soft call if LOGLAM column exists
    if (_isColumnName(logtmp))
    {
        colnum = _getColNo(logtmp);
        fits_read_col(fits_file, TDOUBLE, colnum, 1, 1, curr_N, 0, lambda, &nonull, 
            &status);
        std::for_each(lambda, lambda+curr_N, [](double &ld) { ld = pow(10, ld); });
    }
    // If not, look for LAMBDA column
    else if (_isColumnName(lambtmp))
    {
        colnum = _getColNo(lambtmp);
        fits_read_col(fits_file, TDOUBLE, colnum, 1, 1, curr_N, 0, lambda, &nonull, 
            &status);
    }
    else
        throw std::runtime_error("Wavelength must be in LOGLAM or LAMBDA!\n");

    // Read deltas
    char deltmp[]="DELTA", deltmpblind[]="DELTA_BLIND";
    if (_isColumnName(deltmp))
        colnum = _getColNo(deltmp);
    else if (_isColumnName(deltmpblind))
        colnum = _getColNo(deltmpblind);
    else
        throw std::runtime_error("Deltas must be in DELTA or DELTA_BLIND!\n");

    fits_read_col(fits_file, TDOUBLE, colnum, 1, 1, curr_N, 0, delta, &nonull, 
        &status);

    // Read inverse variance
    char ivartmp[]="IVAR";
    colnum = _getColNo(ivartmp);
    fits_read_col(fits_file, TDOUBLE, colnum, 1, 1, curr_N, 0, noise, &nonull, 
        &status);

    _checkStatus();

    std::for_each(noise, noise+curr_N, [](double &ld)
        { ld = 1./sqrt(ld+std::numeric_limits<double>::epsilon()); }
    );
}

std::unique_ptr<mxhelp::Resolution> PiccaFile::readAllocResolutionMatrix(int oversampling, double dlambda)
{
    std::unique_ptr<mxhelp::Resolution> Rmat;
    int nonull, naxis, colnum;
    long naxes[2];
    char resotmp[]="RESOMAT";
    colnum = _getColNo(resotmp);
    fits_read_tdim(fits_file, colnum, curr_N, &naxis, &naxes[0], &status);
    _checkStatus();

    curr_elem_per_row = naxes[0];
    if (oversampling == -1) // matrix is in dia format
        Rmat = std::make_unique<mxhelp::Resolution>(curr_N, curr_elem_per_row);
    else
        Rmat = std::make_unique<mxhelp::Resolution>(curr_N, curr_elem_per_row, oversampling, dlambda);

    if ((curr_elem_per_row < 1) || (curr_elem_per_row-1)%(2*oversampling) != 0)
        throw std::runtime_error("Resolution matrix is not properly formatted.");

    fits_read_col(fits_file, TDOUBLE, colnum, 1, 1, curr_N*curr_elem_per_row, 0, 
        Rmat->matrix(), &nonull, &status);
    _checkStatus();

    if (oversampling == -1)
        Rmat->orderTranspose();

    return Rmat;
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










