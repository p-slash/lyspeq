#include "io/qso_file.hpp"
#include "mathtools/stats.hpp"

#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric> // std::adjacent_difference
#include <limits>
#include <stdexcept>

namespace qio
{
double _getMediandv(const double *wave, int size) {
    std::vector<double> temp_arr(size - 1);
    for (int i = 0; i < size - 1; ++i)
        temp_arr[i] = log(wave[i + 1] / wave[i]) * SPEED_OF_LIGHT;

    return stats::medianOfUnsortedVector(temp_arr);
}

double _getMediandlambda(const double *wave, int size) {
    std::vector<double> temp_arr(size - 1);
    for (int i = 0; i < size - 1; ++i)
        temp_arr[i] = wave[i + 1] - wave[i];

    return stats::medianOfUnsortedVector(temp_arr);
}

// ============================================================================
// Umbrella QSO file
// ============================================================================

// double size, z_qso, snr, dv_kms, dlambda;
// int R_fwhm;
// double *wave, *delta, *ivar;
// mxhelp::Resolution *Rmat;
QSOFile::QSOFile(const std::string &fname_qso, ifileformat p_or_b)
        : PB(p_or_b), wave_head(nullptr), delta_head(nullptr), ivar_head(nullptr),
          arr_size(0), _fullsize(0), shift(0), num_masked_pixels(0), fname(fname_qso),
          expid(-1), night(-1), fiber(-1), petal(-1)
{
    if (PB == Picca)
        pfile = std::make_unique<PiccaFile>(fname);
    else
        bqfile = std::make_unique<BQFile>(fname);

    dlambda=-1;
    id = 0;
}

QSOFile::QSOFile(const PiccaFile* pf, int hdunum)
        : PB(Picca), wave_head(nullptr), delta_head(nullptr), ivar_head(nullptr),
          arr_size(0), _fullsize(0), shift(0), num_masked_pixels(0), fname(""),
          expid(-1), night(-1), fiber(-1), petal(-1)
{
    pfile = std::make_unique<PiccaFile>(pf, hdunum);
    dlambda = -1;
    id = 0;
}

QSOFile::QSOFile(const qio::QSOFile &qmaster, int i1, int i2)
        : PB(qmaster.PB), shift(0), num_masked_pixels(0), fname(qmaster.fname), 
          z_qso(qmaster.z_qso), snr(qmaster.snr), R_kms(qmaster.R_kms),
          ra(qmaster.ra), dec(qmaster.dec), id(qmaster.id),
          R_fwhm(qmaster.R_fwhm), expid(qmaster.expid),
          night(qmaster.night), fiber(qmaster.fiber), petal(qmaster.petal)
{
    arr_size = i2 - i1;
    _fullsize = arr_size;

    wave_head  = new double[arr_size];
    delta_head = new double[arr_size];
    ivar_head = new double[arr_size];
    process::updateMemory(-process::getMemoryMB(_fullsize * 3));

    std::copy(qmaster.wave()+i1, qmaster.wave()+i2, wave_head);
    std::copy(qmaster.delta()+i1, qmaster.delta()+i2, delta_head);
    std::copy(qmaster.ivar()+i1, qmaster.ivar()+i2, ivar_head);

    _cutMaskedBoundary();

    if (qmaster.Rmat) {
        Rmat = std::make_unique<mxhelp::Resolution>(
            qmaster.Rmat.get(), i1+shift, i1+shift+arr_size);
        process::updateMemory(-Rmat->getMinMemUsage());
    }
    recalcDvDLam();
}

void QSOFile::readParameters()
{
    if (pfile) {
        pfile->readParameters(
            id, arr_size, z_qso, dec, ra, R_kms, snr, dv_kms, dlambda,
            expid, night, fiber, petal);
        R_fwhm = int(SPEED_OF_LIGHT / R_kms / ONE_SIGMA_2_FWHM / 100 + 0.5) * 100;
    }
    else {
        bqfile->readParameters(
            arr_size, z_qso, dec, ra, R_fwhm, snr, dv_kms);
        R_kms = SPEED_OF_LIGHT / R_fwhm / ONE_SIGMA_2_FWHM;
    }
    _fullsize = arr_size;
}

void QSOFile::readData()
{
    if (arr_size <= 0)
        throw std::runtime_error("File is non-positive!");

    wave_head  = new double[arr_size];
    delta_head = new double[arr_size];
    ivar_head = new double[arr_size];
    process::updateMemory(-process::getMemoryMB(_fullsize * 3));

    if (pfile)
        pfile->readData(wave(), delta(), ivar());
    else
        bqfile->readData(wave(), delta(), ivar());

    // _zeroMaskedFlux();

    // Update dv and dlambda
    if (dlambda < 0)
        dlambda = _getMediandlambda(wave(), size());
    if (dv_kms < 0)
        dv_kms  = _getMediandv(wave(), size());
}

void QSOFile::_countMaskedPixels()
{
    num_masked_pixels = 0;
    for (int i = 0; i < size(); ++i)
        if (ivar()[i] == 0)
            ++num_masked_pixels;
}

void QSOFile::_cutMaskedBoundary()
{
    int ni1 = 0, ni2 = size();

    while ((ni1 < size()) && (ivar()[ni1] == 0))
        ++ni1;
    while ((ni2 > 0) && (ivar()[ni2-1] == 0))
        --ni2;

    shift += ni1;
    arr_size = ni2 - ni1;

    if (arr_size <= 0)
        throw std::runtime_error(
            "Empty spectrum when masked pixels at boundary are removed!"
        );

    _countMaskedPixels();
}

void QSOFile::cutBoundary(double z_lower_edge, double z_upper_edge)
{
    double l1 = LYA_REST * (1+z_lower_edge), l2 = LYA_REST * (1+z_upper_edge);
    int wi1, wi2, org_arrsize = arr_size, org_shift = shift;

    assert (org_shift == 0);

    wi1 = std::lower_bound(wave(), wave() + arr_size, l1) - wave();
    wi2 = std::upper_bound(wave(), wave() + arr_size, l2) - wave();
    bool isempty = (wi1 == arr_size) || (wi2 == 0);

    arr_size = wi2 - wi1;

    if (isempty)
        throw std::runtime_error(
            "Empty spectrum when redshift boundary is applied!"
        );

    shift += wi1;

    _cutMaskedBoundary();

    bool nochange = (shift == org_shift) && (arr_size == org_arrsize);
    if (!nochange && Rmat) {
        process::updateMemory(Rmat->getMinMemUsage());
        Rmat->cutBoundary(shift, shift+arr_size);
        process::updateMemory(-Rmat->getMinMemUsage());
    }
}

int QSOFile::maskOutliers(double factor) {
    // more conservative (higher) upper bound compared to Turner24
    static auto mockMeanFluxRecip = [](double l) {
        return exp(1.663e-03 * pow(l / LYA_REST, 4.107));
    };

    int j = 0, nall = size();
    std::vector<double> temp_arr(realSize());

    double *iv = ivar(), *f = delta();

    for (int i = 0; i < nall; ++i) {
        if (iv[i] == 0)
            continue;

        temp_arr[j] = f[i];
        ++j;
    }

    double median = stats::medianOfUnsortedVector(temp_arr);
    std::for_each(
        temp_arr.begin(), temp_arr.end(),
        [median](double &x) { x = fabs(x - median); });

    double mad = 1.4826 * (1. + 1. / sqrt(realSize()))
                 * stats::medianOfUnsortedVector(temp_arr);
    mad = std::max(1.0, mad);

    j = 0;
    for (int i = 0; i < nall; ++i) {
        if (iv[i] == 0)
            continue;

        double s = factor * std::max(1.0 / sqrt(iv[i]), mad);
        double fup = mockMeanFluxRecip(wave()[i]);
        if ( (f[i] < (-1.0 - s)) || (f[i] > (fup + s)) ) {
            f[i] = 0;  iv[i] = 0;  ++j;
        }
    }

    return j;
}


void QSOFile::downsample(int m) {
    int nfirst = arr_size / m, nrem = arr_size % m,
        nnew = nfirst + int(nrem != 0);

    double *wn = new double[nnew], *dn = new double[nnew], *in = new double[nnew];

    for (int i = 0; i < nfirst; ++i) {
        wn[i] = 0;  dn[i] = 0;  in[i] = 0;
        for (int j = 0; j < m; ++j) {
            double tivar = ivar()[j + m * i];
            wn[i] += wave()[j + m * i] * tivar;
            dn[i] += delta()[j + m * i] * tivar;
            in[i] += tivar;
        }

        if (in[i] == 0) {
            dn[i] = 0;
            for (int j = 0; j < m; ++j)
                wn[i] += wave()[j + m * i] / m;
        }
        else {
            wn[i] /= in[i];  dn[i] /= in[i];
        }
    }

    if (nrem != 0) {
        wn[nfirst] = 0;  dn[nfirst] = 0;  in[nfirst] = 0;
        for (int j = 0; j < nrem; ++j) {
            double tivar = ivar()[j + m * nfirst];
            wn[nfirst] += wave()[j + m * nfirst] * tivar;
            dn[nfirst] += delta()[j + m * nfirst] * tivar;
            in[nfirst] += tivar;
        }

        if (in[nfirst] == 0) {
            dn[nfirst] = 0;
            for (int j = 0; j < nrem; ++j)
                wn[nfirst] += wave()[j + m * nfirst] / nrem;
        }
        else {
            wn[nfirst] /= in[nfirst];  dn[nfirst] /= in[nfirst];
        }
    }

    process::updateMemory(getMinMemUsage());
    arr_size = nnew;
    _fullsize = arr_size;
    shift = 0;
    num_masked_pixels = 0;

    process::updateMemory(-process::getMemoryMB(_fullsize * 3));
    delete [] wave_head;
    delete [] delta_head;
    delete [] ivar_head;

    wave_head = wn;  delta_head = dn;  ivar_head = in;
    recalcDvDLam();
}


void QSOFile::readMinMaxMedRedshift(double &zmin, double &zmax, double &zmed)
{
    if (wave_head == nullptr)
    {
        wave_head  = new double[size()];
        delta_head = new double[size()];
        ivar_head = new double[size()];
        process::updateMemory(-process::getMemoryMB(size() * 3));

        if (pfile)
            pfile->readData(wave(), delta(), ivar());
        else
            bqfile->readData(wave(), delta(), ivar());

        _cutMaskedBoundary();
    }

    zmin = wave()[0] / LYA_REST - 1;
    zmax = wave()[size() - 1] / LYA_REST - 1;
    zmed = stats::medianOfSortedArray(wave(), size()) / LYA_REST - 1;
}

void QSOFile::readAllocResolutionMatrix()
{
    if (pfile) {
        Rmat = pfile->readAllocResolutionMatrix();
        process::updateMemory(-Rmat->getMinMemUsage());
    }
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
void BQFile::readParameters(
        int &data_number, double &z, double &dec, double &ra,
        int &fwhm_resolution, double &sig2noi, double &dv_kms
) {
    rewind(qso_file);

    if (fread(&header, sizeof(qso_io_header), 1, qso_file) != 1)
        throw std::runtime_error("fread error in header BQFile!");

    data_number = header.data_size;
    z = header.redshift;
    dec = header.declination;
    ra = header.right_ascension;
    fwhm_resolution = header.spectrograph_resolution_fwhm;
    sig2noi = header.signal_to_noise;
    dv_kms = header.pixel_width;
}

void BQFile::readData(double *lambda, double *fluxfluctuations, double *ivar)
{
    int rl, rf, rn;
    fseek(qso_file, sizeof(qso_io_header), SEEK_SET);

    rl = fread(lambda,             sizeof(double), header.data_size, qso_file);
    rf = fread(fluxfluctuations,   sizeof(double), header.data_size, qso_file);
    rn = fread(ivar,               sizeof(double), header.data_size, qso_file);

    if (rl != header.data_size || rf != header.data_size || rn != header.data_size)
        throw std::runtime_error("fread error in data BQFile!");

    std::for_each(ivar, ivar + header.data_size, [](double &iv) {
        iv = 1.0 / (iv * iv);
        if (iv < DOUBLE_EPSILON)
            iv = 0;
    });
}

// ============================================================================
// Picca File
// ============================================================================

std::string decomposeFname(const std::string &fname, int &hdunum)
{
    std::size_t i1 = fname.rfind('['), i2 = fname.rfind(']');

    if (i1 == std::string::npos) {
        hdunum = 2;
        return fname;
    }

    std::string basefname = fname.substr(0, i1);
    hdunum = std::stoi(fname.substr(i1 + 1, i2 - i1 - 1)) + 1;

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
bool PiccaFile::use_cache = true;

// Normals
// Assume fname to be ..fits.gz[1]
PiccaFile::PiccaFile(const std::string &fname_qso) : status(0)
{
    int hdunum, hdutype;
    std::string basefname = decomposeFname(fname_qso, hdunum);

    if (PiccaFile::use_cache) {
        auto it = cache.find(basefname);
        if (it != cache.end()) {
            fits_file = it->second;
        }
        else {
            if (cache.size() == MAX_NO_FILES) {
                fits_close_file(cache.begin()->second, &status);
                cache.erase(cache.begin());
            }

            fits_open_file(&fits_file, fname_qso.c_str(), READONLY, &status);
            cache[basefname] = fits_file;
            fits_get_num_hdus(fits_file, &no_spectra, &status);
            no_spectra--;
        }
    }
    else {
        fits_open_file(&fits_file, fname_qso.c_str(), READONLY, &status);
        fits_get_num_hdus(fits_file, &no_spectra, &status);
        no_spectra--;
    }

    fits_movabs_hdu(fits_file, hdunum, &hdutype, &status);

    if (hdutype != BINARY_TBL)
        throw std::runtime_error("HDU type is not BINARY!");

    _setHeaderKeys();
    _setColumnNames();

    if (status != 0)
        _handleStatus();

    // fits_get_hdu_num(fits_file, &curr_spec_index);
    // _move(fname_qso[fname_qso.size-2] - '0');
}


PiccaFile::PiccaFile(const PiccaFile *pf, int hdunum)
        : fits_file(pf->fits_file), status(0) 
{
    int hdutype;
    fits_movabs_hdu(fits_file, hdunum + 1, &hdutype, &status);

    if (hdutype != BINARY_TBL)
        throw std::runtime_error("HDU type is not BINARY!");

    _setHeaderKeys();
    _setColumnNames();

    if (status != 0)
        _handleStatus();
}


void PiccaFile::_handleStatus()
{
    char fits_msg[50];
    fits_get_errstatus(status, fits_msg);
    std::string error_msg = std::string("FITS ERROR ") + std::string(fits_msg);

    throw std::runtime_error(error_msg);
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

void PiccaFile::_readOptionalInt(const std::string &key, int &output) {
    if (_isHeaderKey(key))
        fits_read_key(fits_file, TINT, key.c_str(), &output, NULL, &status);
    else
        output = -1;
}

void PiccaFile::readParameters(
        long &thid, int &N, double &z, double &dec, double &ra,
        double &R_kms, double &sig2noi, double &dv_kms, double &dlambda,
        int &expid, int &night, int &fiber, int &petal
) {
    status = 0;
    curr_elem_per_row = -1;

    // This is not ndiags in integer, but length in bytes that includes other columns
    // fits_read_key(fits_file, TINT, "NAXIS1", &curr_ndiags, NULL, &status);
    // fits_get_num_rows(fits_file, &curr_N, &status);
    fits_read_key(fits_file, TINT, "NAXIS2", &curr_N, NULL, &status);
    N = curr_N;

    if (_isHeaderKey("TARGETID"))
        fits_read_key(fits_file, TLONG, "TARGETID", &thid, NULL, &status);
    else if (_isHeaderKey("THING_ID"))
        fits_read_key(fits_file, TLONG, "THING_ID", &thid, NULL, &status);
    else
        throw std::runtime_error("Header must have TARGETID  or THING_ID!");


    fits_read_key(fits_file, TDOUBLE, "Z", &z, NULL, &status);
    fits_read_key(fits_file, TDOUBLE, "RA", &ra, NULL, &status);
    fits_read_key(fits_file, TDOUBLE, "DEC", &dec, NULL, &status);

    fits_read_key(fits_file, TDOUBLE, "MEANRESO", &R_kms, NULL, &status);

    fits_read_key(fits_file, TDOUBLE, "MEANSNR", &sig2noi, NULL, &status);

    const double LN10 = 2.30258509299;
    // Soft read DLL, if not present set to -1 to be fixed later while reading data
    if (_isHeaderKey("DLL"))
    {
        fits_read_key(fits_file, TDOUBLE, "DLL", &dv_kms, NULL, &status);
        dv_kms = dv_kms*SPEED_OF_LIGHT*LN10;
    }
    else
        dv_kms = -1;

    // Soft read DLAMBDA, if not present set to -1 to be fixed later while reading data
    if (_isHeaderKey("DLAMBDA"))
        fits_read_key(fits_file, TDOUBLE, "DLAMBDA", &dlambda, NULL, &status);
    else
        dlambda = -1;

    _readOptionalInt("EXPID", expid);
    _readOptionalInt("NIGHT", night);
    _readOptionalInt("FIBER", fiber);
    _readOptionalInt("PETAL_LOC", petal);

    if (status != 0)
        _handleStatus();
}

int PiccaFile::_getColNo(char *tmplt)
{
    int colnum;
    status = 0;
    fits_get_colnum(fits_file, CASEINSEN, tmplt, &colnum, &status);
    
    if (status != 0)
        _handleStatus();

    return colnum;
}

void PiccaFile::readData(double *lambda, double *delta, double *ivar)
{
    int nonull, colnum;
    status = 0;

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
        throw std::runtime_error("Wavelength must be in LOGLAM or LAMBDA!");

    // Read deltas
    char deltmp[]="DELTA", deltmpblind[]="DELTA_BLIND";
    if (_isColumnName(deltmp))
        colnum = _getColNo(deltmp);
    else if (_isColumnName(deltmpblind))
        colnum = _getColNo(deltmpblind);
    else
        throw std::runtime_error("Deltas must be in DELTA or DELTA_BLIND!");

    fits_read_col(fits_file, TDOUBLE, colnum, 1, 1, curr_N, 0, delta, &nonull, 
        &status);

    // Read inverse variance
    char ivartmp[]="IVAR";
    colnum = _getColNo(ivartmp);
    fits_read_col(fits_file, TDOUBLE, colnum, 1, 1, curr_N, 0, ivar, &nonull, 
        &status);

    if (status != 0)
        _handleStatus();
}

std::unique_ptr<mxhelp::Resolution> PiccaFile::readAllocResolutionMatrix() {
    std::unique_ptr<mxhelp::Resolution> Rmat;
    int nonull, naxis, colnum;
    long naxes[2];
    char resotmp[] = "RESOMAT";
    colnum = _getColNo(resotmp);

    status = 0;
    fits_read_tdim(fits_file, colnum, curr_N, &naxis, &naxes[0], &status);
    if (status != 0)
        _handleStatus();

    curr_elem_per_row = naxes[0];
    Rmat = std::make_unique<mxhelp::Resolution>(curr_N, curr_elem_per_row);

    fits_read_col(
        fits_file, TDOUBLE, colnum, 1, 1, curr_N * curr_elem_per_row, 0, 
        Rmat->matrix(), &nonull, &status);

    if (status != 0)
        _handleStatus();

    Rmat->orderTranspose();

    return Rmat;
}
}










