#include "io/sq_lookup_table_file.hpp"
#include "io/io_helper_functions.hpp"

#include <stdexcept>
#include <sstream>
#include <iomanip>

double _nullval = 0;

namespace sqhelper
{
std::string SQTableFileNameConvention(
        const std::string &OUTPUT_DIR, const std::string &OUTPUT_FILEBASE_S,
        int r, double dv
) {
    std::ostringstream st_fname;
    st_fname << OUTPUT_DIR << "/" << OUTPUT_FILEBASE_S << "_R" << r 
        << std::fixed << std::setprecision(1) << "_dv" << dv << ".fits";

    return st_fname.str();
}


SQLookupTableFile::SQLookupTableFile(const std::string &fname, bool towrite)
    : file_name(fname)
{
    if (towrite)
        fitsfile_ptr = ioh::create_unique_fitsfile_ptr(file_name);
    else
        fitsfile_ptr = ioh::open_unique_fitsfile_ptr(file_name, READONLY);

    fits_file = fitsfile_ptr.get();
}


SQ_IO_Header SQLookupTableFile::readMeta() {
    char extname[] = "METADATA";
    int status = 0;
    SQ_IO_Header hdr;

    fits_movnam_hdu(fits_file, IMAGE_HDU, extname, 0, &status);
    ioh::checkFitsStatus(status);

    fits_read_key(fits_file, TINT, "NVPTS", &hdr.nvpoints, nullptr, &status);
    fits_read_key(fits_file, TINT, "NZPTS", &hdr.nzpoints, nullptr, &status);

    fits_read_key(fits_file, TDOUBLE, "LENV", &hdr.v_length, nullptr, &status);
    fits_read_key(fits_file, TDOUBLE, "Z1", &hdr.z1, nullptr, &status);
    fits_read_key(fits_file, TDOUBLE, "LENZ", &hdr.z_length, nullptr, &status);

    fits_read_key(fits_file, TINT, "RFWHM", &hdr.spectrograph_resolution, nullptr, &status);
    fits_read_key(fits_file, TDOUBLE, "DV", &hdr.pixel_width, nullptr, &status);

    fits_read_key(fits_file, TDOUBLE, "K1", &hdr.k1, nullptr, &status);
    fits_read_key(fits_file, TDOUBLE, "DKLIN", &hdr.dklin, nullptr, &status);
    fits_read_key(fits_file, TDOUBLE, "DKLOG", &hdr.dklog, nullptr, &status);

    fits_read_key(fits_file, TINT, "NKLIN", &hdr.nklin, nullptr, &status);
    fits_read_key(fits_file, TINT, "NKLOG", &hdr.nklog, nullptr, &status);

    ioh::checkFitsStatus(status);
    meta = hdr;

    return hdr;
}


void SQLookupTableFile::writeMeta(SQ_IO_Header &hdr) {
    char extname[] = "METADATA";
    int status = 0;
    long naxis = 1, size = 0, naxes[1] = { 0 };

    fits_create_img(fits_file, DOUBLE_IMG, naxis, naxes, &status);
    ioh::checkFitsStatus(status);

    fits_update_key_str(fits_file, "EXTNAME", extname, nullptr, &status);
    fits_write_key(fits_file, TINT, "NVPTS", &hdr.nvpoints, nullptr, &status);
    fits_write_key(fits_file, TINT, "NZPTS", &hdr.nzpoints, nullptr, &status);

    fits_write_key(fits_file, TDOUBLE, "LENV", &hdr.v_length, nullptr, &status);
    fits_write_key(fits_file, TDOUBLE, "Z1", &hdr.z1, nullptr, &status);
    fits_write_key(fits_file, TDOUBLE, "LENZ", &hdr.z_length, nullptr, &status);

    fits_write_key(fits_file, TINT, "RFWHM", &hdr.spectrograph_resolution, nullptr, &status);
    fits_write_key(fits_file, TDOUBLE, "DV", &hdr.pixel_width, nullptr, &status);

    fits_write_key(fits_file, TDOUBLE, "K1", &hdr.k1, nullptr, &status);
    fits_write_key(fits_file, TDOUBLE, "DKLIN", &hdr.dklin, nullptr, &status);
    fits_write_key(fits_file, TDOUBLE, "DKLOG", &hdr.dklog, nullptr, &status);

    fits_write_key(fits_file, TINT, "NKLIN", &hdr.nklin, nullptr, &status);
    fits_write_key(fits_file, TINT, "NKLOG", &hdr.nklog, nullptr, &status);

    ioh::checkFitsStatus(status);
    meta = hdr;
}


void SQLookupTableFile::readDeriv(double *data) {
    char extname[] = "DERIVATIVE";
    int status = 0, nfound;
    long naxes[2];

    fits_movnam_hdu(fits_file, IMAGE_HDU, extname, 0, &status);
    ioh::checkFitsStatus(status);

    fits_read_keys_lng(fits_file, "NAXIS", 1, 2, naxes, &nfound, &status);

    if (naxes[0] != (meta.nklin + meta.nklog))
        throw std::runtime_error("SQLookupTableFile::readDeriv::Inconsistent Nk.");

    if (naxes[1] != meta.nvpoints)
        throw std::runtime_error("SQLookupTableFile::readDeriv::Inconsistent Nv.");

    long size = naxes[0] * naxes[1];
    fits_read_img(fits_file, TDOUBLE, 1, size, &_nullval, data, nullptr, &status);
    ioh::checkFitsStatus(status);
}


void SQLookupTableFile::writeDeriv(const double *data) {
    char extname[] = "DERIVATIVE";
    int status = 0;
    long naxis = 2, naxes[2] = { meta.nklin + meta.nklog, meta.nvpoints };
    long size = naxes[0] * naxes[1];

    fits_create_img(fits_file, DOUBLE_IMG, naxis, naxes, &status);
    ioh::checkFitsStatus(status);

    fits_update_key_str(fits_file, "EXTNAME", extname, nullptr, &status);
    fits_write_img(fits_file, TDOUBLE, 1, size, (void *) data, &status);
    ioh::checkFitsStatus(status);
}


void SQLookupTableFile::readSignal(double *data) {
    char extname[] = "SIGNAL";
    int status = 0, nfound;
    long naxes[2];

    fits_movnam_hdu(fits_file, IMAGE_HDU, extname, 0, &status);
    ioh::checkFitsStatus(status);

    fits_read_keys_lng(fits_file, "NAXIS", 1, 2, naxes, &nfound, &status);

    if (naxes[0] != meta.nzpoints)
        throw std::runtime_error("SQLookupTableFile::readSignal::Inconsistent Nz.");

    if (naxes[1] != meta.nvpoints)
        throw std::runtime_error("SQLookupTableFile::readSignal::Inconsistent Nv.");

    long size = naxes[0] * naxes[1];
    fits_read_img(fits_file, TDOUBLE, 1, size, &_nullval, data, nullptr, &status);
    ioh::checkFitsStatus(status);
}


void SQLookupTableFile::writeSignal(const double *data) {
    char extname[] = "SIGNAL";
    int status = 0;
    long naxis = 2, naxes[2] = { meta.nzpoints, meta.nvpoints };
    long size = naxes[0] * naxes[1];

    fits_create_img(fits_file, DOUBLE_IMG, naxis, naxes, &status);
    ioh::checkFitsStatus(status);

    fits_update_key_str(fits_file, "EXTNAME", extname, nullptr, &status);
    fits_write_img(fits_file, TDOUBLE, 1, size, (void *) data, &status);
    ioh::checkFitsStatus(status);
}
}



