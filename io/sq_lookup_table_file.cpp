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

    fits_file.set(fitsfile_ptr.get());
}


SQ_IO_Header SQLookupTableFile::readMeta() {
    char extname[] = "METADATA";
    SQ_IO_Header hdr;
    
    fits_file.moveToHdu(extname, IMAGE_HDU);
    fits_file.check();

    fits_file.readKey("NVPTS", hdr.nvpoints);
    fits_file.readKey("NZPTS", hdr.nzpoints);
    fits_file.readKey("LENV", hdr.v_length);
    fits_file.readKey("Z1", hdr.z1);
    fits_file.readKey("LENZ", hdr.z_length);
    fits_file.readKey("RFWHM", hdr.spectrograph_resolution);
    fits_file.readKey("DV", hdr.pixel_width);
    fits_file.readKey("K1", hdr.k1);
    fits_file.readKey("DKLIN", hdr.dklin);
    fits_file.readKey("DKLOG", hdr.dklog);
    fits_file.readKey("NKLIN", hdr.nklin);
    fits_file.readKey("NKLOG", hdr.nklog);

    fits_file.check();
    meta = hdr;

    return hdr;
}


void SQLookupTableFile::writeMeta(SQ_IO_Header &hdr) {
    char extname[] = "METADATA";
    long naxis = 1, size = 0, naxes[1] = { 0 };

    fits_file.createImage(naxis, naxes, extname);
    fits_file.check();

    fits_file.writeKey("NVPTS", hdr.nvpoints);
    fits_file.writeKey("NZPTS", hdr.nzpoints);
    fits_file.writeKey("LENV", hdr.v_length);
    fits_file.writeKey("Z1", hdr.z1);
    fits_file.writeKey("LENZ", hdr.z_length);
    fits_file.writeKey("RFWHM", hdr.spectrograph_resolution);
    fits_file.writeKey("DV", hdr.pixel_width);
    fits_file.writeKey("K1", hdr.k1);
    fits_file.writeKey("DKLIN", hdr.dklin);
    fits_file.writeKey("DKLOG", hdr.dklog);
    fits_file.writeKey("NKLIN", hdr.nklin);
    fits_file.writeKey("NKLOG", hdr.nklog);
    fits_file.check();
    meta = hdr;
}


void SQLookupTableFile::readDeriv(double *data) {
    char extname[] = "DERIVATIVE";
    int nfound;
    std::vector<long> naxes(2);

    fits_file.moveToHdu(extname, IMAGE_HDU);
    fits_file.check();
    fits_file.readLongKey(naxes, nfound);

    if (naxes[0] != (meta.nklin + meta.nklog))
        throw std::runtime_error("SQLookupTableFile::readDeriv::Inconsistent Nk.");

    if (naxes[1] != meta.nvpoints)
        throw std::runtime_error("SQLookupTableFile::readDeriv::Inconsistent Nv.");

    long size = naxes[0] * naxes[1];
    fits_file.readImage(data, size);
    fits_file.check();
}


void SQLookupTableFile::writeDeriv(const double *data) {
    char extname[] = "DERIVATIVE";
    long naxis = 2, naxes[2] = { meta.nklin + meta.nklog, meta.nvpoints };
    long size = naxes[0] * naxes[1];

    fits_file.createImage(naxis, naxes, extname);
    fits_file.check();

    fits_file.writeImage(size, data);
    fits_file.check();
}


void SQLookupTableFile::readSignal(double *data) {
    char extname[] = "SIGNAL";
    int nfound;
    std::vector<long> naxes(2);

    fits_file.moveToHdu(extname, IMAGE_HDU);
    fits_file.check();

    fits_file.readLongKey(naxes, nfound);

    if (naxes[0] != meta.nzpoints)
        throw std::runtime_error("SQLookupTableFile::readSignal::Inconsistent Nz.");

    if (naxes[1] != meta.nvpoints)
        throw std::runtime_error("SQLookupTableFile::readSignal::Inconsistent Nv.");

    long size = naxes[0] * naxes[1];
    fits_file.readImage(data, size);
    fits_file.check();
}


void SQLookupTableFile::writeSignal(const double *data) {
    char extname[] = "SIGNAL";
    long naxis = 2, naxes[2] = { meta.nzpoints, meta.nvpoints };
    long size = naxes[0] * naxes[1];

    fits_file.createImage(naxis, naxes, extname);
    fits_file.check();
    fits_file.writeImage(size, data);
    fits_file.check();
}

void SQLookupTableFile::readGuide(double *data) {
    char extname[] = "GUIDE";
    int nfound;
    std::vector<long> naxes(1);

    fits_file.moveToHdu(extname, IMAGE_HDU);
    fits_file.check();

    fits_file.readLongKey(naxes, nfound);

    if (naxes[0] != meta.nvpoints)
        throw std::runtime_error("SQLookupTableFile::readGuide::Inconsistent Nv.");

    long size = naxes[0];
    fits_file.readImage(data, size);
    fits_file.check();
}

void SQLookupTableFile::writeGuide(const double *data) {
    char extname[] = "GUIDE";
    long naxis = 1, naxes[1] = { meta.nvpoints };
    long size = naxes[0];

    fits_file.createImage(naxis, naxes, extname);
    fits_file.check();

    fits_file.writeImage(size, data);
    fits_file.check();
}

void SQLookupTableFile::writeKbinCorrection(
    const double *kcenters, const double *kkernel,
    const double *kguide, const double *kcorrection
) {
    constexpr const char *INV_VEL_UNIT = "s km^-1";
    const std::vector<ioh::FitsColumn> columns = {
        {"KC", ioh::fitsioTForm<double>, INV_VEL_UNIT},
        {"KKERNEL", ioh::fitsioTForm<double>, INV_VEL_UNIT},
        {"KGUIDE", ioh::fitsioTForm<double>, INV_VEL_UNIT},
        {"CORRECTION", ioh::fitsioTForm<double>, ""}
    };

    int Nk = meta.nklin + meta.nklog;
    char extname[] = "KCORRECTION";
    fits_file.createTable(Nk, columns, extname);
    fits_file.check();
    fits_file.writeColumn("KC", Nk, kcenters);
    fits_file.writeColumn("KKERNEL", Nk, kkernel);
    fits_file.writeColumn("KGUIDE", Nk, kguide);
    fits_file.writeColumn("CORRECTION", Nk, kcorrection);
    fits_file.check();
    fits_file.flush();
}

void SQLookupTableFile::readKbinCorrection(double *kcorrection) {
    const int ncolumns = 4;
    char extname[] = "KCORRECTION";
    fits_file.moveToHdu(extname);
    fits_file.check();

    int colnum = 4;  // Correction is the 4th column
    long nelements = meta.nklin + meta.nklog;
    fits_file.readColumn(colnum, nelements, kcorrection);
    fits_file.check();
}
}


