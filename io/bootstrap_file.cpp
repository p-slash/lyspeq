#include "io/bootstrap_file.hpp"
#include "mathtools/matrix_helper.hpp"

#include <algorithm>
#include <stdexcept>

/* void _createEmptyHdu(fitsfile *fits_file) {
    int status = 0, bitpix = SHORT_IMG, naxis = 0;
    long *naxes = nullptr;
    fits_create_img(fits_file, bitpix, naxis, naxes, &status);
    ioh::checkFitsStatus(status);
} */

std::unique_ptr<ioh::BootstrapFile> ioh::boot_saver;


std::string ioh::saveBootstrapRealizations(
        const std::string &base, const double *allpowers, const double *invfisher,
        unsigned int nboots, int nk, int nz, bool fastbootstrap, const char *comment
) {
    int status = 0, bitpix = DOUBLE_IMG;
    long naxis = 2, size = nboots * nk * nz, naxes[2] = { nk * nz, nboots };
    std::string out_fname = "!" + base + "-bootstrap-realizations.fits";

    auto fitsfile_ptr = ioh::create_unique_fitsfile_ptr(out_fname);
    fitsfile *fits_file = fitsfile_ptr.get();

    fits_create_img(fits_file, bitpix, naxis, naxes, &status);
    ioh::checkFitsStatus(status);

    fits_update_key_str(fits_file, "EXTNAME", "REALIZATIONS", nullptr, &status);
    fits_write_key(fits_file, TUINT, "NBOOTS", &nboots, nullptr, &status);
    fits_write_key(fits_file, TINT, "NK", &nk, nullptr, &status);
    fits_write_key(fits_file, TINT, "NZ", &nz, nullptr, &status);
    fits_write_key(
        fits_file, TLOGICAL, "FASTBOOT", &fastbootstrap, nullptr, &status);
    if (comment != nullptr)
        fits_write_comment(fits_file, comment, &status);

    fits_write_img(fits_file, TDOUBLE, 1, size, (void *) allpowers, &status);
    ioh::checkFitsStatus(status);

    // Write inverse fisher solver
    size = naxes[0] * naxes[0];
    naxes[1] = naxes[0];
    fits_create_img(fits_file, bitpix, naxis, naxes, &status);
    ioh::checkFitsStatus(status);

    fits_update_key_str(fits_file, "EXTNAME", "SOLV_INVF", nullptr, &status);
    fits_write_key(fits_file, TINT, "NK", &nk, nullptr, &status);
    fits_write_key(fits_file, TINT, "NZ", &nz, nullptr, &status);
    /*
    fits_write_key(
        fits_file, TDOUBLE, "RTCORNER", &invfisher[naxes[0] - 1],
        "Row 0, last column real value.", &status);
    double pivotF = 9e9;
    fits_write_key(
        fits_file, TDOUBLE, "RTPIVOT", &pivotF,
        "Pivot value to find orientation.", &status);
    invfisher[naxes[0] - 1] = pivotF;
    */
    fits_write_key(
        fits_file, TLOGICAL, "FASTBOOT", &fastbootstrap, nullptr, &status);
    fits_write_comment(
        fits_file,
        "Inverse Fisher solver matrix. "
        "Not necessarily equal to the inverse Fisher "
        "or symmetric.",
        &status);
    if (comment != nullptr)
        fits_write_comment(fits_file, comment, &status);

    fits_write_img(fits_file, TDOUBLE, 1, size, (void *) invfisher, &status);
    ioh::checkFitsStatus(status);

    return out_fname;
}


void ioh::readBootstrapRealizations(
        const std::string &fname,
        std::unique_ptr<double[]> &allpowers,
        std::unique_ptr<double[]> &invfisher,
        unsigned int &nboots, int &nk, int &nz, bool &fastbootstrap
) {
    int status = 0, nfound, fboot;
    long naxes[2];
    double nullval = 0.;
    auto fitsfile_ptr = ioh::open_unique_fitsfile_ptr(fname, READONLY);
    fitsfile *fits_file = fitsfile_ptr.get();

    char extname[] = "REALIZATIONS";
    fits_movnam_hdu(fits_file, IMAGE_HDU, extname, 0, &status);
    ioh::checkFitsStatus(status);

    fits_read_keys_lng(fits_file, "NAXIS", 1, 2, naxes, &nfound, &status);
    fits_read_key(fits_file, TUINT, "NBOOTS", &nboots, nullptr, &status);
    fits_read_key(fits_file, TINT, "NK", &nk, nullptr, &status);
    fits_read_key(fits_file, TINT, "NZ", &nz, nullptr, &status);
    fits_read_key(
        fits_file, TLOGICAL, "FASTBOOT", &fboot, nullptr, &status);
    ioh::checkFitsStatus(status);
    fastbootstrap = fboot == 1;

    long size = naxes[0] * naxes[1];
    allpowers = std::make_unique<double[]>(size);
    fits_read_img(
        fits_file, TDOUBLE, 1, size, &nullval, allpowers.get(), nullptr, &status);


    char extname2[] = "SOLV_INVF";
    fits_movnam_hdu(fits_file, IMAGE_HDU, extname2, 0, &status);
    fits_read_keys_lng(fits_file, "NAXIS", 1, 2, naxes, &nfound, &status);

    size = naxes[0] * naxes[1];
    invfisher = std::make_unique<double[]>(size);
    fits_read_img(
        fits_file, TDOUBLE, 1, size, &nullval, invfisher.get(), nullptr, &status);

    ioh::checkFitsStatus(status);
}

ioh::BootstrapChunksFile::BootstrapChunksFile(
        const std::string &base, int thispe
) {
    status = 0;
    std::string out_fname =
        "!" + base + "-bootchunks-" + std::to_string(thispe) + ".fits";
    fitsfile_ptr = create_unique_fitsfile_ptr(out_fname);
    fits_file = fitsfile_ptr.get();
}


void ioh::BootstrapChunksFile::writeChunk(
        const double *pk, const double *nk, const double *tk,
        const double *fisher, int ndim,
        int fisher_index_start, long id, double z_qso,
        double ra, double dec
) {
    int bitpix = DOUBLE_IMG;
    long naxis = 1;
    int cf_size = 3 * ndim + (ndim * (ndim + 1)) / 2;
    auto data_buffer_ptr = std::make_unique<double[]>(cf_size);
    double *data_buffer = data_buffer_ptr.get();
    std::copy(pk, pk + ndim, data_buffer);
    std::copy(nk, nk + ndim, data_buffer + ndim);
    std::copy(tk, tk + ndim, data_buffer + 2 * ndim);

    double *v = data_buffer + 3 * ndim;
    for (int d = 0; d < ndim; ++d)
    {
        mxhelp::getDiagonal(fisher, ndim, d, v);
        v += ndim - d;
    }

    long naxes[1] = {cf_size};
    fits_create_img(fits_file, bitpix, naxis, naxes, &status);
    ioh::checkFitsStatus(status);

    fits_write_key(fits_file, TLONG, "TARGETID", &id, nullptr, &status);
    fits_write_key(fits_file, TDOUBLE, "ZQSO", &z_qso, nullptr, &status);
    fits_write_key(fits_file, TDOUBLE, "RA", &ra, nullptr, &status);
    fits_write_key(fits_file, TDOUBLE, "DEC", &dec, nullptr, &status);
    fits_write_key(fits_file, TINT, "NQDIM", &ndim, nullptr, &status);
    fits_write_key(
        fits_file, TINT, "ISTART", &fisher_index_start, nullptr, &status);

    fits_write_img(
        fits_file, TDOUBLE, 1, cf_size, (void *) data_buffer, &status);
    ioh::checkFitsStatus(status);
}


#if defined(ENABLE_MPI)
ioh::BootstrapFile::BootstrapFile(const std::string &base, int nk, int nz, int thispe)
: nkbins(nk), nzbins(nz), pe(thispe)
{
    int r=0;
    std::string out_fname = base + "-bootresults.dat";

    r += MPI_File_open(MPI_COMM_WORLD, out_fname.c_str(), 
        MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &bootfile);

    nkzbins = nk * nz;
    ndiags  = 3 * nkbins;
    cf_size = nkzbins*ndiags - (ndiags*(ndiags-1))/2;

    elems_count = cf_size+nkzbins;

    if (pe == 0)
    {
        r += MPI_File_write(bootfile, &nkbins, 1, MPI_INT, MPI_STATUS_IGNORE);
        r += MPI_File_write(bootfile, &nzbins, 1, MPI_INT, MPI_STATUS_IGNORE);
        r += MPI_File_write(bootfile, &ndiags, 1, MPI_INT, MPI_STATUS_IGNORE);
    }
    // #else
    // bootfile = ioh::open_file(oss_fname.str().c_str(), "wb");

    // r += fwrite(&nkbins, sizeof(int), 1, bootfile)-1;
    // r += fwrite(&nzbins, sizeof(int), 1, bootfile)-1;
    // r += fwrite(&ndiags, sizeof(int), 1, bootfile)-1;
    // #endif

    if (r != 0) 
        throw std::runtime_error("Bootstrap file first Nk write.");

    data_buffer = new double[elems_count];
}

ioh::BootstrapFile::~BootstrapFile()
{
    MPI_File_close(&bootfile);
    delete [] data_buffer;
}

void ioh::BootstrapFile::writeBoot(const double *pk, const double *fisher)
{
    int r=0;

    std::copy(pk, pk + nkzbins, data_buffer);

    double *v = data_buffer+nkzbins;
    for (int d = 0; d < ndiags; ++d)
    {
        mxhelp::getDiagonal(fisher, nkzbins, d, v);
        v += nkzbins-d;
    }

    // Offset is the header first three integer plus shift by PE
    MPI_Offset offset = 3*sizeof(int) + pe*elems_count*sizeof(double);
    r += MPI_File_write_at_all(bootfile, offset, data_buffer,
        elems_count, MPI_DOUBLE, MPI_STATUS_IGNORE);
    // r += fwrite(data_buffer, sizeof(double), elems_count, bootfile)-elems_count;
    if (r != 0)
        throw std::runtime_error("Bootstrap write one results.");
}
#endif
