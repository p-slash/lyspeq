#include "io/bootstrap_file.hpp"
#include "mathtools/matrix_helper.hpp"

#include <string>
#include <algorithm>
#include <stdexcept>
#include <memory>


ioh::BootstrapChunksFile::BootstrapChunksFile(
        const std::string &base, int thispe
) {
    status = 0;
    std::string out_fname =
        "!" + base + "-bootchunks-" + std::to_string(thispe) + ".fits";
    fits_create_file(&fits_file, out_fname.c_str(), &status);
    _checkStatus();
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
    _checkStatus();

    fits_write_key(fits_file, TLONG, "TARGETID", &id, nullptr, &status);
    fits_write_key(fits_file, TDOUBLE, "ZQSO", &z_qso, nullptr, &status);
    fits_write_key(fits_file, TDOUBLE, "RA", &ra, nullptr, &status);
    fits_write_key(fits_file, TDOUBLE, "DEC", &dec, nullptr, &status);
    fits_write_key(fits_file, TINT, "NQDIM", &ndim, nullptr, &status);
    fits_write_key(
        fits_file, TINT, "ISTART", &fisher_index_start, nullptr, &status);

    fits_write_img(
        fits_file, TDOUBLE, 1, cf_size, (void *) data_buffer, &status);
    _checkStatus();
}


void ioh::BootstrapChunksFile::_checkStatus() {
    if (status == 0)
        return;

    char fits_msg[50];
    fits_get_errstatus(status, fits_msg);
    std::string error_msg = std::string("FITS ERROR ") + std::string(fits_msg);

    throw std::runtime_error(error_msg);
}


#if defined(ENABLE_MPI)
std::unique_ptr<ioh::BootstrapFile> ioh::boot_saver;

ioh::BootstrapFile::BootstrapFile(const std::string &base, int nk, int nz, int thispe)
: nkbins(nk), nzbins(nz), pe(thispe)
{
    int r=0;
    std::string out_fname = base + "-bootresults.dat";

    r += MPI_File_open(MPI_COMM_WORLD, out_fname.c_str(), 
        MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &bootfile);

    nkzbins = nk*nz;
    // #ifdef FISHER_OPTIMIZATION
    // ndiags  = 3;  // 2*nkbins
    // cf_size = 3*nkzbins-nkbins-1;
    // #else
    ndiags  = 2*nkbins;
    cf_size = nkzbins*ndiags - (ndiags*(ndiags-1))/2;
    // #endif
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
        // #ifdef FISHER_OPTIMIZATION
        // if (d == 2) d = nkbins;
        // #endif
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
