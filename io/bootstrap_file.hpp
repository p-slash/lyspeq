#ifndef BOOTSTRAP_FILE_H
#define BOOTSTRAP_FILE_H

#include <string>
#include <memory>

#include <fitsio.h>

#if defined(ENABLE_MPI)
#include "mpi.h" 
#endif

namespace ioh {
class BootstrapChunksFile
{
public:
    BootstrapChunksFile(const std::string &base, int thispe);
    ~BootstrapFile() { fits_close_file(fits_file, &status); };

    void writeChunk(
        const double *pk, const double *fisher, int ndim,
        int fisher_index_start, long id, double z_qso);
private:
    int status;
    fitsfile *fits_file;

    void _checkStatus();
}


#if defined(ENABLE_MPI)
// Saves results to outdir/bootresults.dat
// Fisher matrix is compressed, only saved upper 2Nk diagonals.
// This is the 3 diagonal when FISHER_OPTIMIZATION is on.
class BootstrapFile
{
    MPI_File bootfile;
    int nkbins, nzbins, pe,
        nkzbins, ndiags, cf_size,
        elems_count;
    // First bins::TOTAL_KZ_BINS elements are the power spectrum
    double *data_buffer;

public:
    BootstrapFile(const std::string &base, int nk, int nz, int thispe);
    ~BootstrapFile();

    void writeBoot(const double *pk, const double *fisher);
};

extern std::unique_ptr<BootstrapFile> boot_saver;
#endif

}

#endif
