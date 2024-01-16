#ifndef BOOTSTRAP_FILE_H
#define BOOTSTRAP_FILE_H

#include <memory>
#include <string>

#include <fitsio.h>

#if defined(ENABLE_MPI)
#include "mpi.h" 
#endif

namespace ioh {
void checkFitsStatus(int status);
void saveBootstrapRealizations(
    const std::string &base, const double *allpowers, const double *invfisher,
    unsigned int nboots, int nk, int nz, bool fastbootstrap,
    const char *comment=nullptr
);

void readBootstrapRealizations(
    const std::string &fname,
    std::unique_ptr<double[]> &allpowers,
    std::unique_ptr<double[]> &invfisher,
    unsigned int &nboots, int &nk, int &nz, bool &fastbootstrap
);

class BootstrapChunksFile
{
public:
    BootstrapChunksFile(const std::string &base, int thispe);
    ~BootstrapChunksFile() { fits_close_file(fits_file, &status); };

    void writeChunk(
        const double *pk, const double *nk, const double *tk,
        const double *fisher, int ndim,
        int fisher_index_start, long id, double z_qso,
        double ra, double dec);
private:
    int status;
    fitsfile *fits_file;
};


#if defined(ENABLE_MPI)
// Saves results to outdir/bootresults.dat
// Fisher matrix is compressed, only saved upper 3Nk diagonals.
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
