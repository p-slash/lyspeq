#if defined(ENABLE_MPI)

#ifndef BOOTSTRAP_FILE_H
#define BOOTSTRAP_FILE_H

#include <string>
#include <memory>
#include "mpi.h" 

namespace ioh
{

// Saves results to outdir/bootresults.dat
// Fisher matrix is compressed, only saved upper 2Nk diagonals.
// This is the 3 diagonal when FISHER_OPTIMIZATION is on.
class BootstrapFile
{
    MPI_File bootfile;
    int nkbins, nzbins, nkzbins, pe,
        ndiags, cf_size, elems_count;
    // First bins::TOTAL_KZ_BINS elements are the power spectrum
    std::unique_ptr<double[]> data_buffer;

public:
    BootstrapFile(const std::string &base, int nk, int nz, int thispe);
    ~BootstrapFile();

    void writeBoot(const double *pk, const double *fisher);
};

extern std::unique_ptr<BootstrapFile> boot_saver;

}
#endif

#endif
