#if defined(ENABLE_MPI)

#include "io/bootstrap_file.hpp"
#include "mathtools/matrix_helper.hpp"

#include <string>
#include <algorithm>
#include <stdexcept>

std::unique_ptr<ioh::BootstrapFile> ioh::boot_saver;

ioh::BootstrapFile::BootstrapFile(const std::string &base, int nk, int nz, int thispe)
: nkbins(nk), nzbins(nz), pe(thispe)
{
    int r=0;
    std::string out_fname = base + "-bootresults.dat";

    r += MPI_File_open(MPI_COMM_WORLD, out_fname.c_str(), 
        MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &bootfile);

    nkzbins = nk*nz;
    #ifdef FISHER_OPTIMIZATION
    ndiags  = 3;
    cf_size = 3*nkzbins-nkbins-1;
    #else
    ndiags  = 2*nkbins;
    cf_size = nkzbins*ndiags - (ndiags*(ndiags-1))/2;
    #endif
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

    data_buffer = std::make_unique<double[]>(elems_count);
}

ioh::BootstrapFile::~BootstrapFile()
{
    MPI_File_close(&bootfile);
}

void ioh::BootstrapFile::writeBoot(const double *pk, const double *fisher)
{
    int r=0;

    std::copy(pk, pk + nkzbins, data_buffer.get());

    double *v = data_buffer.get()+nkzbins;
    for (int d = 0; d < ndiags; ++d)
    {
        #ifdef FISHER_OPTIMIZATION
        if (d == 2) d = nkbins;
        #endif
        mxhelp::getDiagonal(fisher, nkzbins, d, v);
        v += nkzbins-d;
    }

    // Offset is the header first three integer plus shift by PE
    MPI_Offset offset = 3*sizeof(int) + pe*elems_count*sizeof(double);
    r += MPI_File_write_at_all(bootfile, offset, data_buffer.get(),
        elems_count, MPI_DOUBLE, MPI_STATUS_IGNORE);
    // r += fwrite(data_buffer, sizeof(double), elems_count, bootfile)-elems_count;
    if (r != 0)
        throw std::runtime_error("Bootstrap write one results.");
}
#endif


// void ioh::BootstrapFile::writeBoot(int thingid, double *pk, double *fisher)
// {
//     double *v = comp_fisher;
//     for (int d = 0; d < NDIAGS; ++d)
//     {
//         #ifdef FISHER_OPTIMIZATION
//         if (d == 2) d = nkbins;
//         #endif
//         mxhelp::getDiagonal(fisher, nkzbins, d, v);
//         v += nkzbins-d;
//     }

//     int r = fwrite(&thingid, sizeof(int), 1, bootfile);
//     r+=fwrite(comp_fisher, sizeof(double), CF_SIZE, bootfile);
//     r+=fwrite(pk, sizeof(double), nkzbins, bootfile);

//     if (r != 1+CF_SIZE+nkzbins)
//         throw std::runtime_error("Bootstrap write one results.");
// }

// MPI_Datatype etype;

// MPI_Aint pkindex, fisherindex;
// MPI_Type_extent(MPI_INT, &pkindex);
// MPI_Type_extent(MPI_DOUBLE, &fisherindex);
// int blocklengths[] = {1, nkzbins, FISHER_SIZE};
// MPI_Datatype types[] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE};
// MPI_Aint offsets[] = { 0, pkindex,  nkzbins*fisherindex + pkindex};

// MPI_Type_create_struct(3, blocklengths, offsets, types, &etype);
// MPI_Type_commit(&etype);

// MPI_File_open(MPI_COMM_WORLD, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
// // thing id (int), pk (double*N), Fisher (double*N*N) 
// MPI_Offset offset = sizeof(int) + (nkzbins+FISHER_SIZE)*sizeof(double);
// int nprevious_sp = 0;
// for (int peno = 0; peno < process::this_pe; ++peno)
//     nprevious_sp += nospecs_perpe[peno];
// offset *= nprevious_sp;

// MPI_File_set_view(fh, disp, etype, etype, "native", MPI_INFO_NULL);






