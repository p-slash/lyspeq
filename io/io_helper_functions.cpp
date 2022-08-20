#include "io/io_helper_functions.hpp"
#include "core/matrix_helper.hpp"

#include <iostream>
#include <string>

#include <algorithm>
#include <new>
#include <stdexcept>

bool ioh::file_exists(const char *fname)
{
    FILE *toCheck;

    toCheck = fopen(fname, "r");

    if (toCheck == NULL)
    {
        return false;
    }

    fclose(toCheck);
    return true;
}

void ioh::create_tmp_file(char *fname, const char *TMP_FOLDER)
{
    int s;
    sprintf(fname, "%s/tmplyspeqfileXXXXXX", TMP_FOLDER);

    s = mkstemp(fname);

    if (s == -1)    throw std::runtime_error("tmp filename");
}

template <class T>
T* ioh::copyArrayAlloc(const T* source, int size)
{
    T* target = new T[size];
    std::copy(&source[0], &source[0] + size, &target[0]);

    return target;
}
template int* ioh::copyArrayAlloc<int>(const int *source, int size);
template double* ioh::copyArrayAlloc<double>(const double *source, int size);

FILE * ioh::open_file(const char *fname, const char *read_write)
{
    FILE *file_to_read_write;

    file_to_read_write = fopen(fname, read_write);

    if (file_to_read_write == NULL)
    {
        char err_buf[500];
        sprintf(err_buf, "Cannot open file %s.", fname);

        fprintf(stderr, "ERROR FILE with %s: %s\n", read_write, fname);
        throw std::runtime_error(err_buf);
    }

    return file_to_read_write;
}

template <class T>
T ioh::open_fstream(const char *fname, char binary)
{
    T file_fs;
    if (binary=='b')
        file_fs.open(fname, std::ios::binary);
    else
        file_fs.open(fname);

    if (!file_fs)
    {
        char err_buf[500];
        sprintf(err_buf, "Cannot open file %s.", fname);
        
        fprintf(stderr, "ERROR FSTREAM: %s\n", fname);
        throw std::runtime_error(err_buf);
    }

    return file_fs;
}

template std::ifstream ioh::open_fstream<std::ifstream>(const char*, char);
template std::ofstream ioh::open_fstream<std::ofstream>(const char*, char);
template std::fstream  ioh::open_fstream<std::fstream>(const char*, char);

template <class T>
int ioh::readList(const char *fname, std::vector<T> &list_values)
{
    int nr, ic=0;

    std::ifstream toRead = ioh::open_fstream<std::ifstream>(fname);
    
    toRead >> nr;

    list_values.reserve(nr);
    
    T tmp;
    while (toRead >> tmp && ic < nr)
    {
        list_values.push_back(tmp);
        ++ic;
    }

    toRead.close();

    return nr;
}

template int ioh::readList(const char *fname, std::vector<int> &list_values);
template int ioh::readList(const char *fname, std::vector<std::string> &list_values);

int ioh::readListRdv(const char *fname, std::vector<std::pair<int, double>> &list_values)
{
    int nr, ic=0;

    std::ifstream toRead = ioh::open_fstream<std::ifstream>(fname);
    
    toRead >> nr;

    list_values.reserve(nr);
    
    int c1; double c2;

    while (toRead.good() && ic < nr)
    {
        toRead >> c1 >> c2;
        list_values.push_back(std::make_pair(c1, c2));
        ++ic;
    }

    toRead.close();

    return nr;
}

/* 
-----------------------------------------------
------------- BootstrapFile Class -------------
----------------------------------------------- 
*/

#if defined(ENABLE_MPI)
namespace ioh
{
    BootstrapFile *boot_saver = NULL;
}

ioh::BootstrapFile::BootstrapFile(const char *outdir, const char *base, int nk, int nz, int thispe)
: nkbins(nz), nzbins(nz), nkzbins(nk*nz), pe(thispe)
{
    int r=0;
    std::ostringstream oss_fname(outdir, std::ostringstream::ate);
    oss_fname << "/"<<base<<"-bootresults.dat";

    r += MPI_File_open(MPI_COMM_WORLD, oss_fname.str().c_str(), 
        MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &bootfile);

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

    data_buffer = new double[elems_count];
}

ioh::BootstrapFile::~BootstrapFile() { MPI_File_close(&bootfile); delete [] data_buffer; }

void ioh::BootstrapFile::writeBoot(const double *pk, const double *fisher)
{
    int r=0;

    std::copy(pk, pk + nkzbins, data_buffer);

    double *v = data_buffer+nkzbins;
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
    r += MPI_File_write_at_all(bootfile, offset, data_buffer,
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






