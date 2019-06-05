#include "io/io_helper_functions.hpp"

#include <cstdio>
#include <algorithm>
#include <new>

bool file_exists(const char *fname)
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

template <class T>
T* copyArrayAlloc(const T* source, int size)
{
    T* target = new T[size];
    std::copy(  &source[0], \
                &source[0] + size, \
                &target[0]);

    return target;
}
template int* copyArrayAlloc<int>(const int *source, int size);
template double* copyArrayAlloc<double>(const double *source, int size);

void copyArray(const std::complex<double> *source, std::complex<double> *target, long size)
{
    std::copy(  &source[0], \
                &source[0] + size, \
                &target[0]);
}

void copyArray(const double *source, double *target, long size)
{
    std::copy(  &source[0], \
                &source[0] + size, \
                &target[0]);
}

FILE * open_file(const char *fname, const char *read_write)
{
    FILE *file_to_read_write;
    char buffer[15];

    if (read_write[0] == 'r')
    {
        sprintf(buffer, "READING FROM");
    }
    else
    {
        sprintf(buffer, "WRITING TO");
    }

    file_to_read_write = fopen(fname, read_write);

    if (file_to_read_write == NULL)
    {
        printf("ERROR: %s FILE: %s\n", buffer, fname);
        
        throw "ERROR: FILE OPERATION";
    }

    printf("%s file: %s\n", buffer, fname);
    fflush(stdout);

    return file_to_read_write;
}

//void writeArrayToFile(const char *fname, fftw_complex *source, long size);
//void readArrayFromFile(const char *fname, fftw_complex *target);
//fftw_complex * fftw_alloc_from_file(const char *fname, int &n_bin);
//fftw_complex * fftw_alloc_from_file(const char *fname);
/*
void writeArrayToFile(const char *fname, fftw_complex *source, long long int size)
{
    FILE *toWrite;

    toWrite = open_file(fname, "wb");

    fwrite(&size, sizeof(long long int), 1, toWrite);

    fwrite(source, sizeof(fftw_complex), size, toWrite);

    fclose(toWrite);

    printf("fftw_complex array saved as %s\n", fname);
}

long long int readArrayFromFile(const char *fname, fftw_complex *target)
{
    FILE *toRead;
    long long int size;

    toRead = open_file(fname, "rb");
    
    fread(&size, sizeof(long long int), 1, toRead);

    fread(target, sizeof(fftw_complex), size, toRead);

    fclose(toRead);

    printf("Reading fftw_complex from file %s: Done!\n", fname);
}

fftw_complex * fftw_alloc_from_file(const char *fname, long long int &n_bin)
{
    FILE *toRead;
    long long int size;
    fftw_complex *result;

    toRead = open_file(fname, "rb");

    fread(&size, sizeof(long long int), 1, toRead);
    n_bin = size;
    size = size * size * (size / 2 + 1);
    
    result = fftw_alloc_complex(size);

    if (result == NULL)
    {
        printf("ERROR: CANNOT ALLOCATE MEMORY FOR FFTW_COMPLEX FROM FILE!\n");
        throw std::bad_alloc();
    }
    
    fread(result, sizeof(fftw_complex), size, toRead);

    fclose(toRead);

    printf("Allocating fftw_complex from file %s: Done!\n", fname);
    
    return result;
}

fftw_complex * fftw_alloc_from_file(const char *fname)
{
    long long int temp;

    return fftw_alloc_from_file(fname, temp);
}
*/
