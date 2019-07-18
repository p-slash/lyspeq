#include "io/io_helper_functions.hpp"

#include <cstdio>
#include <algorithm>
#include <new>

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
        fprintf(stderr, "ERROR FILE with %s: %s\n", read_write, fname);
        throw "ERROR: FILE OPERATION";
    }

    return file_to_read_write;
}


