#include "io/io_helper_functions.hpp"

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
        throw std::runtime_error("Cannot open file");
    }

    return file_to_read_write;
}

std::fstream ioh::open_file(const char *fname)
{
    std::fstream file_fs;
    file_fs.open(fname);

    if (!file_fs)
    {
        fprintf(stderr, "ERROR FSTREAM: %s\n", fname);
        throw std::runtime_error("Cannot open file");
    }

    return file_fs;
}


template <class T>
int ioh::readList(const char *fname, std::vector<T> &list_values)
{
    int nr;

    std::fstream toRead = ioh::open_file(fname);
    
    toRead >> nr;

    list_values.reserve(nr);
    
    T tmp;
    while (toRead >> tmp)
        list_values.push_back(tmp);

    toRead.close();

    return nr;
}

template int ioh::readList(const char *fname, std::vector<int> &list_values);
template int ioh::readList(const char *fname, std::vector<std::string> &list_values);



