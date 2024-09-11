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

void ioh::create_tmp_file(char *fname, const std::string &TMP_FOLDER, size_t n)
{
    int s;
    snprintf(fname, n, "%s/tmplyspeqfileXXXXXX", TMP_FOLDER.c_str());

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
        std::string err_buf = std::string("Cannot open file: ") + std::string(fname);

        fprintf(stderr, "ERROR FILE with %s: %s\n", read_write, fname);
        throw std::runtime_error(err_buf);
    }

    return file_to_read_write;
}

// template <class T>
// T ioh::open_fstream(const char *fname, char binary)
// {
//     T file_fs;
//     if (binary=='b')
//         file_fs.open(fname, std::ios::binary);
//     else
//         file_fs.open(fname);

//     if (!file_fs)
//     {
//         std::string err_buf = "Cannot open file " + fname;
//         std::cerr << err_buf << std::endl;
//         throw std::runtime_error(err_buf);
//     }

//     return file_fs;
// }

// template std::ifstream ioh::open_fstream<std::ifstream>(const char*, char);
// template std::ofstream ioh::open_fstream<std::ofstream>(const char*, char);
// template std::fstream  ioh::open_fstream<std::fstream>(const char*, char);

template <class T>
T ioh::open_fstream(const std::string &fname, char binary)
{
    T file_fs;
    if (binary=='b')
        file_fs.open(fname, std::ios::binary);
    else
        file_fs.open(fname);

    if (!file_fs)
    {
        std::string err_buf = "Cannot open file " + fname;
        std::cerr << err_buf << std::endl;
        throw std::runtime_error(err_buf);
    }

    return file_fs;
}

template std::ifstream ioh::open_fstream<std::ifstream>(const std::string&, char);
template std::ofstream ioh::open_fstream<std::ofstream>(const std::string&, char);
template std::fstream  ioh::open_fstream<std::fstream>(const std::string&, char);

template <class T>
int ioh::readList(const char *fname, std::vector<T> &list_values, bool hdr)
{
    int nr = 2000000, ic = 0;

    std::ifstream toRead = ioh::open_fstream<std::ifstream>(fname);

    if (hdr) {
        toRead >> nr;
        list_values.reserve(nr);
    }

    T tmp;
    while (toRead.good() && toRead >> tmp && ic < nr) {
        list_values.push_back(tmp);
        ++ic;
    }

    toRead.close();

    return nr;
}

template int ioh::readList(const char *fname, std::vector<int> &list_values, bool hdr);
template int ioh::readList(const char *fname, std::vector<long> &list_values, bool hdr);
template int ioh::readList(const char *fname, std::vector<std::string> &list_values, bool hdr);

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

