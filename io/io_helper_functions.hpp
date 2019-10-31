#ifndef IO_HELPER_FUNCTIONS_H
#define IO_HELPER_FUNCTIONS_H

#include <cstdio>
#include <fstream>

#include <complex>
#include <vector>

namespace ioh
{
    bool file_exists(const char *fname);

    template <class T>
    T* copyArrayAlloc(const T* source, int size);

    FILE * open_file(const char *fname, const char *read_write);
    // Open binary file if binary='b'
    // T can be ifstream, ofstream or fstream.
    template <class T>
    T open_fstream(const char *fname, char binary='0');

    // Allocates list_values, must dellocate manually!
    // Returns number of elements
    template <class T>
    int readList(const char *fname, std::vector<T> &list_values);
}

#endif
