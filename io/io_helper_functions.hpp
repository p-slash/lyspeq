#ifndef IO_HELPER_FUNCTIONS_H
#define IO_HELPER_FUNCTIONS_H

#include <complex>

namespace ioh
{
    bool file_exists(const char *fname);

    template <class T>
    T* copyArrayAlloc(const T* source, int size);

    FILE * open_file(const char *fname, const char *read_write);
}

#endif
