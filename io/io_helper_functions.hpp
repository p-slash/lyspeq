#ifndef IO_HELPER_FUNCTIONS_H
#define IO_HELPER_FUNCTIONS_H

#include <cstdio>
#include <fstream>

#include <complex>
#include <vector>

namespace ioh
{
    bool file_exists(const char *fname);

    void create_tmp_file(char *fname, const char *TMP_FOLDER);

    template <class T>
    T* copyArrayAlloc(const T* source, int size);

    FILE * open_file(const char *fname, const char *read_write);
    // Open binary file if binary='b'
    // T can be ifstream, ofstream or fstream.
    template <class T>
    T open_fstream(const char *fname, char binary='0');

    // Returns number of elements
    template <class T>
    int readList(const char *fname, std::vector<T> &list_values);

    // Reads for 2 column separated by space into vector of pairs
    // Returns number of elements
    int readListRdv(const char *fname, std::vector<std::pair<int, double>> &list_values);

    // Saves results to outdir/bootresults-{this_pe}.dat
    // Fisher matrix is compressed, only saved upper 2Nk diagonals.
    // This is the 3 diagonal when FISHER_OPTIMIZATION is on.
    class BootstrapFile
    {
        FILE *bootfile;
    public:
        BootstrapFile(const char *outdir);
        ~BootstrapFile();

        void writeBoot(int thingid, double *pk, double *fisher);
    };

    extern BootstrapFile *boot_saver;
}

#endif
