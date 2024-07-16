#ifndef CONTMARG_FILE_H
#define CONTMARG_FILE_H

#include <algorithm>
#include <cstdio>
#include <memory>
#include <unordered_map>

#include "core/omp_manager.hpp"
#include "io/io_helper_functions.hpp"

namespace ioh {
    class ContMargFile {
    public:
        ContMargFile(const std::string &base) : fbase(base) {};

        std::string write(double *data, int N, long int targetid) {
            FILE *fptr;
            std::string fname = _openFile(targetid, fptr);

            size_t Min = N * N,
                   Mout = fwrite(data, sizeof(double), Min, fptr);

            if (Min != Mout)
                throw std::runtime_error("ERROR in ContMargFile::write");

            fclose(fptr);

            return std::move(fname);
        }

        void read(const char *fname, int N, double *out) {
            FILE *fptr = open_file(fname, "rb");

            size_t Min = N * N, 
                   Mout = fread(out, sizeof(double), Min, fptr);
            if (Min != Mout)
                throw std::runtime_error("ERROR in ContMargFile::read::fread");
        }

    private:
        std::string fbase;

        std::string _openFile(long int targetid, FILE *fptr) {
            std::string fname = fbase + "/qu3d-rmat-" + std::to_string(targetid) + ".dat";
            fptr = open_file(fname.c_str(), "wb");
            return std::move(fname);
        }
    };

    extern std::unique_ptr<ContMargFile> continuumMargFileHandler;
}

#endif
